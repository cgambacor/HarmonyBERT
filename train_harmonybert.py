"""
train_harmonybert.py
=====================
Pipeline de pre-entrenamiento y fine-tuning de HarmonyBERT.

Estrategia de masking: COMPOUND WORD MASKING
────────────────────────────────────────────
A diferencia del MLM estándar que enmascara atributos individuales,
aquí se aplica el masking a nivel de POSICIÓN COMPLETA (compound word):
cuando una posición t es seleccionada para masking, los 9 atributos
de esa posición se enmascaran simultáneamente. Esto obliga al modelo
a reconstruir el "token musical completo" en lugar de un solo campo.

  - 15% de posiciones seleccionadas para masking
  - De las seleccionadas:
      80% → todos sus 9 atributos se reemplazan por [MASK]
      10% → todos sus 9 atributos se reemplazan por valores aleatorios
      10% → se dejan sin cambio (el modelo no sabe cuáles)

Adicionalmente, el atributo Armonía puede recibir masking individual
extra (harmony_boost) para reforzar el aprendizaje armónico.

Dependencias:
  pip install torch transformers tqdm

Uso (pre-entrenamiento):
  python train_harmonybert.py \\
      --train_file corpus/octabeat_train.jsonl \\
      --val_file   corpus/octabeat_val.jsonl   \\
      --vocab_file corpus/vocabs.json          \\
      --output_dir checkpoints/harmonybert     \\
      --epochs 50 --batch_size 32 --lr 1e-4

Uso (fine-tuning armonía):
  python train_harmonybert.py \\
      --task harmony_pred \\
      --pretrained_model checkpoints/harmonybert/best \\
      --train_file corpus/octabeat_train.jsonl \\
      --output_dir checkpoints/harmonybert_ft  \\
      --epochs 20 --lr 5e-5
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

try:
    from transformers import get_linear_schedule_with_warmup
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False

from harmonybert_model import (
    HarmonyBERTConfig,
    HarmonyBERTForMLM,
    HarmonyBERTForHarmonyPrediction,
    count_parameters,
    model_summary,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ===========================================================================
# 1.  DATASET
# ===========================================================================

class OctaBeat9Dataset(Dataset):
    """
    Carga secuencias OctaBeat-9 desde un fichero JSONL.
    Cada línea: {"piece_id": str, "tokens": [[9 int], ...], "metadata": {...}}

    Cada ítem devuelve:
        input_ids      : (max_seq_len, 9)  dtype=torch.long
        attention_mask : (max_seq_len,)    dtype=torch.long  1=real 0=PAD
    """

    def __init__(self, jsonl_path: str, max_seq_len: int = 512):
        self.max_seq_len = max_seq_len
        self.sequences:  List[List[List[int]]] = []
        self.piece_ids:  List[str] = []

        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                toks = rec["tokens"]
                if len(toks) < 4:
                    continue
                self.sequences.append(toks)
                self.piece_ids.append(rec.get("piece_id", "?"))

        log.info(f"Dataset: {len(self.sequences):,} secuencias ← {jsonl_path}")

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        toks    = self.sequences[idx]
        T       = min(len(toks), self.max_seq_len)
        pad_len = self.max_seq_len - T

        arr = torch.tensor(toks[:T], dtype=torch.long)  # (T, 9)
        if pad_len > 0:
            arr = torch.cat([arr, torch.zeros(pad_len, 9, dtype=torch.long)], dim=0)

        mask = torch.zeros(self.max_seq_len, dtype=torch.long)
        mask[:T] = 1

        return {"input_ids": arr, "attention_mask": mask}


# ===========================================================================
# 2.  COMPOUND WORD MLM COLLATOR
# ===========================================================================
class CompoundWordMLMCollator:
    """
    Data collator oficial de la estrategia MusicBERT: Bar-Level Masking.
    
    Lógica:
    1. Agrupa la música en "Unidades" = (Número de Compás + Tipo de Atributo).
       Por ejemplo: (Compás 5, Pitch), (Compás 12, Duración).
    2. Selecciona el 15% de estas UNIDADES para enmascarar.
    3. Si una unidad es seleccionada (ej. Compás 5, Pitch), se aplica a 
       TODAS las notas de ese compás la regla 80/10/10.
    """
    
    def __init__(
        self, 
        vocab_sizes: List[int], 
        mask_prob: float = 0.15, 
        bar_attr_idx: int = 0,    # Asumimos que la columna 0 es el 'Bar'
        mask_token_id: int = 2    # El ID de tu token [MASK]
    ):
        self.vocab_sizes = vocab_sizes
        self.mask_prob = mask_prob
        self.bar_attr_idx = bar_attr_idx
        self.mask_token_id = mask_token_id
        self.n_attrs = len(vocab_sizes)

    def __call__(self, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        input_ids = torch.stack([b["input_ids"] for b in batch])  # (B, T, 9)
        attention_mask = torch.stack([b["attention_mask"] for b in batch])  # (B, T)

        B, T, N = input_ids.shape
        masked_input = input_ids.clone()
        
        # -100 le dice a PyTorch: "No calcules pérdida en estas posiciones"
        labels = torch.full((B, T, N), -100, dtype=torch.long)

        # Iteramos por cada canción del batch (lo hacemos así porque 
        # cada canción tiene números de compases diferentes)
        for b in range(B):
            # Saber hasta dónde llega la música real (ignorando el PAD del final)
            valid_len = attention_mask[b].sum().item()
            if valid_len == 0:
                continue

            # Extraemos la columna de compases de esta canción
            bars = input_ids[b, :valid_len, self.bar_attr_idx]
            unique_bars = torch.unique(bars)

            # Por cada compás único en la canción...
            for bar_val in unique_bars:
                # Encontrar TODAS las posiciones (índices) que pertenecen a este compás
                bar_indices = (bars == bar_val).nonzero(as_tuple=True)[0]

                # Evaluamos de forma independiente cada uno de los 9 atributos
                for attr_idx in range(N):
                    
                    # ¿Enmascaramos este atributo para TODO el compás? (15% prob)
                    if torch.rand(1).item() < self.mask_prob:
                        
                        # 1. Guardar las respuestas correctas en 'labels'
                        labels[b, bar_indices, attr_idx] = input_ids[b, bar_indices, attr_idx]

                        # 2. Aplicar la regla 80 / 10 / 10 a todo el bloque
                        rand_p = torch.rand(1).item()
                        
                        if rand_p < 0.80:
                            # 80%: Reemplazar todo con [MASK]
                            masked_input[b, bar_indices, attr_idx] = self.mask_token_id
                            
                        elif rand_p < 0.90:
                            # 10%: Reemplazar con ruido aleatorio
                            # Generamos un token aleatorio válido para este vocabulario específico
                            vocab_size = self.vocab_sizes[attr_idx]
                            rand_tokens = torch.randint(
                                1, max(2, vocab_size), 
                                (len(bar_indices),), dtype=torch.long
                            )
                            masked_input[b, bar_indices, attr_idx] = rand_tokens
                            
                        # else (10% restante): Se deja intacto, no modificamos masked_input

        return {
            "input_ids": masked_input,
            "attention_mask": attention_mask,
            "labels": labels
        }
class CompoundWordMLMCollator_back:
    """
    Data collator que implementa Compound Word Masking para OctaBeat-9.

    Estrategia principal – masking por posición completa:
    ─────────────────────────────────────────────────────
      Para cada posición t en la secuencia:
        Con prob `mask_prob` se selecciona para masking.
        De las seleccionadas:
          80%  → los 9 atributos se sustituyen por MASK_ID (=2)
          10%  → los 9 atributos se sustituyen por ids aleatorios
                 (uno por vocabulario, respetando tamaños)
          10%  → se dejan intactos (el modelo no sabe que están seleccionados)

    Estrategia secundaria – masking individual de armonía (opcional):
    ─────────────────────────────────────────────────────────────────
      Con prob `harmony_extra_prob` adicional, se enmascara SOLO el
      atributo de armonía (attr 8) en posiciones NO ya seleccionadas.
      Esto refuerza la señal de aprendizaje armónico sin perturbar
      el resto del compound word.

    Parámetros:
        vocab_sizes         : lista de 9 tamaños de vocabulario
        mask_prob           : probabilidad de seleccionar una posición (0.15)
        harmony_extra_prob  : prob extra de masking solo en armonía (0.05)
        mask_token_id       : id del token [MASK] (2)
    """

    N_ATTRS       = 9
    HARMONY_ATTR  = 8       # índice del atributo armonía en el vector de 9
    MASK_TOKEN_ID = 2       # [MASK] compartido por todos los vocabularios

    def __init__(
        self,
        vocab_sizes:        List[int],
        mask_prob:          float = 0.15,
        harmony_extra_prob: float = 0.05,
    ):
        self.vocab_sizes        = vocab_sizes
        self.mask_prob          = mask_prob
        self.harmony_extra_prob = harmony_extra_prob

    def __call__(
        self,
        batch: List[Dict[str, torch.Tensor]],
    ) -> Dict[str, torch.Tensor]:

        input_ids      = torch.stack([b["input_ids"]      for b in batch])  # (B, T, 9)
        attention_mask = torch.stack([b["attention_mask"] for b in batch])  # (B, T)

        B, T, _ = input_ids.shape
        masked_input = input_ids.clone()
        # labels: -100 en posiciones no enmascaradas (CrossEntropy las ignora)
        labels = torch.full((B, T, self.N_ATTRS), -100, dtype=torch.long)

        real_mask = attention_mask.bool()  # (B, T)  True = posición real

        # ── Compound Word Masking ─────────────────────────────────────────
        # selected: (B, T)  – posiciones elegidas para masking
        rand_select = torch.rand(B, T)
        selected    = real_mask & (rand_select < self.mask_prob)  # (B, T)

        # Decisión por posición: MASK(80%) / RANDOM(10%) / KEEP(10%)
        rand_decision = torch.rand(B, T)
        do_mask   = selected & (rand_decision < 0.80)   # (B, T)
        do_random = selected & (rand_decision >= 0.80) & (rand_decision < 0.90)

        # Expandir máscaras a (B, T, 9) para operar sobre los 9 atributos a la vez
        # selected_3d[b, t, :] = True  si la posición (b,t) fue seleccionada
        selected_3d = selected.unsqueeze(-1).expand(B, T, self.N_ATTRS)  # (B, T, 9)
        do_mask_3d  = do_mask.unsqueeze(-1).expand(B, T, self.N_ATTRS)   # (B, T, 9)

        # Guardar valores originales como etiquetas en las posiciones seleccionadas
        labels[selected_3d] = input_ids[selected_3d]

        # Aplicar [MASK] a todos los atributos de las posiciones seleccionadas
        masked_input[do_mask_3d] = self.MASK_TOKEN_ID

        # Aplicar RANDOM: cada atributo con su propio vocabulario
        if do_random.any():
            for attr in range(self.N_ATTRS):
                attr_random = do_random  # (B, T)
                if not attr_random.any():
                    continue
                vocab_size = self.vocab_sizes[attr]
                n_random   = int(attr_random.sum().item())
                rand_ids   = torch.randint(1, max(2, vocab_size), (n_random,),
                                           dtype=torch.long)
                # Indexación correcta: máscara 2D sobre tensor 3D en la dim del atributo
                masked_input[:, :, attr][attr_random] = rand_ids

        # ── Masking individual extra de Armonía ───────────────────────────
        if self.harmony_extra_prob > 0:
            not_selected  = real_mask & ~selected
            rand_harm     = torch.rand(B, T)
            harm_selected = not_selected & (rand_harm < self.harmony_extra_prob)

            if harm_selected.any():
                # Etiqueta solo del atributo armonía (dim 8)
                labels[:, :, self.HARMONY_ATTR][harm_selected] = \
                    input_ids[:, :, self.HARMONY_ATTR][harm_selected]

                rand_dec_harm = torch.rand(B, T)
                harm_mask     = harm_selected & (rand_dec_harm < 0.80)
                harm_random   = harm_selected & (rand_dec_harm >= 0.80) & \
                                (rand_dec_harm < 0.90)

                masked_input[:, :, self.HARMONY_ATTR][harm_mask] = self.MASK_TOKEN_ID
                if harm_random.any():
                    n_hr  = int(harm_random.sum().item())
                    v_h   = self.vocab_sizes[self.HARMONY_ATTR]
                    r_ids = torch.randint(1, max(2, v_h), (n_hr,), dtype=torch.long)
                    masked_input[:, :, self.HARMONY_ATTR][harm_random] = r_ids

        return {
            "input_ids":      masked_input,     # (B, T, 9)
            "attention_mask": attention_mask,    # (B, T)
            "labels":         labels,            # (B, T, 9)  -100=ignorar
        }


# ===========================================================================
# 3.  MÉTRICAS
# ===========================================================================

@torch.no_grad()
def compute_accuracy(
    logits_dict: Dict[str, torch.Tensor],
    labels:      torch.Tensor,
    attr_names:  List[str],
) -> Dict[str, float]:
    """Accuracy por atributo, sólo sobre posiciones enmascaradas (label ≠ -100)."""
    metrics: Dict[str, float] = {}
    for i, name in enumerate(attr_names):
        logits = logits_dict[f"logits_{name}"]   # (B, T, V)
        lbl    = labels[:, :, i]                  # (B, T)
        mask   = lbl != -100
        if not mask.any():
            metrics[f"acc_{name}"] = float("nan")
            continue
        preds   = logits.argmax(dim=-1)
        correct = (preds[mask] == lbl[mask]).float()
        metrics[f"acc_{name}"] = correct.mean().item()
    return metrics


# ===========================================================================
# 4.  TRAINER
# ===========================================================================

class HarmonyBERTTrainer:
    """
    Trainer completo para pre-entrenamiento (MLM) y fine-tuning de HarmonyBERT.

    Características:
      - Compound Word MLM Collator
      - Mixed precision fp16 / bf16
      - Gradient checkpointing
      - AdamW con warmup lineal
      - Logging por atributo (loss + acc)
      - Checkpointing best/periódico/final
      - W&B opcional
    """

    def __init__(
        self,
        config:            HarmonyBERTConfig,
        train_dataset:     OctaBeat9Dataset,
        val_dataset:       Optional[OctaBeat9Dataset],
        output_dir:        str,
        # ── Hiperparámetros ──────────────────────────────────────────────────
        task:              str   = "mlm",          # "mlm" | "harmony_pred"
        pretrained_model:  Optional[str] = None,
        batch_size:        int   = 32,
        lr:                float = 1e-4,
        weight_decay:      float = 0.01,
        epochs:            int   = 50,
        warmup_steps:      int   = 1000,
        grad_clip:         float = 1.0,
        mask_prob:         float = 0.15,
        collator_type:     str   = "bar",
        harmony_extra_prob:float = 0.05,
        fp16:              bool  = False,
        bf16:              bool  = False,
        # ── Logging / checkpoint ─────────────────────────────────────────────
        log_every:         int   = 50,
        eval_every:        int   = 500,
        save_every:        int   = 1000,
        use_wandb:         bool  = False,
        wandb_project:     str   = "harmonybert",
        # ── Sistema ──────────────────────────────────────────────────────────
        device:            str   = "auto",
        num_workers:       int   = 4,
    ):
        self.config   = config
        self.out_dir  = Path(output_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.task     = task

        # ── Dispositivo ───────────────────────────────────────────────────
        if device == "auto":
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else
                "mps"  if torch.backends.mps.is_available() else "cpu"
            )
        else:
            self.device = torch.device(device)
        log.info(f"Dispositivo: {self.device}")

        # ── Precisión ─────────────────────────────────────────────────────
        self.scaler  = None
        cuda_ok      = self.device.type == "cuda"
        if fp16 and cuda_ok:
            self.scaler  = torch.cuda.amp.GradScaler()
            self.amp_ctx = torch.cuda.amp.autocast(dtype=torch.float16)
        elif bf16 and cuda_ok:
            self.amp_ctx = torch.cuda.amp.autocast(dtype=torch.bfloat16)
        else:
            self.amp_ctx = torch.cuda.amp.autocast(enabled=False)

        # ── Modelo ────────────────────────────────────────────────────────
        if task == "mlm":
            self.model = (
                HarmonyBERTForMLM.from_pretrained(pretrained_model)
                if pretrained_model
                else HarmonyBERTForMLM(config)
            )
        elif task == "harmony_pred":
            self.model = (
                HarmonyBERTForHarmonyPrediction.from_pretrained(pretrained_model)
                if pretrained_model
                else HarmonyBERTForHarmonyPrediction(config)
            )
        else:
            raise ValueError(f"task desconocida: {task}")

        self.model.to(self.device)

        # Gradient checkpointing (ahorra memoria, algo más lento) ESTO SE PUEDE ACTIVR PARA MEJORAR LA MEMORIA
        #if config.n_layers >= 4:
        #    self.model.gradient_checkpointing_enable()

        log.info(f"Parámetros entrenables: {count_parameters(self.model):,}")

        # ── Compound Word Collator ─────────────────────────────────────────
        if collator_type == "bar":
            collator = CompoundWordMLMCollator(
                vocab_sizes = config.vocab_sizes,
                mask_prob   = mask_prob,
            )
        elif collator_type == "token":
            collator = CompoundWordMLMCollator_back(
                vocab_sizes        = config.vocab_sizes,
                mask_prob          = mask_prob,
                harmony_extra_prob = harmony_extra_prob,
            )
        else:
            raise ValueError(f"collator_type desconocido: {collator_type}")
        log.info(f"Collator activo: {collator_type}")

        # ── DataLoaders ───────────────────────────────────────────────────
        self.train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True,
            collate_fn=collator, num_workers=num_workers, pin_memory=cuda_ok,
        )
        self.val_loader = (
            DataLoader(
                val_dataset, batch_size=batch_size * 2, shuffle=False,
                collate_fn=collator, num_workers=num_workers, pin_memory=cuda_ok,
            )
            if val_dataset else None
        )

        # ── Optimizador ───────────────────────────────────────────────────
        no_decay = ["bias", "LayerNorm", "layer_norm"]
        grouped_params = [
            {
                "params": [p for n, p in self.model.named_parameters()
                           if not any(nd in n for nd in no_decay)],
                "weight_decay": weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters()
                           if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        self.optimizer = torch.optim.AdamW(grouped_params, lr=lr)

        total_steps = len(self.train_loader) * epochs
        self.scheduler = (
            get_linear_schedule_with_warmup(
                self.optimizer, warmup_steps, total_steps
            )
            if HAS_TRANSFORMERS else None
        )

        # ── Estado ────────────────────────────────────────────────────────
        self.epochs       = epochs
        self.grad_clip    = grad_clip
        self.log_every    = log_every
        self.eval_every   = eval_every
        self.save_every   = save_every
        self.global_step  = 0
        self.best_val_loss = float("inf")

        # ── W&B ───────────────────────────────────────────────────────────
        self.use_wandb = use_wandb and HAS_WANDB
        if self.use_wandb:
            wandb.init(project=wandb_project, config={
                "d_model":      config.d_model,
                "d_compound":   config.d_compound,
                "n_layers":     config.n_layers,
                "n_heads":      config.n_heads,
                "task":         task,
                "mask_prob":    mask_prob,
                "harmony_boost":harmony_extra_prob,
            })

    # ========================================================================
    # TRAIN EPOCH
    # ========================================================================

    def train_epoch(self, epoch: int) -> float:
        self.model.train()
        total_loss = 0.0
        n_steps    = 0

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch:3d}", leave=False)
        for batch in pbar:
            batch = {k: v.to(self.device) for k, v in batch.items()}

            with self.amp_ctx:
                out  = self.model(**batch)
                loss = out["loss"]

            if self.scaler:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.optimizer.step()

            self.optimizer.zero_grad(set_to_none=True)
            if self.scheduler:
                self.scheduler.step()

            total_loss       += loss.item()
            n_steps          += 1
            self.global_step += 1

            # ── Logging ─────────────────────────────────────────────────
            if self.global_step % self.log_every == 0:
                lr_now = self.optimizer.param_groups[0]["lr"]
                pbar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{lr_now:.2e}")

                if self.use_wandb:
                    log_dict = {
                        "train/loss": loss.item(),
                        "train/lr":   lr_now,
                        "step":       self.global_step,
                    }
                    for attr in self.config.attr_names:
                        key = f"loss_{attr}"
                        if key in out:
                            log_dict[f"train/{key}"] = out[key].item()
                    wandb.log(log_dict)

            # ── Evaluación periódica ─────────────────────────────────────
            if self.global_step % self.eval_every == 0 and self.val_loader:
                val_metrics = self.evaluate()
                val_loss    = val_metrics["val/loss"]
                log.info(
                    f"  step={self.global_step:6d} | "
                    f"val_loss={val_loss:.4f} | "
                    f"acc_harmony={val_metrics.get('val/acc_harmony', float('nan')):.3f} | "
                    f"best={self.best_val_loss:.4f}"
                )
                if self.use_wandb:
                    wandb.log({**val_metrics, "step": self.global_step})
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.save_checkpoint("best")
                self.model.train()

            # ── Checkpoint periódico ─────────────────────────────────────
            if self.global_step % self.save_every == 0:
                self.save_checkpoint(f"step_{self.global_step:07d}")

        return total_loss / max(n_steps, 1)

    # ========================================================================
    # EVALUACIÓN
    # ========================================================================

    @torch.no_grad()
    def evaluate(self) -> Dict[str, float]:
        if not self.val_loader:
            return {"val/loss": float("inf")}

        self.model.eval()
        total_loss     = 0.0
        n_batches      = 0
        acc_accum: Dict[str, float] = {f"acc_{a}": 0.0 for a in self.config.attr_names}

        for batch in self.val_loader:
            batch = {k: v.to(self.device) for k, v in batch.items()}
            with self.amp_ctx:
                out = self.model(**batch)

            total_loss += out["loss"].item()
            n_batches  += 1

            if "labels" in batch and self.task == "mlm":
                accs = compute_accuracy(out, batch["labels"], self.config.attr_names)
                for k, v in accs.items():
                    if not np.isnan(v):
                        acc_accum[k] = acc_accum.get(k, 0.0) + v

        avg_loss = total_loss / max(n_batches, 1)
        metrics  = {"val/loss": avg_loss}
        for k, v in acc_accum.items():
            metrics[f"val/{k}"] = v / max(n_batches, 1)

        return metrics

    # ========================================================================
    # CHECKPOINTS
    # ========================================================================

    def save_checkpoint(self, tag: str) -> None:
        path = self.out_dir / tag
        path.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(str(path))
        self.config.save_pretrained(str(path))
        torch.save({
            "optimizer":     self.optimizer.state_dict(),
            "scheduler":     self.scheduler.state_dict() if self.scheduler else None,
            "global_step":   self.global_step,
            "best_val_loss": self.best_val_loss,
        }, path / "trainer_state.pt")
        log.info(f"  → Checkpoint: {path}")

    def load_checkpoint(self, path: str) -> None:
        state = torch.load(Path(path) / "trainer_state.pt", map_location="cpu")
        self.optimizer.load_state_dict(state["optimizer"])
        if self.scheduler and state["scheduler"]:
            self.scheduler.load_state_dict(state["scheduler"])
        self.global_step   = state["global_step"]
        self.best_val_loss = state["best_val_loss"]
        log.info(f"Checkpoint cargado: {path} (step={self.global_step})")

    # ========================================================================
    # LOOP PRINCIPAL
    # ========================================================================

    def train(self) -> None:
        log.info(model_summary(self.config))
        log.info(
            f"Entrenamiento: {self.epochs} epochs | "
            f"task={self.task} | "
            f"device={self.device} | "
            f"steps/epoch={len(self.train_loader)}"
        )

        for epoch in range(1, self.epochs + 1):
            t0         = time.time()
            train_loss = self.train_epoch(epoch)
            elapsed    = time.time() - t0

            log.info(
                f"Epoch {epoch:3d}/{self.epochs} | "
                f"train_loss={train_loss:.4f} | "
                f"time={elapsed:.1f}s | "
                f"step={self.global_step}"
            )

        # Evaluación final
        if self.val_loader:
            final_metrics = self.evaluate()
            log.info(
                f"Val final: loss={final_metrics['val/loss']:.4f} | "
                f"acc_harmony={final_metrics.get('val/acc_harmony', float('nan')):.3f}"
            )

        self.save_checkpoint("final")
        log.info("Entrenamiento completado.")
        if self.use_wandb:
            wandb.finish()


# ===========================================================================
# 5.  CONFIG DESDE VOCABS.JSON
# ===========================================================================

def config_from_vocabs(vocabs_path: str, **kwargs) -> HarmonyBERTConfig:
    with open(vocabs_path, "r", encoding="utf-8") as f:
        vocabs = json.load(f)

    size_map = {}
    for k, v in vocabs.items():
        if "[PAD]" in v or "0" in v.values():
            size_map[k] = len(v)
        else:
            size_map[k] = len(v) + 5

    config = HarmonyBERTConfig(
        vocab_bar        = size_map.get("bar",        517),
        vocab_position   = size_map.get("position",   517),
        vocab_instrument = size_map.get("instrument", 134),
        vocab_pitch      = size_map.get("pitch",      133),
        vocab_duration   = size_map.get("duration",   197),
        vocab_velocity   = size_map.get("velocity",   37),
        vocab_timesig    = size_map.get("timesig",    17),
        vocab_tempo      = size_map.get("tempo",      55),
        vocab_harmony    = size_map.get("harmony",    202),
        **kwargs,
    )

    log.info("Config desde vocabs.json:")
    for name, size in zip(config.attr_names, config.vocab_sizes):
        log.info(f"  {name:12s}: vocab_size={size}")

    return config


# ===========================================================================
# 6.  CLI
# ===========================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Entrenamiento HarmonyBERT")

    # Datos
    p.add_argument("--train_file",    required=True)
    p.add_argument("--val_file",      default=None)
    p.add_argument("--vocab_file",    default=None,
                   help="vocabs.json de dcml_to_octabeat.py")

    # Modelo
    p.add_argument("--d_model",     type=int,   default=512,
                   help="Dimensión de CADA embedding de atributo")
    p.add_argument("--n_heads",     type=int,   default=8)
    p.add_argument("--n_layers",    type=int,   default=6)
    p.add_argument("--d_ff",        type=int,   default=2048)
    p.add_argument("--max_seq_len", type=int,   default=512)

    # Entrenamiento
    p.add_argument("--task",              default="mlm",
                   choices=["mlm", "harmony_pred"])
    p.add_argument("--pretrained_model",  default=None)
    p.add_argument("--output_dir",        default="checkpoints/harmonybert")
    p.add_argument("--epochs",            type=int,   default=50)
    p.add_argument("--batch_size",        type=int,   default=32)
    p.add_argument("--lr",                type=float, default=1e-4)
    p.add_argument("--weight_decay",      type=float, default=0.01)
    p.add_argument("--warmup_steps",      type=int,   default=1000)
    p.add_argument("--grad_clip",         type=float, default=1.0)
    p.add_argument("--mask_prob",         type=float, default=0.15)
    p.add_argument("--harmony_extra_prob",type=float, default=0.05)
    p.add_argument("--collator", default="bar", choices=["bar", "token"],
                   help="'bar' = Bar-Level (activo) | 'token' = Token-Level (_back)")
    p.add_argument("--fp16",              action="store_true")
    p.add_argument("--bf16",              action="store_true")
    p.add_argument("--num_workers",       type=int,   default=4)

    # Logging
    p.add_argument("--log_every",      type=int, default=50)
    p.add_argument("--eval_every",     type=int, default=500)
    p.add_argument("--save_every",     type=int, default=1000)
    p.add_argument("--use_wandb",      action="store_true")
    p.add_argument("--wandb_project",  default="harmonybert")
    p.add_argument("--device",         default="auto")
    p.add_argument("--seed",           type=int, default=42)
    p.add_argument("--resume",         default=None,
                   help="Ruta de checkpoint para reanudar")

    return p.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


if __name__ == "__main__":
    args = parse_args()
    set_seed(args.seed)

    # Config
    model_kwargs = dict(
        d_model     = args.d_model,
        n_heads     = args.n_heads,
        n_layers    = args.n_layers,
        d_ff        = args.d_ff,
        max_seq_len = args.max_seq_len,
    )
    config = (
        config_from_vocabs(args.vocab_file, **model_kwargs)
        if args.vocab_file
        else HarmonyBERTConfig(**model_kwargs)
    )

    # Datasets
    train_ds = OctaBeat9Dataset(args.train_file, max_seq_len=args.max_seq_len)
    val_ds   = (OctaBeat9Dataset(args.val_file, max_seq_len=args.max_seq_len)
                if args.val_file else None)

    # Trainer
    trainer = HarmonyBERTTrainer(
        config             = config,
        train_dataset      = train_ds,
        val_dataset        = val_ds,
        output_dir         = args.output_dir,
        task               = args.task,
        pretrained_model   = args.pretrained_model,
        batch_size         = args.batch_size,
        lr                 = args.lr,
        weight_decay       = args.weight_decay,
        epochs             = args.epochs,
        warmup_steps       = args.warmup_steps,
        grad_clip          = args.grad_clip,
        mask_prob          = args.mask_prob,
        harmony_extra_prob = args.harmony_extra_prob,
        collator_type      = args.collator,
        fp16               = args.fp16,
        bf16               = args.bf16,
        log_every          = args.log_every,
        eval_every         = args.eval_every,
        save_every         = args.save_every,
        use_wandb          = args.use_wandb,
        wandb_project      = args.wandb_project,
        device             = args.device,
        num_workers        = args.num_workers,
    )

    if args.resume:
        trainer.load_checkpoint(args.resume)

    trainer.train()