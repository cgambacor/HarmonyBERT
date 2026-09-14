"""
harmonybert_model.py
=====================
Arquitectura HarmonyBERT con Compound Word Token Embedding.

Idea central (fiel al paper MusicBERT):
  Cada nota se representa como un "compound word" de 9 atributos.
  En lugar de sumar o concatenar embeddings pequeños, CADA atributo
  tiene su propio espacio de embedding completo de dimensión d_model (512).
  La fusión ocurre DESPUÉS de generar los 9 vectores completos:

      attr_i  ──► Embedding_i(vocab_i, 512) ──► e_i   (B, T, 512)
                                                  │
      e_0 ⊕ e_1 ⊕ … ⊕ e_8  →  concat  →  (B, T, 9×512 = 4608)
                                                  │
                                          Linear(4608 → 512)
                                          + LayerNorm
                                          + Dropout
                                                  │
                                          Transformer Encoder
                                                  │
                                          (B, T, 512)

  Esto garantiza que cada atributo aprende su propio espacio semántico
  completo antes de fusionarse, al contrario de dividir 512/9 ≈ 56 dims
  por atributo.

Componentes:
  1. CompoundWordEmbedding   – 9 embeddings×512 → concat 4608 → proj 512
  2. HarmonyBERTLayer        – capa Transformer estándar (BERT)
  3. HarmonyBERTEncoder      – pila de capas Transformer
  4. HarmonyBERT             – modelo base (embedding + encoder)
  5. MLMHead                 – 9 cabezas de decodificación para MLM
  6. HarmonyBERTForMLM       – pre-entrenamiento
  7. HarmonyBERTForHarmonyPrediction – fine-tuning armonía token-a-token
  8. HarmonyBERTForHarmonyContinuation – predicción multi-atributo

Dependencias:
  pip install torch transformers
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PretrainedConfig, PreTrainedModel
from transformers.modeling_outputs import BaseModelOutput


# ===========================================================================
# 1.  CONFIGURACIÓN
# ===========================================================================

@dataclass
class HarmonyBERTConfig(PretrainedConfig):
    """
    Configuración de HarmonyBERT.

    Parámetro clave: d_model es la dimensión de CADA embedding de atributo.
    La dimensión interna del compound word antes de proyectar es
    d_model × n_attrs = 512 × 9 = 4608.
    """
    model_type: str = "harmony_bert"
    
    def __init__(self, **kwargs):
        # ── Tamaños de vocabulario (5 especiales + valores reales) ───────────
        #   vocab = VOCAB_OFFSET(5) + n_valores_reales
        self.vocab_bar        = kwargs.pop("vocab_bar",        517)  # 5 + 512 compases
        self.vocab_position   = kwargs.pop("vocab_position",   517)   # 5 + 512 subdivisiones
        self.vocab_instrument = kwargs.pop("vocab_instrument", 134)  # 5 + 129 (128 MIDI + drum)
        self.vocab_pitch      = kwargs.pop("vocab_pitch",      133)  # 5 + 128 pitches MIDI
        self.vocab_duration   = kwargs.pop("vocab_duration",   197)  # 5 + 192 ticks
        self.vocab_velocity   = kwargs.pop("vocab_velocity",   37)   # 5 + 32 bins
        self.vocab_timesig    = kwargs.pop("vocab_timesig",    17)   # 5 + 12 firmas
        self.vocab_tempo      = kwargs.pop("vocab_tempo",      55)   # 5 + 50 bins BPM
        self.vocab_harmony    = kwargs.pop("vocab_harmony",    1005) # 5 + 1000 etiquetas DCML

        # ── Arquitectura ──────────────────────────────────────────────────────
        self.d_model     = kwargs.pop("d_model",     512)
        self.n_heads     = kwargs.pop("n_heads",     8)
        self.n_layers    = kwargs.pop("n_layers",    6)
        self.d_ff        = kwargs.pop("d_ff",        2048)
        self.max_seq_len = kwargs.pop("max_seq_len", 512)
        self.dropout     = kwargs.pop("dropout",     0.1)

        # ── Tokens especiales propios (no estándar en PretrainedConfig) ───────
        self.unk_token_id  = kwargs.pop("unk_token_id",  1)
        self.mask_token_id = kwargs.pop("mask_token_id", 2)

        # ── PretrainedConfig acepta pad/bos/eos como kwargs estándar ─────────
        super().__init__(
            pad_token_id = kwargs.pop("pad_token_id", 0),
            bos_token_id = kwargs.pop("bos_token_id", 3),
            eos_token_id = kwargs.pop("eos_token_id", 4),
            **kwargs,   # permite _name_or_path, _commit_hash, etc.
        )

    """# ── Tamaños de vocabulario por atributo (OctaBeat-9) ───────────────────
    vocab_bar:        int = 517    # 512 compases + PAD=0 y resto de caracteres especiles 5 en total  UNK=1, MASK=2, BOS=3, EOS=4
    vocab_position:   int = 53     # 48 subdivisiones + y resto de caracteres especiles 5 en total  UNK=1, MASK=2, BOS=3, EOS=4
    vocab_instrument: int = 134    # 128 programas MIDI + drum(128) + PAD y resto de caracteres especiles 5 en total  UNK=1, MASK=2, BOS=3, EOS=4
    vocab_pitch:      int = 133    # MIDI 0-127 + PAD y resto de caracteres especiles 5 en total  UNK=1, MASK=2, BOS=3, EOS=4
    vocab_duration:   int = 197    # 192 ticks (1 tick=1/48 negra) + PAD y resto de caracteres especiles 5 en total  UNK=1, MASK=2, BOS=3, EOS=4
    vocab_velocity:   int = 37     # 32 bins de dinámica + PAD y resto de caracteres especiles 5 en total  UNK=1, MASK=2, BOS=3, EOS=4
    vocab_timesig:    int = 17     # 12 compases comunes + PAD y resto de caracteres especiles 5 en total  UNK=1, MASK=2, BOS=3, EOS=4
    vocab_tempo:      int = 55     # 50 bins BPM + PAD y resto de caracteres especiles 5 en total  UNK=1, MASK=2, BOS=3, EOS=4
    vocab_harmony:    int = 1005   # ~1000 etiquetas DCML + PAD y resto de caracteres especiles 5 en total  UNK=1, MASK=2, BOS=3, EOS=4

    # ── Arquitectura del Transformer ────────────────────────────────────────
    # d_model : dimensión de CADA embedding individual (y del Transformer)
    d_model:     int   = 512
    n_heads:     int   = 8
    n_layers:    int   = 6
    d_ff:        int   = 2048
    max_seq_len: int   = 512
    dropout:     float = 0.1

    # ── Tokens especiales ────────────────────────────────────────────────────
    pad_token_id:  int = 0
    unk_token_id:  int = 1   # [UNK]
    mask_token_id: int = 2   # [MASK]
    bos_token_id:  int = 3   # [BOS]
    eos_token_id:  int = 4   # [EOS]

    def __post_init__(self):
        super().__init__(
            pad_token_id=self.pad_token_id,
            unk_token_id=self.unk_token_id,
            mask_token_id=self.mask_token_id,
            bos_token_id=self.bos_token_id,
            eos_token_id=self.eos_token_id,
        )
"""
    # ── Propiedades derivadas ────────────────────────────────────────────────

    @property
    def n_attrs(self) -> int:
        return 9

    @property
    def d_compound(self) -> int:
        """Dimensión del vector concatenado antes de proyectar: 9 × d_model."""
        return self.n_attrs * self.d_model   # 9 × 512 = 4608

    @property
    def vocab_sizes(self) -> List[int]:
        return [
            self.vocab_bar, self.vocab_position, self.vocab_instrument,
            self.vocab_pitch, self.vocab_duration, self.vocab_velocity,
            self.vocab_timesig, self.vocab_tempo, self.vocab_harmony,
        ]

    @property
    def attr_names(self) -> List[str]:
        return [
            "bar", "position", "instrument", "pitch",
            "duration", "velocity", "timesig", "tempo", "harmony",
        ]
    @property
    def special_token_ids(self) -> Dict[str, int]:
            """Mapa nombre → id para los 5 tokens especiales."""
            return {
                "[PAD]":  self.pad_token_id,
                "[UNK]":  self.unk_token_id,
                "[MASK]": self.mask_token_id,
                "[BOS]":  self.bos_token_id,
                "[EOS]":  self.eos_token_id,
            }

# ===========================================================================
# 2.  COMPOUND WORD EMBEDDING  (núcleo de la arquitectura)
# ===========================================================================

class CompoundWordEmbedding(nn.Module):
    """
    Compound Word Token Embedding para OctaBeat-9.

    Por cada token (nota), el modelo genera 9 embeddings independientes,
    cada uno de tamaño d_model (512). Los concatena en un vector de
    d_compound = 9 × 512 = 4608 y lo proyecta de vuelta a d_model.

    Flujo detallado:
    ─────────────────────────────────────────────────────────────────
    Entrada x: (B, T, 9)  – 9 índices enteros por token

    Para cada atributo i:
        e_i = Embedding_i(x[:,:,i])    →  (B, T, 512)
              (vocab_sizes[i] × 512, padding_idx=0)

    compound = concat([e_0, e_1, …, e_8], dim=-1)  →  (B, T, 4608)

    projected = Linear(4608 → 512)(compound)        →  (B, T, 512)

    pos_emb   = PosEmbedding(0..T-1)                →  (1, T, 512)

    out = LayerNorm(projected + pos_emb)
    out = Dropout(out)                               →  (B, T, 512)
    ─────────────────────────────────────────────────────────────────

    Ventaja frente a sum/small-concat:
      Cada atributo tiene su propio espacio de 512 dims para aprender
      representaciones ricas antes de combinarse. El modelo puede capturar
      interacciones no lineales entre atributos a través de la proyección
      y luego en las capas del Transformer.
    """

    N_ATTRS = 9

    def __init__(self, config: HarmonyBERTConfig):
        super().__init__()
        self.d_model    = config.d_model
        self.d_compound = config.d_compound   # 9 × 512 = 4608

        # ── 9 tablas de embedding, cada una de tamaño vocab_i × d_model ─────
        self.attr_embeddings = nn.ModuleList([
            nn.Embedding(vocab_size, config.d_model, padding_idx=0)
            for vocab_size in config.vocab_sizes
        ])

        # ── Proyección compound → d_model ────────────────────────────────────
        # Linear(4608 → 512): fusiona los 9 espacios semánticos
        self.compound_proj = nn.Linear(self.d_compound, config.d_model, bias=False)

        # ── Embedding posicional (posición en la secuencia) ──────────────────
        self.pos_embedding = nn.Embedding(config.max_seq_len + 2, config.d_model)

        # ── Normalización y regularización ───────────────────────────────────
        self.layer_norm = nn.LayerNorm(config.d_model)
        self.dropout    = nn.Dropout(config.dropout)

        self._init_weights()

    def _init_weights(self) -> None:
        # Embeddings de atributos: normal(0, 0.02), PAD=0
        for emb in self.attr_embeddings:
            nn.init.normal_(emb.weight, std=0.02)
            with torch.no_grad():
                emb.weight[0].zero_()   # PAD token → vector cero

        # Proyección: Xavier uniforme
        nn.init.xavier_uniform_(self.compound_proj.weight)

        # Posicionales: normal(0, 0.02)
        nn.init.normal_(self.pos_embedding.weight, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, 9)  índices enteros, dtype=torch.long

        Returns:
            out: (B, T, d_model)  representación compound proyectada
        """
        B, T, _ = x.shape

        # ── Paso 1: embedding individual por atributo ─────────────────────
        # Genera 9 tensores (B, T, d_model) y los concatena en el último eje
        attr_embs = [
            self.attr_embeddings[i](x[:, :, i])   # (B, T, 512)
            for i in range(self.N_ATTRS)
        ]
        # compound: (B, T, 9 × 512) = (B, T, 4608)
        compound = torch.cat(attr_embs, dim=-1)

        # ── Paso 2: proyección compound → d_model ────────────────────────
        projected = self.compound_proj(compound)  # (B, T, 512)

        # ── Paso 3: embedding posicional de secuencia ─────────────────────
        positions = torch.arange(T, device=x.device).unsqueeze(0)  # (1, T)
        pos_emb   = self.pos_embedding(positions)                   # (1, T, 512)

        # ── Paso 4: combinar, normalizar, regularizar ─────────────────────
        out = self.layer_norm(projected + pos_emb)
        return self.dropout(out)                  # (B, T, 512)


# ===========================================================================
# 3.  TRANSFORMER ENCODER  (BERT-style)
# ===========================================================================

class HarmonyBERTLayer(nn.Module):
    """Capa Transformer estándar: Multi-Head Self-Attention + FFN."""

    def __init__(self, config: HarmonyBERTConfig):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            embed_dim   = config.d_model,
            num_heads   = config.n_heads,
            dropout     = config.dropout,
            batch_first = True,
        )
        self.ff = nn.Sequential(
            nn.Linear(config.d_model, config.d_ff),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_ff, config.d_model),
            nn.Dropout(config.dropout),
        )
        self.norm1 = nn.LayerNorm(config.d_model)
        self.norm2 = nn.LayerNorm(config.d_model)

    def forward(
        self,
        x:                torch.Tensor,                    # (B, T, d_model)
        key_padding_mask: Optional[torch.Tensor] = None,  # (B, T) True=ignorar
    ) -> torch.Tensor:
        # Self-attention + residual + norm
        attn_out, _ = self.self_attn(x, x, x, key_padding_mask=key_padding_mask)
        x = self.norm1(x + attn_out)
        # FFN + residual + norm
        x = self.norm2(x + self.ff(x))
        return x


class HarmonyBERTEncoder(nn.Module):
    """Pila de n_layers capas Transformer."""

    def __init__(self, config: HarmonyBERTConfig):
        super().__init__()
        self.layers = nn.ModuleList([
            HarmonyBERTLayer(config) for _ in range(config.n_layers)
        ])

    def forward(
        self,
        x:                torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, key_padding_mask)
        return x


# ===========================================================================
# 4.  MODELO BASE
# ===========================================================================

class HarmonyBERTPreTrainedModel(PreTrainedModel):
    config_class              = HarmonyBERTConfig
    base_model_prefix         = "harmonybert"
    supports_gradient_checkpointing = True

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)


class HarmonyBERT(HarmonyBERTPreTrainedModel):
    """
    Modelo base: CompoundWordEmbedding → Transformer Encoder.
    Salida: (B, T, d_model) representaciones contextualizadas.
    """

    def __init__(self, config: HarmonyBERTConfig):
        super().__init__(config)
        self.embedding = CompoundWordEmbedding(config)
        self.encoder   = HarmonyBERTEncoder(config)
        self.post_init()

    def forward(
        self,
        input_ids:      torch.Tensor,                   # (B, T, 9)
        attention_mask: Optional[torch.Tensor] = None,  # (B, T) 1=real, 0=PAD
    ) -> BaseModelOutput:
        # key_padding_mask: True = posición a ignorar
        kpm = (attention_mask == 0) if attention_mask is not None else None

        hidden = self.embedding(input_ids)       # (B, T, 512)
        hidden = self.encoder(hidden, kpm)       # (B, T, 512)

        return BaseModelOutput(last_hidden_state=hidden)


# ===========================================================================
# 5.  CABEZA MLM
# ===========================================================================

class MLMHead(nn.Module):
    """
    Cabeza de decodificación para MLM.

    Recibe la representación contextualizada (B, T, d_model) y genera
    9 distribuciones de probabilidad, una por atributo.

    Arquitectura interna:
        hidden  →  Dense(d_model → d_model)  →  GELU  →  LayerNorm
                →  [Decoder_i(d_model → vocab_i)]  × 9
    """

    def __init__(self, config: HarmonyBERTConfig):
        super().__init__()
        self.dense      = nn.Linear(config.d_model, config.d_model)
        self.act        = nn.GELU()
        self.layer_norm = nn.LayerNorm(config.d_model)

        # Un decodificador lineal por atributo
        self.decoders = nn.ModuleList([
            nn.Linear(config.d_model, vocab_size)
            for vocab_size in config.vocab_sizes
        ])

    def forward(self, hidden: torch.Tensor) -> List[torch.Tensor]:
        """
        hidden: (B, T, d_model)
        Returns: lista de 9 tensores (B, T, vocab_i)
        """
        x = self.layer_norm(self.act(self.dense(hidden)))
        return [dec(x) for dec in self.decoders]


# ===========================================================================
# 6.  PRE-ENTRENAMIENTO: MLM
# ===========================================================================

class HarmonyBERTForMLM(HarmonyBERTPreTrainedModel):
    """
    HarmonyBERT con cabeza MLM para pre-entrenamiento.

    El masking se aplica a nivel de COMPOUND WORD: cuando se enmascara
    una posición, se enmascaran TODOS sus atributos simultáneamente
    (comportamiento auténtico de MusicBERT). El collator del training
    gestiona esta lógica; el modelo simplemente recibe los índices
    enmascarados y las etiquetas.
    """

    def __init__(self, config: HarmonyBERTConfig):
        super().__init__(config)
        self.harmonybert = HarmonyBERT(config)
        self.mlm_head    = MLMHead(config)
        self.post_init()

    def forward(
        self,
        input_ids:      torch.Tensor,                   # (B, T, 9)
        attention_mask: Optional[torch.Tensor] = None,  # (B, T)
        labels:         Optional[torch.Tensor] = None,  # (B, T, 9) -100=ignorar
    ) -> Dict[str, torch.Tensor]:

        outputs = self.harmonybert(input_ids, attention_mask)
        hidden  = outputs.last_hidden_state               # (B, T, d_model)

        logits_list = self.mlm_head(hidden)               # 9 × (B, T, vocab_i)

        result: Dict[str, torch.Tensor] = {
            f"logits_{name}": lg
            for name, lg in zip(self.config.attr_names, logits_list)
        }

        if labels is not None:
            total_loss = hidden.new_zeros(())
            for i, (name, logits) in enumerate(
                zip(self.config.attr_names, logits_list)
            ):
                lbl  = labels[:, :, i].contiguous()        # (B, T)
                loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    lbl.view(-1),
                    ignore_index=-100,
                )
                result[f"loss_{name}"] = loss
                total_loss = total_loss + loss

            # Pérdida media sobre los 9 atributos
            result["loss"] = total_loss / self.config.n_attrs

        return result


# ===========================================================================
# 7.  FINE-TUNING: PREDICCIÓN DE ARMONÍA (token-a-token)
# ===========================================================================

class HarmonyBERTForHarmonyPrediction(HarmonyBERTPreTrainedModel):
    """
    Fine-tuning: clasificación de acorde en cada posición temporal.
    Input:  secuencia OctaBeat-9 (con harmony = [MASK] o real)
    Output: distribución sobre vocab_harmony en cada posición.
    """

    def __init__(self, config: HarmonyBERTConfig):
        super().__init__(config)
        self.harmonybert = HarmonyBERT(config)
        self.dropout     = nn.Dropout(config.dropout)
        self.classifier  = nn.Linear(config.d_model, config.vocab_harmony)
        self.post_init()

    def forward(
        self,
        input_ids:      torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels:         Optional[torch.Tensor] = None,  # (B, T) índice de acorde
    ) -> Dict[str, torch.Tensor]:

        hidden  = self.harmonybert(input_ids, attention_mask).last_hidden_state
        logits  = self.classifier(self.dropout(hidden))  # (B, T, vocab_harmony)
        result  = {"logits": logits}

        if labels is not None:
            result["loss"] = F.cross_entropy(
                logits.view(-1, self.config.vocab_harmony),
                labels.view(-1),
                ignore_index=-100,
            )
        return result


# ===========================================================================
# 8.  FINE-TUNING: PREDICCIÓN MULTI-ATRIBUTO (armonización completa)
# ===========================================================================

class HarmonyBERTForHarmonyContinuation(HarmonyBERTPreTrainedModel):
    """
    Fine-tuning: predice los 9 atributos en cada posición.
    Útil para armonización (predecir función armónica dado el contexto
    melódico) o para completar fragmentos.
    """

    def __init__(self, config: HarmonyBERTConfig):
        super().__init__(config)
        self.harmonybert = HarmonyBERT(config)
        self.dropout     = nn.Dropout(config.dropout)
        self.heads = nn.ModuleDict({
            name: nn.Linear(config.d_model, vocab)
            for name, vocab in zip(config.attr_names, config.vocab_sizes)
        })
        self.post_init()

    def forward(
        self,
        input_ids:      torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels:         Optional[torch.Tensor] = None,  # (B, T, 9)
    ) -> Dict[str, torch.Tensor]:

        hidden = self.dropout(
            self.harmonybert(input_ids, attention_mask).last_hidden_state
        )
        all_logits = {name: head(hidden) for name, head in self.heads.items()}
        result = {f"logits_{k}": v for k, v in all_logits.items()}

        if labels is not None:
            total_loss = hidden.new_zeros(())
            for i, name in enumerate(self.config.attr_names):
                loss = F.cross_entropy(
                    all_logits[name].view(-1, all_logits[name].size(-1)),
                    labels[:, :, i].view(-1),
                    ignore_index=-100,
                )
                result[f"loss_{name}"] = loss
                total_loss = total_loss + loss
            result["loss"] = total_loss / self.config.n_attrs

        return result


# ===========================================================================
# 9.  UTILIDADES
# ===========================================================================

def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def model_summary(config: HarmonyBERTConfig) -> str:
    model    = HarmonyBERTForMLM(config)
    n_params = count_parameters(model)

    # Desglose de parámetros del embedding compound
    emb_params = sum(
        p.numel()
        for p in model.harmonybert.embedding.parameters()
        if p.requires_grad
    )
    proj_params = sum(
        p.numel()
        for p in model.harmonybert.embedding.compound_proj.parameters()
        if p.requires_grad
    )

    lines = [
        "=" * 64,
        "  HarmonyBERT – Compound Word Embedding Architecture",
        "=" * 64,
        f"  d_model per attr : {config.d_model}",
        f"  n_attrs          : {config.n_attrs}",
        f"  d_compound       : {config.d_compound}  ({config.n_attrs} × {config.d_model})",
        f"  proj shape       : Linear({config.d_compound} → {config.d_model})",
        f"  n_heads          : {config.n_heads}",
        f"  n_layers         : {config.n_layers}",
        f"  d_ff             : {config.d_ff}",
        f"  max_seq_len      : {config.max_seq_len}",
        "-" * 64,
        f"  Params embedding : {emb_params:>12,}",
        f"    of which proj  : {proj_params:>12,}",
        f"  Total params     : {n_params:>12,}",
        "=" * 64,
        "  Vocabularios por atributo:",
    ]
    for name, size in zip(config.attr_names, config.vocab_sizes):
        table_params = size * config.d_model
        lines.append(f"    {name:12s}: vocab={size:5d}  table={table_params:>9,} params")
    lines.append("=" * 64)
    return "\n".join(lines)


# ===========================================================================
# SMOKE TEST
# ===========================================================================

if __name__ == "__main__":
    cfg = HarmonyBERTConfig()
    print(model_summary(cfg))

    B, T = 2, 64
    x    = torch.randint(1, 10, (B, T, 9))          # (B, T, 9) índices
    mask = torch.ones(B, T, dtype=torch.long)
    lbl  = torch.randint(0, 9, (B, T, 9))
    lbl[lbl == 0] = -100                              # simular ignorar PAD

    model = HarmonyBERTForMLM(cfg)
    out   = model(x, mask, lbl)

    print(f"\n  Loss total     : {out['loss'].item():.4f}")
    print(f"  Loss armonía   : {out['loss_harmony'].item():.4f}")
    print(f"  Logits harmony : {out['logits_harmony'].shape}")
    print("\n  Smoke test OK ✓")
