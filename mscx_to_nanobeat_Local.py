"""
mscx_to_nanobeat.py
====================
Convierte ficheros MuseScore (.mscx / .mscz) con anotaciones de armonía DCML
al formato NanoBeat-9 utilizado por HarmonyBERT.

Fusiona:
  - Parser XML nativo de MuseScore de new_mscx_to_nanobeat.py
  - Normalización de armonía (normalize_label_lvl1) de midi_to_nanobeat.py
  - Soporte de corpus multi-autor con estructura de directorios estándar

IMPORTANTE sobre el formato .mscx
-----------------------------------
Los ficheros .mscx son XML NATIVO de MuseScore con raíz <museScore>.
NO son MusicXML estándar (<score-partwise>). Por tanto music21 no puede
parsearlos directamente. Este módulo usa xml.etree.ElementTree para leer
la estructura MuseScore directamente, sin dependencias externas pesadas.

Estructura XML relevante de un .mscx
--------------------------------------
<museScore version="4.x">
  <Score>
    <Part>
      <Instrument>
        <programValue>0</programValue>   ← programa MIDI
      </Instrument>
      <Staff id="1">
        <Measure number="1">
          <voice>
            <TimeSig>
              <sigN>4</sigN>
              <sigD>4</sigD>
            </TimeSig>
            <Tempo>
              <tempo>2.0</tempo>         ← negras/segundo (×60 = BPM)
            </Tempo>
            <Harmony>
              <root>14</root>
              <name>V7</name>            ← etiqueta armónica
            </Harmony>
            <Chord>
              <durationType>quarter</durationType>
              <Note>
                <pitch>60</pitch>
                <velocity>80</velocity>
              </Note>
            </Chord>
          </voice>
        </Measure>
      </Staff>
    </Part>
  </Score>
</museScore>

Estructura de directorios esperada
-------------------------------------
  <corpus_root>/
    <autor_1>/
      MS3/
        obra_1.mscx
        obra_2.mscz
    <autor_2>/
      MS3/
        obra_3.mscx
    ...

Uso:
  python mscx_to_nanobeat.py \\
      --input_dir /ruta/corpus \\
      --output    /ruta/salida/nanobeat.jsonl

  # Sin estructura MS3 (busca .mscx/.mscz en cualquier subdirectorio):
  python mscx_to_nanobeat.py \\
      --input_dir /ruta/corpus \\
      --output    /ruta/salida/nanobeat.jsonl \\
      --no_ms3

Tokens de salida [9 dimensiones]:
  [Compás, Posición, Instrumento, Tono, Duración, Velocidad, Compás(Firma), Tempo, Armonía]

Granularidad del vocabulario de armonía (controlada en normalize_harmony_lvl1.py):
  normalize_label_lvl1 aplica la normalización adaptada al corpus propio.

Dependencias:
  pip install numpy tqdm lxml   (lxml es opcional, mejora velocidad)
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import shutil
import tempfile
import warnings
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from xml.etree import ElementTree as ET

import numpy as np
from tqdm import tqdm

from normalize_harmony_lvl1 import normalize_label_lvl1

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)


# ===========================================================================
# 1.  VOCABULARIOS
# ===========================================================================

SPECIAL_TOKENS = {"[PAD]": 0, "[UNK]": 1, "[MASK]": 2, "[BOS]": 3, "[EOS]": 4}
UNK_TOKEN_ID   = 1


def build_bar_vocab(max_bars: int = 512) -> Dict:
    vocab = SPECIAL_TOKENS.copy()
    vocab.update({b: i + 5 for i, b in enumerate(range(max_bars))})
    return vocab


def build_position_vocab(subdivisions: int = 512) -> Dict:
    vocab = SPECIAL_TOKENS.copy()
    vocab.update({p: i + 5 for i, p in enumerate(range(subdivisions))})
    return vocab


def build_instrument_vocab() -> Dict:
    vocab = SPECIAL_TOKENS.copy()
    vocab.update({p: i + 5 for i, p in enumerate(range(129))})
    return vocab


def build_pitch_vocab() -> Dict:
    vocab = SPECIAL_TOKENS.copy()
    vocab.update({p: i + 5 for i, p in enumerate(range(128))})
    return vocab


def build_duration_vocab(max_ticks: int = 192) -> Dict:
    vocab = SPECIAL_TOKENS.copy()
    vocab.update({d: i + 5 for i, d in enumerate(range(1, max_ticks + 1))})
    return vocab


def build_velocity_vocab(bins: int = 32) -> Dict:
    vocab = SPECIAL_TOKENS.copy()
    vocab.update({min(127, b * 4): i + 5 for i, b in enumerate(range(bins))})
    return vocab


def build_timesig_vocab() -> Dict:
    vocab = SPECIAL_TOKENS.copy()
    common = ["2/2", "2/4", "3/4", "4/4", "6/8", "9/8", "12/8",
              "3/8", "5/4", "7/8", "6/4", "3/2"]
    vocab.update({ts: i + 5 for i, ts in enumerate(common)})
    return vocab


def build_tempo_vocab(bins: int = 50) -> Dict:
    vocab = SPECIAL_TOKENS.copy()
    edges   = np.linspace(20, 300, bins + 1)
    centers = ((edges[:-1] + edges[1:]) / 2).astype(int).tolist()
    vocab.update({bpm: i + 5 for i, bpm in enumerate(centers)})
    return vocab


def build_harmony_vocab_from_labels(labels: List[str]) -> Dict[str, int]:
    """
    Construye el vocabulario de armonía desde las etiquetas REALES del corpus,
    normalizadas con normalize_label_lvl1.

    Los acordes más frecuentes reciben ids más bajos.

    Args:
        labels: lista de TODAS las etiquetas raw extraídas del corpus.

    Returns:
        dict {label_normalizada: id}  con especiales en 0-4 y reales desde 5.
    """
    freq: Dict[str, int] = {}
    for raw in labels:
        normalized = normalize_label_lvl1(raw)
        if normalized and normalized != "[UNK]":
            freq[normalized] = freq.get(normalized, 0) + 1

    labels_sorted = sorted(freq.keys(), key=lambda l: -freq[l])

    vocab: Dict[str, int] = dict(SPECIAL_TOKENS)
    for i, label in enumerate(labels_sorted):
        vocab[label] = i + 5

    log.info(
        f"Vocab armonía [normalize_label_lvl1]: "
        f"{len(vocab)} tokens totales  "
        f"({len(vocab) - 5} acordes únicos)"
    )
    return vocab


# ---------------------------------------------------------------------------
# Vocabularios fijos (se construyen al importar el módulo).
# harmony se construye DESPUÉS del escaneo en convert_corpus().
# ---------------------------------------------------------------------------
VOCABS: Dict[str, Dict] = {
    "bar":        build_bar_vocab(),
    "position":   build_position_vocab(),
    "instrument": build_instrument_vocab(),
    "pitch":      build_pitch_vocab(),
    "duration":   build_duration_vocab(),
    "velocity":   build_velocity_vocab(),
    "timesig":    build_timesig_vocab(),
    "tempo":      build_tempo_vocab(),
    "harmony":    dict(SPECIAL_TOKENS),   # placeholder; se rellena en convert_corpus
}

VOCAB_SIZES: Dict[str, int] = {
    k: max(v.values()) + 1 for k, v in VOCABS.items()
}


def save_vocabs(path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(
            {k: {str(kk): vv for kk, vv in v.items()} for k, v in VOCABS.items()},
            f, ensure_ascii=False, indent=2,
        )
    log.info(f"Vocabularios → {path}")


# ===========================================================================
# 2.  DATACLASSES
# ===========================================================================

@dataclass
class NanoBeat9:
    bar: int; position: int; instrument: int; pitch: int
    duration: int; velocity: int; timesig: int; tempo: int; harmony: int

    def to_list(self) -> List[int]:
        return [self.bar, self.position, self.instrument, self.pitch,
                self.duration, self.velocity, self.timesig, self.tempo,
                self.harmony]


@dataclass
class NanoBeatSequence:
    piece_id: str
    author:   str
    tokens:   List[List[int]]
    metadata: Dict = field(default_factory=dict)


# ===========================================================================
# 3.  CODIFICADORES AUXILIARES
# ===========================================================================

TICKS_PER_BEAT = 48

DURATION_TYPE_TO_QN: Dict[str, float] = {
    "1024th": 1/256, "512th": 1/128, "256th": 1/64,
    "128th":  1/32,  "64th":  1/16,  "32nd":  1/8,
    "16th":   1/4,   "eighth": 1/2,  "quarter": 1.0,
    "half":   2.0,   "whole":  4.0,  "breve":  8.0,
    "long":   16.0,  "maxima": 32.0,
}


def _enc_bar(bar_0idx: int) -> int:
    return VOCABS["bar"].get(min(bar_0idx, 511), UNK_TOKEN_ID)


def _enc_position(offset_qn: float) -> int:
    tick = int(round(offset_qn * TICKS_PER_BEAT))
    return VOCABS["position"].get(min(tick, 511), UNK_TOKEN_ID)


def _enc_duration(dur_qn: float) -> int:
    ticks = max(1, min(192, int(round(dur_qn * TICKS_PER_BEAT))))
    return VOCABS["duration"].get(ticks, UNK_TOKEN_ID)


def _enc_velocity(vel: int) -> int:
    binned = min(31, vel // 4) * 4
    return VOCABS["velocity"].get(binned, UNK_TOKEN_ID)


def _enc_timesig(num: int, den: int) -> int:
    return VOCABS["timesig"].get(
        f"{num}/{den}", VOCABS["timesig"].get("4/4", UNK_TOKEN_ID)
    )


def _enc_tempo(bpm: float) -> int:
    numeric_keys = [k for k in VOCABS["tempo"].keys() if isinstance(k, (int, float))]
    if not numeric_keys:
        return UNK_TOKEN_ID
    closest = min(numeric_keys, key=lambda k: abs(k - bpm))
    return VOCABS["tempo"][closest]


def _enc_harmony(label: str) -> int:
    """
    Normaliza la etiqueta con normalize_label_lvl1 y la busca en el vocabulario.
    Con el vocab construido desde el corpus, el UNK rate debería ser ~0%.
    """
    normalized = normalize_label_lvl1(label)

    if normalized in VOCABS["harmony"]:
        return VOCABS["harmony"][normalized]

    # Intento secundario: eliminar espacios residuales
    ns = normalized.replace(" ", "")
    if ns in VOCABS["harmony"]:
        return VOCABS["harmony"][ns]

    if normalized not in ("[UNK]", ""):
        log.debug(f"Acorde no encontrado tras normalizar '{label}' → '{normalized}'")

    return VOCABS["harmony"].get("[UNK]", UNK_TOKEN_ID)


def _txt(el: Optional[ET.Element], tag: str, default: str = "") -> str:
    if el is None:
        return default
    child = el.find(tag)
    return (child.text or default) if child is not None else default


def _int(el: Optional[ET.Element], tag: str, default: int = 0) -> int:
    try:
        return int(_txt(el, tag, str(default)))
    except ValueError:
        return default


def _float(el: Optional[ET.Element], tag: str, default: float = 0.0) -> float:
    try:
        return float(_txt(el, tag, str(default)))
    except ValueError:
        return default


# ===========================================================================
# 4.  DESCUBRIMIENTO DE ARCHIVOS Y ESTRUCTURA DE CORPUS
# ===========================================================================

def discover_files(input_dir: Path, use_ms3: bool = True) -> List[Tuple[str, Path]]:
    """
    Recorre el corpus y devuelve una lista de (autor, path) para cada
    fichero .mscx / .mscz encontrado.

    Estructura esperada (use_ms3=True):
        <input_dir>/<autor>/MS3/<obra>.mscx
        <input_dir>/<autor>/MS3/<obra>.mscz

    Estructura alternativa (use_ms3=False):
        <input_dir>/<autor>/**/<obra>.mscx   (búsqueda recursiva)

    Si no se encuentra ningún fichero con la estructura MS3, se hace
    fallback automático a búsqueda recursiva completa con autor="unknown".
    """
    results: List[Tuple[str, Path]] = []

    if use_ms3:
        # Buscar en subdirectorios <autor>/MS3/
        for author_dir in sorted(input_dir.iterdir()):
            if not author_dir.is_dir():
                continue
            author = author_dir.name
            ms3_dir = author_dir / "MS3"
            if ms3_dir.is_dir():
                for ext in ("*.mscx", "*.mscz"):
                    for f in sorted(ms3_dir.glob(ext)):
                        results.append((author, f))
            else:
                # El autor no tiene carpeta MS3: buscar recursivamente dentro
                # de su directorio (tolerancia a estructuras ligeramente distintas)
                for ext in ("*.mscx", "*.mscz"):
                    for f in sorted(author_dir.rglob(ext)):
                        results.append((author, f))

    if not results:
        # Fallback: búsqueda recursiva completa sin estructura de autor
        log.warning(
            "No se encontraron ficheros con estructura <autor>/MS3/. "
            "Haciendo búsqueda recursiva completa (autor='unknown')."
        )
        for ext in ("*.mscx", "*.mscz"):
            for f in sorted(input_dir.rglob(ext)):
                results.append(("unknown", f))

    log.info(f"Ficheros encontrados: {len(results)}")
    authors_found = sorted(set(a for a, _ in results))
    log.info(f"Autores detectados ({len(authors_found)}): {', '.join(authors_found)}")
    return results


# ===========================================================================
# 5.  PARSER NATIVO DE .mscx / .mscz (ElementTree)
# ===========================================================================

def _get_safe_path(original: Path) -> Path:
    _UNSAFE = re.compile(r"[^A-Za-z0-9_\-.]")
    if not _UNSAFE.search(original.name):
        return original
    safe_dir  = Path(tempfile.mkdtemp(prefix="nbeat_"))
    safe_path = safe_dir / f"score{original.suffix}"
    shutil.copy2(str(original), str(safe_path))
    return safe_path


def _unpack_mscz(mscz_path: Path) -> Optional[Path]:
    try:
        with zipfile.ZipFile(mscz_path, "r") as zf:
            inner = [n for n in zf.namelist() if n.endswith(".mscx")]
            if not inner:
                log.warning(f"✗ {mscz_path.name}: sin .mscx en ZIP")
                return None
            tmp_dir = Path(tempfile.mkdtemp(prefix="nbeat_"))
            out_p   = tmp_dir / "score.mscx"
            out_p.write_bytes(zf.read(inner[0]))
            return out_p
    except Exception as e:
        log.warning(f"✗ descompresión {mscz_path.name}: {e}")
        return None


def _parse_mscx_xml(mscx_path: Path) -> Optional[ET.Element]:
    safe = _get_safe_path(mscx_path)
    try:
        tree = ET.parse(str(safe))
        return tree.getroot()
    except ET.ParseError:
        try:
            raw = safe.read_bytes()
            if raw.startswith(b"\xef\xbb\xbf"):   # BOM UTF-8
                raw = raw[3:]
            return ET.fromstring(raw)
        except Exception as e:
            log.warning(f"✗ XML parse {mscx_path.name}: {e}")
            return None
    except Exception as e:
        log.warning(f"✗ XML parse {mscx_path.name}: {e}")
        return None


# ===========================================================================
# 6.  EXTRACCIÓN DE DATOS DEL XML MUSESCORE
# ===========================================================================

def _get_instrument_program(root: ET.Element) -> Dict[int, int]:
    """Devuelve {part_idx: midi_program} para cada Part del Score."""
    progs: Dict[int, int] = {}
    score = root.find("Score") or root
    for i, part in enumerate(score.findall("Part")):
        instr = part.find("Instrument") or part.find(".//Instrument")
        prog  = 0
        if instr is not None:
            for tag in ["programValue", "midiProgram", "program"]:
                val = instr.find(tag)
                if val is not None and val.text:
                    try:
                        prog = int(val.text); break
                    except ValueError:
                        pass
        progs[i] = prog % 128
    return progs


def _leer_etiqueta_armonia(el_armonia: ET.Element) -> str:
    """Extrae el texto de una etiqueta <Harmony> probando name, function y root."""
    label = ""
    name_el = el_armonia.find("name")
    if name_el is not None and name_el.text:
        label = name_el.text.strip()
    if not label:
        func_el = el_armonia.find("function")
        if func_el is not None and func_el.text:
            label = func_el.text.strip()
    if not label:
        root_el = el_armonia.find("root")
        if root_el is not None and root_el.text:
            label = f"root{root_el.text.strip()}"
    return label


def _extract_harmony_from_xml(root: ET.Element) -> List[Tuple[float, float, str]]:
    """
    Recorre el XML y construye la línea temporal de armonía como lista de
    (onset_qn, offset_qn, label).  Cada acorde dura hasta el siguiente.
    """
    entries: List[Tuple[float, str]] = []
    score   = root.find("Score") or root

    staves_validos = [
        s for s in score.findall(".//Staff") if s.find(".//Measure") is not None
    ]
    if not staves_validos:
        return []

    for staff in staves_validos:
        cur_ts_num, cur_ts_den = 4, 4
        cur_bar_qn = 0.0
        for measure in staff.findall("Measure"):
            ts = measure.find(".//TimeSig")
            if ts is not None:
                n = _int(ts, "sigN", cur_ts_num)
                d = _int(ts, "sigD", cur_ts_den)
                if n > 0 and d > 0:
                    cur_ts_num, cur_ts_den = n, d
            bar_dur_qn = cur_ts_num * (4.0 / cur_ts_den)

            for voice in measure.findall("voice") + measure.findall("Voice"):
                local_qn = 0.0
                for el in voice:
                    if el.tag in ("Harmony", "harmony"):
                        label = _leer_etiqueta_armonia(el)
                        if label:
                            entries.append((cur_bar_qn + local_qn, label))
                    elif el.tag in ("Chord", "Rest"):
                        local_qn += _chord_duration_qn(el)

            # Armonías fuera de voice (nivel Measure)
            for h in measure.findall("Harmony") + measure.findall("harmony"):
                label = _leer_etiqueta_armonia(h)
                if label:
                    entries.append((cur_bar_qn, label))

            cur_bar_qn += bar_dur_qn

    # Deduplicar y ordenar
    seen   = set()
    unique: List[Tuple[float, str]] = []
    for onset, label in sorted(entries, key=lambda x: x[0]):
        key = (round(onset, 6), label)
        if key not in seen:
            seen.add(key)
            unique.append((onset, label))

    # Construir intervalos (onset, offset, label)
    result: List[Tuple[float, float, str]] = []
    for i, (onset, label) in enumerate(unique):
        offset = unique[i + 1][0] if i + 1 < len(unique) else float("inf")
        result.append((onset, offset, label))
    return result


def _chord_duration_qn(chord_el: ET.Element) -> float:
    """Calcula la duración en quarter-notes de un <Chord> o <Rest>, con puntos y tresillos."""
    dur_type = _txt(chord_el, "durationType", "quarter")
    base_qn  = DURATION_TYPE_TO_QN.get(dur_type, 1.0)

    dots = chord_el.find("dots")
    if dots is not None and dots.text:
        try:
            n_dots     = int(dots.text)
            dot_factor = sum(0.5**i for i in range(n_dots + 1))
            base_qn   *= dot_factor
        except ValueError:
            pass

    tuplet_el = chord_el.find("Tuplet")
    if tuplet_el is not None:
        actual = _int(tuplet_el, "actualNotes", 3)
        normal = _int(tuplet_el, "normalNotes", 2)
        if actual > 0:
            base_qn *= normal / actual

    return base_qn


def _extract_tempo_map(root: ET.Element) -> Dict[float, float]:
    """Mapa {quarter_note_pos → BPM} extraído de los elementos <Tempo> del XML."""
    tempo_map: Dict[float, float] = {0.0: 120.0}
    score       = root.find("Score") or root
    first_staff = score.find(".//Staff[@id='1']") or score.find(".//Staff")
    if first_staff is None:
        return tempo_map

    cur_ts_num, cur_ts_den = 4, 4
    cur_bar_qn = 0.0

    for measure in first_staff.findall("Measure"):
        ts = measure.find(".//TimeSig")
        if ts is not None:
            n = _int(ts, "sigN", cur_ts_num)
            d = _int(ts, "sigD", cur_ts_den)
            if n > 0 and d > 0:
                cur_ts_num, cur_ts_den = n, d
        bar_dur_qn = cur_ts_num * (4.0 / cur_ts_den)
        local_qn   = 0.0

        for voice in measure.findall("voice"):
            for el in voice:
                if el.tag == "Tempo":
                    t_el = el.find("tempo")
                    if t_el is not None and t_el.text:
                        try:
                            tempo_map[cur_bar_qn + local_qn] = float(t_el.text) * 60.0
                        except ValueError:
                            pass
                elif el.tag in ("Chord", "Rest"):
                    local_qn += _chord_duration_qn(el)

        # Tempos fuera de voice (nivel Measure)
        for t_el in measure.findall("Tempo"):
            t_val = t_el.find("tempo")
            if t_val is not None and t_val.text:
                try:
                    tempo_map[cur_bar_qn] = float(t_val.text) * 60.0
                except ValueError:
                    pass

        cur_bar_qn += bar_dur_qn

    return tempo_map


# ===========================================================================
# 7.  ESCANEO PREVIO DEL CORPUS PARA CONSTRUIR EL VOCAB DE ARMONÍA
# ===========================================================================

def _scan_harmony_labels(files: List[Tuple[str, Path]]) -> List[str]:
    """
    Primera pasada sobre el corpus: extrae TODAS las etiquetas <Harmony>
    en crudo para poder construir el vocabulario dinámicamente.
    Acepta la lista (autor, path) que devuelve discover_files().
    """
    all_labels: List[str] = []
    for _, path in files:
        work = path
        if path.suffix.lower() == ".mscz":
            try:
                with zipfile.ZipFile(path, "r") as zf:
                    inner = [n for n in zf.namelist() if n.endswith(".mscx")]
                    if not inner:
                        continue
                    tmp = Path(tempfile.mkdtemp(prefix="nbeat_scan_")) / "score.mscx"
                    tmp.write_bytes(zf.read(inner[0]))
                    work = tmp
            except Exception:
                continue
        try:
            root = ET.parse(str(work)).getroot()
        except Exception:
            try:
                raw = work.read_bytes()
                if raw.startswith(b"\xef\xbb\xbf"):
                    raw = raw[3:]
                root = ET.fromstring(raw)
            except Exception:
                continue

        for h in root.iter("Harmony"):
            label = _leer_etiqueta_armonia(h)
            if label:
                all_labels.append(label)

    return all_labels


def _print_harmony_summary(raw_labels: List[str]) -> None:
    """Imprime un resumen del vocabulario de armonía para el corpus actual."""
    normalized  = [normalize_label_lvl1(l) for l in raw_labels]
    unique_raw  = len(set(raw_labels))
    unique_norm = len(set(n for n in normalized if n and n != "[UNK]"))
    unk_count   = sum(1 for n in normalized if n == "[UNK]" or not n)
    unk_pct     = 100 * unk_count / max(len(raw_labels), 1)

    print("\n" + "=" * 55)
    print("  Resumen vocabulario de armonía (normalize_label_lvl1)")
    print("=" * 55)
    print(f"  Etiquetas raw    : {len(raw_labels):,}")
    print(f"  Únicas raw       : {unique_raw:,}")
    print(f"  Únicas norm      : {unique_norm:,}")
    print(f"  vocab_harmony    : {unique_norm + 5:,}  (+ 5 especiales)")
    print(f"  UNK estimado     : {unk_pct:.1f}%")
    print("=" * 55 + "\n")


# ===========================================================================
# 8.  CONVERSOR PRINCIPAL: .mscx/.mscz → NanoBeatSequence
# ===========================================================================

def mscx_to_nanobeat9(
    mscx_path: Path,
    author: str = "unknown",
) -> Optional[NanoBeatSequence]:
    """
    Convierte un fichero MuseScore en una NanoBeatSequence con tokens de 9 dimensiones:
        [Compás, Posición, Instrumento, Tono, Duración, Velocidad,
         Firma de tiempo, Tempo, Armonía]

    Args:
        mscx_path : ruta al fichero .mscx o .mscz
        author    : nombre del autor (extraído del directorio padre)
    """
    work_path = mscx_path
    if mscx_path.suffix.lower() == ".mscz":
        work_path = _unpack_mscz(mscx_path)
        if work_path is None:
            return None

    root = _parse_mscx_xml(work_path)
    if root is None:
        return None
    if root.tag not in ("museScore", "MuseScore"):
        log.warning(f"✗ {mscx_path.name}: raíz XML inesperada <{root.tag}>")
        return None

    inst_programs = _get_instrument_program(root)
    harmony_tl    = _extract_harmony_from_xml(root)
    tempo_map     = _extract_tempo_map(root)

    # ── Closures de lookup ────────────────────────────────────────────────────
    def _tempo_at(qn: float) -> float:
        keys = sorted(tempo_map.keys())
        bpm  = tempo_map[keys[0]]
        for k in keys:
            if k <= qn: bpm = tempo_map[k]
            else:       break
        return bpm

    def _harmony_at(qn: float) -> str:
        for onset, end, label in harmony_tl:
            if onset <= qn < end:
                return label
        return "[UNK]"

    # ── Construcción de tokens ────────────────────────────────────────────────
    tokens: List[List[int]] = []
    score  = root.find("Score") or root

    # Ruta principal: iterar por Parts (con información de instrumento)
    for part_idx, part in enumerate(score.findall("Part")):
        midi_prog = inst_programs.get(part_idx, 0)
        inst_enc  = VOCABS["instrument"].get(midi_prog % 128, UNK_TOKEN_ID)

        for staff in part.findall("Staff"):
            cur_ts_num, cur_ts_den = 4, 4
            cur_bar_qn  = 0.0
            bar_number  = 0

            for measure in staff.findall("Measure"):
                bar_number += 1
                bar_enc     = _enc_bar(bar_number - 1)

                ts_el = measure.find(".//TimeSig")
                if ts_el is not None:
                    n = _int(ts_el, "sigN", cur_ts_num)
                    d = _int(ts_el, "sigD", cur_ts_den)
                    if n > 0 and d > 0:
                        cur_ts_num, cur_ts_den = n, d
                ts_enc     = _enc_timesig(cur_ts_num, cur_ts_den)
                bar_dur_qn = cur_ts_num * (4.0 / cur_ts_den)

                for voice in measure.findall("voice"):
                    local_qn = 0.0
                    for el in voice:
                        if el.tag == "Chord":
                            dur_qn    = _chord_duration_qn(el)
                            global_qn = cur_bar_qn + local_qn
                            for note_el in el.findall("Note"):
                                pitch_val = _int(note_el, "pitch", -1)
                                if not (0 <= pitch_val <= 127):
                                    continue
                                vel_val = _int(note_el, "velocity", 64)
                                if vel_val <= 0:
                                    vel_val = 64
                                tokens.append(NanoBeat9(
                                    bar        = bar_enc,
                                    position   = _enc_position(local_qn),
                                    instrument = inst_enc,
                                    pitch      = VOCABS["pitch"].get(pitch_val, UNK_TOKEN_ID),
                                    duration   = _enc_duration(dur_qn),
                                    velocity   = _enc_velocity(vel_val),
                                    timesig    = ts_enc,
                                    tempo      = _enc_tempo(_tempo_at(global_qn)),
                                    harmony    = _enc_harmony(_harmony_at(global_qn)),
                                ).to_list())
                            local_qn += dur_qn
                        elif el.tag == "Rest":
                            local_qn += _chord_duration_qn(el)

                cur_bar_qn += bar_dur_qn

    # Fallback: si no hay <Part> (estructuras no estándar de MuseScore)
    if not tokens:
        midi_fallback = [52, 40, 41, 42, 43, 6, 73, 68]
        for inst_idx, staff in enumerate(score.findall(".//Staff")):
            inst_enc   = VOCABS["instrument"].get(
                midi_fallback[inst_idx % len(midi_fallback)], UNK_TOKEN_ID
            )
            cur_ts_num, cur_ts_den = 4, 4
            cur_bar_qn  = 0.0
            bar_number  = 0

            for measure in staff.findall("Measure"):
                bar_number += 1
                bar_enc     = _enc_bar(bar_number - 1)

                ts_el = measure.find(".//TimeSig")
                if ts_el is not None:
                    n = _int(ts_el, "sigN", 4)
                    d = _int(ts_el, "sigD", 4)
                    if n > 0 and d > 0:
                        cur_ts_num, cur_ts_den = n, d
                ts_enc     = _enc_timesig(cur_ts_num, cur_ts_den)
                bar_dur_qn = cur_ts_num * (4.0 / cur_ts_den)

                for voice in measure.findall("voice"):
                    local_qn = 0.0
                    for el in voice:
                        if el.tag == "Chord":
                            dur_qn    = _chord_duration_qn(el)
                            global_qn = cur_bar_qn + local_qn
                            for note_el in el.findall("Note"):
                                pitch_val = _int(note_el, "pitch", -1)
                                if not (0 <= pitch_val <= 127):
                                    continue
                                tokens.append(NanoBeat9(
                                    bar        = bar_enc,
                                    position   = _enc_position(local_qn),
                                    instrument = inst_enc,
                                    pitch      = VOCABS["pitch"].get(pitch_val, UNK_TOKEN_ID),
                                    duration   = _enc_duration(dur_qn),
                                    velocity   = _enc_velocity(_int(note_el, "velocity", 64)),
                                    timesig    = ts_enc,
                                    tempo      = _enc_tempo(_tempo_at(global_qn)),
                                    harmony    = _enc_harmony(_harmony_at(global_qn)),
                                ).to_list())
                            local_qn += dur_qn
                        elif el.tag == "Rest":
                            local_qn += _chord_duration_qn(el)

                cur_bar_qn += bar_dur_qn

    if not tokens:
        log.warning(f"✗ Sin tokens: {mscx_path.name}")
        return None

    tokens.sort(key=lambda x: (x[0], x[1], x[3]))   # (compás, posición, pitch)

    n_unk = sum(1 for t in tokens if t[8] == VOCABS["harmony"].get("[UNK]", UNK_TOKEN_ID))
    pct   = round(100 * (1 - n_unk / len(tokens)), 1)

    return NanoBeatSequence(
        piece_id = mscx_path.stem,
        author   = author,
        tokens   = tokens,
        metadata = {
            "source":               str(mscx_path),
            "author":               author,
            "n_tokens":             len(tokens),
            "n_harmony_labels":     len(harmony_tl),
            "pct_harmony_coverage": pct,
            "vocab_sizes":          VOCAB_SIZES,
        },
    )


# ===========================================================================
# 9.  PIPELINE COMPLETO
# ===========================================================================

def convert_corpus(
    input_dir:   str,
    output_path: str,
    use_ms3:     bool  = True,
    split_ratio: Tuple[float, float, float] = (0.8, 0.1, 0.1),
    max_seq_len: int   = 512,
) -> None:
    """
    Pipeline completo: descubre ficheros → escanea armonías → convierte → guarda.

    Args:
        input_dir   : directorio raíz del corpus (contiene subdirectorios de autor)
        output_path : ruta del fichero JSONL de salida
        use_ms3     : si True, busca en <autor>/MS3/; si False, búsqueda recursiva
        split_ratio : proporciones train/val/test (reservado para uso futuro)
        max_seq_len : longitud máxima de secuencia (reservado para uso futuro)
    """
    input_p  = Path(input_dir)
    output_p = Path(output_path)
    output_p.parent.mkdir(parents=True, exist_ok=True)

    # ── Paso 1: descubrir ficheros ────────────────────────────────────────────
    files = discover_files(input_p, use_ms3=use_ms3)
    if not files:
        log.error("Sin ficheros. Revisa --input_dir.")
        return

    # ── Paso 2: escaneo previo para construir el vocab de armonía ────────────
    log.info("Escaneando corpus para construir vocabulario de armonía...")
    raw_labels = _scan_harmony_labels(files)
    log.info(
        f"Etiquetas armónicas encontradas: {len(raw_labels):,}  "
        f"(únicas: {len(set(raw_labels)):,})"
    )

    if not raw_labels:
        log.warning(
            "⚠ No se encontraron etiquetas <Harmony> en ningún fichero.\n"
            "  Asegúrate de que tus partituras tienen anotaciones armónicas."
        )

    # ── Paso 3: resumen del vocabulario ──────────────────────────────────────
    _print_harmony_summary(raw_labels)

    # ── Paso 4: construir el vocab con normalize_label_lvl1 ──────────────────
    VOCABS["harmony"] = build_harmony_vocab_from_labels(raw_labels)
    VOCAB_SIZES["harmony"] = max(VOCABS["harmony"].values()) + 1
    log.info(
        f"vocab_harmony = {VOCAB_SIZES['harmony']}  "
        f"(usar este valor en HarmonyBERTConfig)"
    )

    # ── Paso 5: convertir cada fichero ───────────────────────────────────────
    sequences: List[NanoBeatSequence] = []
    n_ok = n_fail = n_noharm = 0

    for author, path in tqdm(files, desc=".mscx → NanoBeat-9"):
        seq = mscx_to_nanobeat9(path, author=author)
        if seq is None:
            n_fail += 1
            continue
        if seq.metadata.get("n_harmony_labels", 0) == 0:
            n_noharm += 1
        sequences.append(seq)
        n_ok += 1

    log.info(f"OK={n_ok}  fallos={n_fail}  sin_armonía={n_noharm}")
    if not sequences:
        log.error("Sin secuencias. Abortando.")
        return

    # ── Paso 6: guardar corpus JSONL ──────────────────────────────────────────
    with open(output_p, "w", encoding="utf-8") as f:
        for seq in sequences:
            f.write(json.dumps({
                "piece_id": seq.piece_id,
                "author":   seq.author,
                "tokens":   seq.tokens,
                "metadata": seq.metadata,
            }, ensure_ascii=False) + "\n")

    log.info(f"Corpus guardado: {output_p}  ({len(sequences)} piezas)")
    save_vocabs(str(output_p.parent / "vocabs.json"))

    # ── Paso 7: resumen por autor ─────────────────────────────────────────────
    author_counts: Dict[str, int] = {}
    for seq in sequences:
        author_counts[seq.author] = author_counts.get(seq.author, 0) + 1
    print("\n" + "=" * 45)
    print("  Piezas procesadas por autor")
    print("=" * 45)
    for a, count in sorted(author_counts.items()):
        print(f"  {a:<30s} {count:>4d}")
    print("=" * 45 + "\n")


# ===========================================================================
# 10.  CLI
# ===========================================================================

if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description=(
            "Convierte partituras MuseScore (.mscx/.mscz) con anotaciones\n"
            "armónicas al formato NanoBeat-9.\n\n"
            "Estructura esperada del corpus:\n"
            "  <input_dir>/<autor>/MS3/<obra>.mscx"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--input_dir",
        required=True,
        help="Directorio raíz del corpus (contiene subdirectorios por autor)",
    )
    p.add_argument(
        "--output",
        default="corpus/nanobeat.jsonl",
        help="Ruta del fichero JSONL de salida  (default: corpus/nanobeat.jsonl)",
    )
    p.add_argument(
        "--no_ms3",
        action="store_true",
        help="No buscar en subcarpeta MS3: usar búsqueda recursiva completa",
    )
    p.add_argument(
        "--split_ratio",
        nargs=3, type=float,
        default=[0.8, 0.1, 0.1],
        metavar=("TR", "VA", "TE"),
        help="Proporción train/val/test (reservado para uso futuro)",
    )
    p.add_argument(
        "--max_seq_len",
        type=int,
        default=512,
        help="Longitud máxima de secuencia (reservado para uso futuro)",
    )
    args = p.parse_args()

    convert_corpus(
        input_dir   = args.input_dir,
        output_path = args.output,
        use_ms3     = not args.no_ms3,
        split_ratio = tuple(args.split_ratio),
        max_seq_len = args.max_seq_len,
    )
