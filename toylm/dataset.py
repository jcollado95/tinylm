"""
dataset.py — Carga y preparación del corpus
============================================
Fuente: Fernandoefg/cuentos_es (HuggingFace Hub)
        https://huggingface.co/datasets/Fernandoefg/cuentos_es

Estrategia de carga
-------------------
1. Intenta cargar desde HuggingFace Hub con `datasets`.
2. Si falla (sin conexión), carga desde un archivo .txt local
   llamado "cuentos_es.txt" en el mismo directorio.
3. El texto se limpia y se devuelve como un único string.
"""

import os
import re


# ── Descarga desde HuggingFace ───────────────────────────────

def _load_from_hub(max_stories: int | None = None) -> str:
    """
    Descarga el dataset desde HuggingFace Hub.
    Requiere: pip install datasets
    """
    from datasets import load_dataset

    print("Descargando dataset desde HuggingFace Hub...")
    ds = load_dataset("Fernandoefg/cuentos_es", split="train")

    if max_stories is not None:
        ds = ds.select(range(min(max_stories, len(ds))))

    print(f"  {len(ds)} cuentos cargados")
    print(f"  Columnas disponibles: {ds.column_names}")

    # Detectar columna de texto automáticamente
    text_col = _detect_text_column(ds.column_names)
    print(f"  Columna de texto: '{text_col}'")

    texts = [str(row[text_col]) for row in ds if row[text_col]]
    return "\n\n".join(texts)


def _detect_text_column(columns: list[str]) -> str:
    """Detecta la columna de texto por nombre probable."""
    candidates = ["text", "texto", "story", "cuento", "content", "contenido"]
    for c in candidates:
        if c in columns:
            return c
    # Si no encuentra ninguna conocida, devuelve la primera
    return columns[0]


# ── Carga desde archivo local ────────────────────────────────

def _load_from_file(path: str) -> str:
    """Carga texto desde un archivo .txt local."""
    print(f"Cargando corpus desde archivo local: {path}")
    with open(path, encoding="utf-8") as f:
        return f.read()


# ── Limpieza de texto ────────────────────────────────────────

def clean_text(text: str) -> str:
    """
    Limpieza básica del corpus:
      - Normaliza saltos de línea múltiples
      - Elimina caracteres de control y no imprimibles
      - Normaliza comillas tipográficas a estándar
      - Conserva: letras (incluye acentos/ñ), puntuación básica, espacios
    """
    # Normalizar comillas tipográficas
    text = text.replace("\u201c", '"').replace("\u201d", '"')
    text = text.replace("\u2018", "'").replace("\u2019", "'")
    text = text.replace("«", '"').replace("»", '"')

    # Normalizar guiones largos
    text = text.replace("\u2014", "-").replace("\u2013", "-")

    # Eliminar caracteres de control (excepto \n)
    text = re.sub(r"[^\S\n]+", " ", text)          # espacios múltiples → uno
    text = re.sub(r"\n{3,}", "\n\n", text)          # saltos excesivos → doble

    # Eliminar líneas que sean solo números (índices, numeración)
    text = re.sub(r"^\s*\d+\s*$", "", text, flags=re.MULTILINE)

    return text.strip()


# ── API pública ──────────────────────────────────────────────

def load_corpus(
    local_path: str | None = None,
    max_stories: int | None = None,
) -> str:
    """
    Carga el corpus con fallback automático.

    Parámetros
    ----------
    local_path  : ruta a un .txt local (opcional, se usa si Hub falla)
    max_stories : límite de cuentos a cargar desde Hub (None = todos)

    Retorna
    -------
    text : string con todo el corpus limpio
    """
    text = None

    # 1. Intentar Hub
    try:
        text = _load_from_hub(max_stories=max_stories)
    except Exception as e:
        print(f"  [Hub no disponible: {e}]")

    # 2. Fallback a archivo local
    if text is None:
        candidates = [
            local_path,
            "cuentos_es.txt",
            os.path.join(os.path.dirname(__file__), "cuentos_es.txt"),
        ]
        for path in candidates:
            if path and os.path.exists(path):
                text = _load_from_file(path)
                break

    if text is None:
        raise FileNotFoundError(
            "No se pudo cargar el corpus.\n"
            "Opciones:\n"
            "  1. Ejecuta con conexión a Internet para descargarlo de HuggingFace.\n"
            "  2. Coloca un archivo 'cuentos_es.txt' en el directorio del proyecto."
        )

    text = clean_text(text)

    n_chars  = len(text)
    n_words  = len(text.split())
    n_lines  = text.count("\n")
    print(f"  Corpus listo: {n_chars:,} caracteres | "
          f"{n_words:,} palabras | "
          f"{n_lines:,} líneas")

    return text


# ── Preparación de tensores ──────────────────────────────────

def build_dataset(
    ids: list[int],
    block_size: int,
) -> tuple:
    """
    Crea pares (x, y) deslizando una ventana de tamaño block_size.

      x[i] = ids[i : i+block_size]          ← entrada
      y[i] = ids[i+1 : i+block_size+1]      ← objetivo (siguiente token)

    Retorna
    -------
    X, Y : tensores de forma (N, block_size)
    """
    import torch
    data = torch.tensor(ids, dtype=torch.long)
    n    = len(data) - block_size

    if n <= 0:
        raise ValueError(
            f"El corpus tokenizado ({len(data)} tokens) es demasiado corto "
            f"para block_size={block_size}. Usa un corpus más grande o "
            f"reduce block_size."
        )

    X = torch.stack([data[i     : i + block_size]     for i in range(n)])
    Y = torch.stack([data[i + 1 : i + block_size + 1] for i in range(n)])
    return X, Y


def get_batch(
    X,          # torch.Tensor
    Y,          # torch.Tensor
    batch_size: int,
    device:     str,
) -> tuple:
    """Extrae un minibatch aleatorio de (X, Y)."""
    import torch
    idx = torch.randint(len(X), (batch_size,))
    return X[idx].to(device), Y[idx].to(device)