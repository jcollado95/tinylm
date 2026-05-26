"""
dataset.py — Carga y preparación del corpus
============================================
Fuente: Fernandoefg/cuentos_es (HuggingFace Hub)
        https://huggingface.co/datasets/Fernandoefg/cuentos_es

Estrategia de carga
-------------------
Siempre usa la librería `datasets` de HuggingFace.

- Con conexión:    descarga y cachea el dataset en ~/.cache/huggingface/datasets
- Sin conexión:    lee desde esa misma caché activando HF_DATASETS_OFFLINE=1,
                   que instruye a la librería a no intentar ninguna conexión.

Para instalar la dependencia:
    pip install datasets
"""

import os
import re


# ── Carga del dataset ────────────────────────────────────────

def _load_from_datasets(max_stories: int | None = None) -> str:
    """
    Carga el dataset con la librería HuggingFace `datasets`.

    Si no hay conexión a internet, activa HF_DATASETS_OFFLINE=1 antes
    de la llamada para que la librería lea directamente desde la caché
    sin intentar contactar con el Hub. Si tampoco hay caché, lanza un
    error descriptivo.
    """
    # Activar modo offline preventivamente — si hay conexión, la librería
    # la ignorará y descargará igualmente; si no la hay, usará la caché.
    os.environ["HF_DATASETS_OFFLINE"] = "1"

    try:
        from datasets import load_dataset
    except ModuleNotFoundError:
        raise ModuleNotFoundError(
            "La librería 'datasets' no está instalada.\n"
            "Instálala con:  pip install datasets"
        )

    print("  Cargando dataset (datasets + caché local)...")
    try:
        ds = load_dataset("Fernandoefg/cuentos_es", split="train")
    except Exception as e:
        raise RuntimeError(
            f"No se pudo cargar el dataset desde la caché: {e}\n"
            "Asegúrate de haber descargado el dataset previamente con conexión.\n"
            "Puedes hacerlo ejecutando una vez desde una máquina con internet:\n"
            "  python -c \"from datasets import load_dataset; "
            "load_dataset('Fernandoefg/cuentos_es')\""
        ) from e

    if max_stories is not None:
        ds = ds.select(range(min(max_stories, len(ds))))

    text_col = _detect_text_column(ds.column_names)
    print(f"  {len(ds)} cuentos | columna: '{text_col}'")

    texts = [str(row[text_col]) for row in ds if row[text_col]]
    return texts


def _detect_text_column(columns: list[str]) -> str:
    """Detecta la columna de texto por nombre probable."""
    candidates = ["text", "texto", "story", "cuento", "content", "contenido"]
    for c in candidates:
        if c in columns:
            return c
    return columns[0]


# ── Limpieza de texto ────────────────────────────────────────

def clean_text(text: str) -> str:
    """
    Limpieza básica del corpus:
      - Normaliza comillas tipográficas y guiones largos
      - Colapsa espacios múltiples y saltos de línea excesivos
      - Elimina líneas que sean solo números (índices, numeración)
    """
    text = text.replace("\u201c", '"').replace("\u201d", '"')
    text = text.replace("\u2018", "'").replace("\u2019", "'")
    text = text.replace("«", '"').replace("»", '"')
    text = text.replace("\u2014", "-").replace("\u2013", "-")

    text = re.sub(r"[^\S\n]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"^\s*\d+\s*$", "", text, flags=re.MULTILINE)

    return text.strip()


# ── API pública ──────────────────────────────────────────────

def load_corpus(
    local_path: str | None = None,   # ignorado, mantenido por compatibilidad
    max_stories: int | None = None,
) -> list[str]:
    """
    Carga y limpia el corpus de cuentos.

    Parámetros
    ----------
    local_path  : ignorado (mantenido por compatibilidad con versiones anteriores)
    max_stories : número máximo de cuentos a cargar (None = todos)

    Retorna
    -------
    stories : lista de strings, uno por cuento, ya limpios
    """
    if local_path is not None:
        print("  [Nota] local_path ignorado — usando siempre datasets + caché HuggingFace")

    stories = [clean_text(s) for s in _load_from_datasets(max_stories=max_stories) if s.strip()]

    n_chars = sum(len(s) for s in stories)
    n_words = sum(len(s.split()) for s in stories)
    print(f"  Corpus listo: {len(stories)} cuentos | "
          f"{n_chars:,} caracteres | {n_words:,} palabras")
    return stories


# ── Preparación de tensores ──────────────────────────────────

def build_dataset(ids: list[int], block_size: int) -> tuple:
    """
    Crea pares (x, y) deslizando una ventana de tamaño block_size.

      x[i] = ids[i   : i +   block_size]   ← entrada
      y[i] = ids[i+1 : i + 1+block_size]   ← objetivo (siguiente token)

    Retorna
    -------
    X, Y : tensores de forma (N, block_size)
    """
    import torch
    data = torch.tensor(ids, dtype=torch.long)
    n = len(data) - block_size

    if n <= 0:
        raise ValueError(
            f"El corpus tokenizado ({len(data)} tokens) es demasiado corto "
            f"para block_size={block_size}."
        )

    X = torch.stack([data[i     : i +     block_size] for i in range(n)])
    Y = torch.stack([data[i + 1 : i + 1 + block_size] for i in range(n)])
    return X, Y


def get_batch(
    X: "torch.Tensor",
    Y: "torch.Tensor",
    batch_size: int,
    device: str,
) -> tuple:
    """Extrae un minibatch aleatorio de (X, Y)."""
    import torch
    idx = torch.randint(len(X), (batch_size,))
    return X[idx].to(device), Y[idx].to(device)