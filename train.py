"""
train.py — Script principal de entrenamiento de TinyLM
=======================================================
Uso:
    python train.py                        # entrenamiento completo
    python train.py --max_stories 200      # solo 200 cuentos
    python train.py --resume               # continuar desde checkpoint

Estructura del proyecto
-----------------------
tinylm/
  tokenizer.py   — BPETokenizer
  model.py       — TinyLM + Config
  dataset.py     — carga del corpus y preparación de tensores
train.py         — este archivo
"""

import os
import argparse
import torch

from tinylm.tokenizer import BPETokenizer
from tinylm.model     import TinyLM, Config
from tinylm.dataset   import load_corpus, build_dataset, get_batch


# ─────────────────────────────────────────────────────────────
# HIPERPARÁMETROS
# ─────────────────────────────────────────────────────────────

VOCAB_SIZE   = 512   # tamaño del vocabulario BPE
BLOCK_SIZE   = 128    # longitud de contexto (tokens)
N_EMBD       = 96    # dimensión del embedding
N_HEAD       = 4      # cabezas de atención
N_LAYER      = 4      # bloques Transformer
DROPOUT      = 0.1

BATCH_SIZE   = 64
MAX_ITERS    = 5000
LR           = 3e-3
EVAL_EVERY   = 500    # imprimir loss cada N pasos
EVAL_ITERS   = 50     # pasos para estimar loss de validación
VAL_SPLIT    = 0.1    # fracción del corpus para validación

TOKENIZER_PATH   = "tokenizer.json"
CHECKPOINT_PATH  = "checkpoint.pt"


# -----------------------------------------------------------
# WORKER DE TOKENIZACION (nivel de modulo, requerido por multiprocessing)
# -----------------------------------------------------------

def _tokenize_chunk(args: tuple) -> list:
    # Tokeniza un chunk de cuentos en un proceso worker.
    # Recibe el tokenizador serializado como JSON para evitar
    # problemas de pickling con metodos de instancia.
    chunk, tok_data = args
    import json
    from tinylm.tokenizer import BPETokenizer
    tok = BPETokenizer.load_from_dict(json.loads(tok_data))
    ids = []
    for cuento in chunk:
        ids.extend(tok.encode_with_special(cuento))
    return ids


# ─────────────────────────────────────────────────────────────
# UTILIDADES
# ─────────────────────────────────────────────────────────────

@torch.no_grad()
def estimate_loss(
    model: TinyLM,
    splits: dict[str, tuple],
    batch_size: int,
    device: str,
    eval_iters: int,
) -> dict[str, float]:
    """Estima la pérdida promedio sobre eval_iters batches."""
    model.eval()
    results = {}
    for split, (X, Y) in splits.items():
        losses = []
        for _ in range(eval_iters):
            xb, yb = get_batch(X, Y, batch_size, device)
            _, loss = model(xb, yb)
            losses.append(loss.item())
        results[split] = sum(losses) / len(losses)
    model.train()
    return results


def save_checkpoint(model, optimizer, step, loss, path):
    torch.save({
        "step":                step,
        "model_state_dict":    model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss":                loss,
        "config":              model.config,
    }, path)


def load_checkpoint(model, optimizer, path, device):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    return ckpt["step"], ckpt["loss"]


# ─────────────────────────────────────────────────────────────
# GENERACIÓN  — muestra texto durante el entrenamiento
# ─────────────────────────────────────────────────────────────

def sample(
    model: TinyLM,
    tokenizer: BPETokenizer,
    prompts: list[str],
    device: str,
    max_new_tokens: int = 60,
    temperature: float = 0.8,
    top_k: int = 30,
) -> None:
    model.eval()
    print()
    for prompt in prompts:
        ids = tokenizer.encode(prompt)
        idx = torch.tensor([ids], dtype=torch.long, device=device)
        out = model.generate(
            idx,
            max_new_tokens  = max_new_tokens,
            eos_id          = tokenizer.eos_id,
            temperature     = temperature,
            top_k           = top_k,
        )
        text = tokenizer.decode(out[0].tolist())
        print(f"  [{prompt!r}] → {text}")
    print()
    model.train()


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Entrenar TinyLM")
    parser.add_argument("--max_stories", type=int, default=None,
                        help="Número máximo de cuentos a cargar (default: todos)")
    parser.add_argument("--resume", action="store_true",
                        help="Reanudar desde checkpoint.pt si existe")
    parser.add_argument("--max_iters", type=int, default=MAX_ITERS)
    parser.add_argument("--vocab_size", type=int, default=VOCAB_SIZE)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n{'='*55}")
    print(f"  TinyLM — Entrenamiento")
    print(f"{'='*55}")
    print(f"  Dispositivo : {device}")

    # ── 1. Corpus ────────────────────────────────────────────
    print("\n[1/4] Cargando corpus...")
    corpus = load_corpus(max_stories=args.max_stories)

    # ── 2. Tokenizador ───────────────────────────────────────
    print("\n[2/4] Tokenizador...")
    if os.path.exists(TOKENIZER_PATH):
        tokenizer = BPETokenizer.load(TOKENIZER_PATH)
        if tokenizer.vocab_size != args.vocab_size:
            print(f"  Vocab size no coincide ({tokenizer.vocab_size} vs {args.vocab_size}), reentrenando...")
            os.remove(TOKENIZER_PATH)
            tokenizer = None
        else:
            print(f"  Cargado desde {TOKENIZER_PATH} (vocab={tokenizer.vocab_size})")
    else:
        tokenizer = None

    if tokenizer is None:
        print(f"  Entrenando BPE (vocab_size={args.vocab_size})...")
        tokenizer = BPETokenizer()
        tokenizer.train("\n\n".join(corpus), vocab_size=args.vocab_size, verbose=True)
        tokenizer.save(TOKENIZER_PATH)
        print(f"  Guardado en {TOKENIZER_PATH}")

    # Prueba de codificación
    prueba = "érase una vez un lobo que vivía en el bosque."
    ids_p  = tokenizer.encode(prueba)
    toks_p = [tokenizer.id_to_token[i] for i in ids_p]
    print(f"\n  Ejemplo: {prueba!r}")
    print(f"  Tokens : {toks_p}")
    print(f"  Decode : {tokenizer.decode(ids_p)!r}")

    # ── 3. Dataset ───────────────────────────────────────────
    print("\n[3/4] Preparando dataset...")

    import hashlib, pickle
    from multiprocessing import Pool, cpu_count

    cuentos = corpus   # ya es list[str], sin split frágil

    # Clave de caché: número real de cuentos + vocab
    cache_key = hashlib.md5((str(len(cuentos)) + str(tokenizer.vocab_size)).encode()).hexdigest()[:12]
    cache_path = f"{TOKENIZER_PATH.replace('.json', '')}_{cache_key}_ids.pkl"

    if os.path.exists(cache_path):
        print(f"  Cargando tokens desde caché ({cache_path})...")
        with open(cache_path, "rb") as f:
            all_ids = pickle.load(f)
    else:
        n_workers = min(cpu_count(), 8)
        print(f"  Tokenizando {len(cuentos)} cuentos con {n_workers} workers...")

        # Serializar el tokenizador para pasarlo a los workers
        import json as _json
        tok_data = _json.dumps({"vocab": tokenizer.vocab, "merges": tokenizer.merges})

        # Dividir en chunks (un chunk por worker)
        chunk_size = max(1, len(cuentos) // n_workers)
        chunks = [cuentos[i:i+chunk_size] for i in range(0, len(cuentos), chunk_size)]
        chunk = [(chunk, tok_data) for chunk in chunks]

        with Pool(n_workers) as pool:
            results = pool.map(_tokenize_chunk, chunk)

        all_ids = [tok_id for chunk_ids in results for tok_id in chunk_ids]

        with open(cache_path, "wb") as f:
            pickle.dump(all_ids, f)
        print(f"  Tokens cacheados en {cache_path}")

    print(f"  {len(cuentos)} cuentos | {len(all_ids):,} tokens totales")


    # Split train / validación
    split_idx = int(len(all_ids) * (1 - VAL_SPLIT))
    X_tr, Y_tr = build_dataset(all_ids[:split_idx],      BLOCK_SIZE)
    X_va, Y_va = build_dataset(all_ids[split_idx:],      BLOCK_SIZE)
    print(f"  Train: {len(X_tr):,} secuencias | Val: {len(X_va):,} secuencias")

    splits = {"train": (X_tr, Y_tr), "val": (X_va, Y_va)}

    # ── 4. Modelo ────────────────────────────────────────────
    print("\n[4/4] Modelo...")
    config = Config(
        vocab_size = tokenizer.vocab_size,
        n_embd     = N_EMBD,
        n_head     = N_HEAD,
        n_layer    = N_LAYER,
        block_size = BLOCK_SIZE,
        dropout    = DROPOUT,
    )
    model = TinyLM(config).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LR, weight_decay=0.01
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, args.max_iters
    )

    start_step = 0
    if args.resume and os.path.exists(CHECKPOINT_PATH):
        print(f"  Reanudando desde {CHECKPOINT_PATH}...")
        start_step, _ = load_checkpoint(model, optimizer, CHECKPOINT_PATH, device)
        print(f"  Continuando desde paso {start_step}")

    # ── Entrenamiento ────────────────────────────────────────
    print(f"\n{'─'*55}")
    print(f"  Entrenando {args.max_iters} pasos | "
          f"batch={BATCH_SIZE} | lr={LR}")
    print(f"{'─'*55}\n")

    # Prompts de muestra para monitorear la generación
    sample_prompts = ["érase una vez", "el lobo", "la princesa"]

    model.train()
    for step in range(start_step + 1, args.max_iters + 1):
        xb, yb = get_batch(X_tr, Y_tr, BATCH_SIZE, device)
        _, loss = model(xb, yb)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        if step % EVAL_EVERY == 0:
            metrics = estimate_loss(model, splits, BATCH_SIZE, device, EVAL_ITERS)
            lr_now  = scheduler.get_last_lr()[0]
            print(f"  Paso {step:5d}/{args.max_iters} | "
                  f"train={metrics['train']:.4f} | "
                  f"val={metrics['val']:.4f} | "
                  f"lr={lr_now:.2e}")

            # Mostrar ejemplos de generación
            sample(model, tokenizer, sample_prompts, device)

            # Guardar checkpoint
            save_checkpoint(model, optimizer, step, metrics["val"], CHECKPOINT_PATH)

    # ── Generación final ─────────────────────────────────────
    print(f"\n{'='*55}")
    print("  GENERACIÓN FINAL")
    print(f"{'='*55}")

    model.eval()
    prompts_finales = [
        "érase una vez",
        "el príncipe",
        "había una vez una niña",
        "en el bosque",
        "el rey dijo",
    ]

    for prompt in prompts_finales:
        ids = tokenizer.encode(prompt)
        idx = torch.tensor([ids], dtype=torch.long, device=device)
        out = model.generate(
            idx,
            max_new_tokens = 80,
            eos_id         = tokenizer.eos_id,
            temperature    = 0.85,
            top_k          = 40,
        )
        # Separar el prompt del texto generado para mostrarlo claramente
        n_prompt = len(ids)
        texto_completo  = tokenizer.decode(out[0].tolist())
        texto_generado  = tokenizer.decode(out[0][n_prompt:].tolist())

        print(f"\n  Prompt    : {prompt!r}")
        print(f"  Generado  : {texto_generado.strip()!r}")
        print(f"  Completo  : {texto_completo!r}")

    print(f"\n  Checkpoint final guardado en: {CHECKPOINT_PATH}")
    print(f"  Tokenizador guardado en:      {TOKENIZER_PATH}")
    print("\n¡Entrenamiento completado!\n")


if __name__ == "__main__":
    main()