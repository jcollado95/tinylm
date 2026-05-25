"""
tokenizer.py — Tokenizador BPE (Byte Pair Encoding) desde cero
===============================================================
Soporta:
  - Caracteres especiales del español (á, é, ñ, ¿, ¡, etc.)
  - Puntuación como tokens propios (., ,, ;, :, !, ?, ...)
  - Token <eos> para marcar fin de secuencia (usado en generate)
"""

import re
import json
from collections import Counter


class BPETokenizer:
    """
    Byte Pair Encoding tokenizer minimalista.

    Flujo de entrenamiento
    ----------------------
    1. Pre-tokenización: separa palabras Y signos de puntuación como unidades
       independientes. Cada palabra lleva el prefijo Ġ (convenio GPT-2) para
       poder reconstruir los espacios al decodificar.
    2. Vocabulario base: todos los caracteres únicos del corpus + especiales.
    3. Iteraciones BPE: fusiona el par más frecuente hasta alcanzar vocab_size.
    4. Codificación: aplica las fusiones aprendidas de forma greedy.
    """

    # ── Tokens especiales ────────────────────────────────────
    PAD = "<pad>"
    UNK = "<unk>"
    BOS = "<bos>"
    EOS = "<eos>"
    SPECIAL = [PAD, UNK, BOS, EOS]

    def __init__(self):
        self.vocab:        dict[str, int]          = {}
        self.id_to_token:  dict[int, str]          = {}
        self.merges:       list[tuple[str, str]]   = []
        self._merge_set:   dict[tuple, str]        = {}

    # ── Entrenamiento ────────────────────────────────────────

    def train(self, text: str, vocab_size: int, verbose: bool = True) -> None:
        """Aprende un vocabulario BPE de tamaño vocab_size sobre text."""

        units = self._pretokenize(text)

        # Frecuencia de cada unidad (palabra con Ġ ó signo de puntuación)
        word_freqs: dict[tuple, int] = Counter(
            tuple(u) for u in units
        )

        # Vocabulario base: todos los caracteres únicos
        base_chars: set[str] = set()
        for word in word_freqs:
            base_chars.update(word)

        all_tokens = self.SPECIAL + sorted(base_chars)
        self.vocab        = {tok: i for i, tok in enumerate(all_tokens)}
        self.id_to_token  = {i: tok for tok, i in self.vocab.items()}

        if verbose:
            print(f"  Vocabulario base: {len(self.vocab)} tokens")

        # Iteraciones BPE
        n_merges = vocab_size - len(self.vocab)
        for step in range(n_merges):
            pair_freqs = self._count_pairs(word_freqs)
            if not pair_freqs:
                break

            best_pair  = max(pair_freqs, key=pair_freqs.get)
            new_token  = "".join(best_pair)

            self.merges.append(best_pair)
            self._merge_set[best_pair] = new_token

            new_id = len(self.vocab)
            self.vocab[new_token]       = new_id
            self.id_to_token[new_id]    = new_token

            word_freqs = self._apply_merge(word_freqs, best_pair, new_token)

            if verbose and (step + 1) % 200 == 0:
                print(f"  Paso {step+1:4d}/{n_merges} | "
                      f"vocab={len(self.vocab)} | "
                      f"{best_pair!r} → {new_token!r} "
                      f"(freq={pair_freqs[best_pair]})")

        if verbose:
            print(f"  Vocabulario final: {len(self.vocab)} tokens")

    # ── Codificación / Decodificación ────────────────────────

    def encode(self, text: str) -> list[int]:
        """Texto → lista de ids (sin tokens especiales)."""
        units  = self._pretokenize(text)
        ids: list[int] = []
        for unit in units:
            tokens = self._apply_merges_to_word(list(unit))
            for tok in tokens:
                ids.append(self.vocab.get(tok, self.vocab[self.UNK]))
        return ids

    def decode(self, ids: list[int], skip_special: bool = True) -> str:
        """Lista de ids → texto legible."""
        tokens = []
        for i in ids:
            tok = self.id_to_token.get(i, self.UNK)
            if skip_special and tok in self.SPECIAL:
                continue
            tokens.append(tok)

        text = "".join(tokens)

        # Ġ marca inicio de palabra → espacio
        text = text.replace("Ġ", " ")

        # § marca signo de puntuación → quitar el prefijo, dejar el signo
        text = text.replace("§", "")

        text = text.strip()

        # Eliminar espacios antes de puntuación de cierre
        text = re.sub(r" ([.,;:!?\-—\)\"])", r"\1", text)
        # Eliminar espacios después de puntuación de apertura (¡ ¿ ( ")
        text = re.sub(r"([¡¿\(\"])\s", r"\1", text)

        return text

    def encode_with_special(self, text: str) -> list[int]:
        """Añade <bos> al inicio y <eos> al final."""
        return ([self.vocab[self.BOS]]
                + self.encode(text)
                + [self.vocab[self.EOS]])

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    @property
    def eos_id(self) -> int:
        return self.vocab[self.EOS]

    @property
    def bos_id(self) -> int:
        return self.vocab[self.BOS]

    @property
    def pad_id(self) -> int:
        return self.vocab[self.PAD]

    # ── Persistencia ─────────────────────────────────────────

    def save(self, path: str) -> None:
        data = {"vocab": self.vocab, "merges": self.merges}
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str) -> "BPETokenizer":
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return cls.load_from_dict(data)

    @classmethod
    def load_from_dict(cls, data: dict) -> "BPETokenizer":
        """Construye un tokenizador desde un dict ya parseado.
        Usado por los workers de multiprocessing para evitar I/O."""
        tok = cls()
        tok.vocab        = data["vocab"]
        tok.id_to_token  = {int(v): k for k, v in tok.vocab.items()}
        tok.merges       = [tuple(m) for m in data["merges"]]
        tok._merge_set   = {tuple(m): "".join(m) for m in tok.merges}
        return tok

    # ── Internos ─────────────────────────────────────────────

    # Regex: captura palabras (con acentos/ñ) O signos de puntuación
    # individualmente. Los signos llevan prefijo § para distinguirlos
    # de Ġ (inicio de palabra) al reconstruir.
    _WORD_RE = re.compile(
        r"[a-záéíóúüñA-ZÁÉÍÓÚÜÑ']+|[.,;:!?\u00a1\u00bf\-\u2014\(\)\"«»]",
        re.UNICODE
    )
    _PUNCT_RE = re.compile(
        r"[.,;:!?\u00a1\u00bf\-\u2014\(\)\"«»]"
    )

    def _pretokenize(self, text: str) -> list[str]:
        """
        Divide el texto en unidades léxicas:
          - Palabras      → "Ġhola", "Ġmundo"
          - Puntuación    → "§.", "§,", "§¡", "§?"  (token individual)
        """
        units: list[str] = []
        for m in self._WORD_RE.finditer(text):
            token = m.group()
            if self._PUNCT_RE.match(token):
                units.append("§" + token)   # signo de puntuación
            else:
                units.append("Ġ" + token.lower())   # palabra
        return units

    @staticmethod
    def _count_pairs(word_freqs: dict) -> Counter:
        pairs: Counter = Counter()
        for word, freq in word_freqs.items():
            for a, b in zip(word, word[1:]):
                pairs[(a, b)] += freq
        return pairs

    @staticmethod
    def _apply_merge(
        word_freqs: dict,
        pair: tuple[str, str],
        new_token: str
    ) -> dict:
        a, b = pair
        new_wf: dict[tuple, int] = {}
        for word, freq in word_freqs.items():
            new_word, i = [], 0
            while i < len(word):
                if i < len(word) - 1 and word[i] == a and word[i+1] == b:
                    new_word.append(new_token)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_wf[tuple(new_word)] = freq
        return new_wf

    def _apply_merges_to_word(self, tokens: list[str]) -> list[str]:
        for pair, merged in self._merge_set.items():
            a, b = pair
            new_tokens, i = [], 0
            while i < len(tokens):
                if i < len(tokens) - 1 and tokens[i] == a and tokens[i+1] == b:
                    new_tokens.append(merged)
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            tokens = new_tokens
        return tokens