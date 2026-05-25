"""
model.py — TinyLM: Transformer minimalista autoregresivo
=========================================================
Arquitectura:
  - Token embeddings + Positional embeddings
  - N bloques Transformer (CausalSelfAttention + FeedForward)
  - Pre-LayerNorm + conexiones residuales
  - Weight tying entre wte y lm_head
  - generate() con temperature, top_k y parada en <eos>
"""

import torch
import torch.nn as nn
from torch.nn import functional as F
from dataclasses import dataclass


@dataclass
class Config:
    vocab_size:  int   = 512
    n_embd:      int   = 64
    n_head:      int   = 4
    n_layer:     int   = 3
    block_size:  int   = 128
    dropout:     float = 0.1

# ── Bloques internos ─────────────────────────────────────────

class CausalSelfAttention(nn.Module):
    """
    Multi-head self-attention con máscara causal.
    El token i solo puede atender a los tokens 0..i (autoregresivo).
    """

    def __init__(self, config: Config):
        super().__init__()
        assert config.n_embd % config.n_head == 0, \
            "n_embd debe ser divisible entre n_head"

        self.n_head  = config.n_head
        self.n_embd  = config.n_embd
        self.head_dim = config.n_embd // config.n_head

        # Q, K, V en una sola proyección (eficiencia)
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=False)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd,     bias=False)
        self.drop   = nn.Dropout(config.dropout)

        # Máscara causal estática (no es un parámetro, pero se guarda en el estado)
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(config.block_size, config.block_size))
            .view(1, 1, config.block_size, config.block_size)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape

        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)

        # (B, T, C) → (B, n_head, T, head_dim)
        def split_heads(t: torch.Tensor) -> torch.Tensor:
            return t.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        q, k, v = split_heads(q), split_heads(k), split_heads(v)

        # Scaled dot-product attention + máscara causal
        scale = self.head_dim ** -0.5
        att   = (q @ k.transpose(-2, -1)) * scale
        att   = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        att   = torch.softmax(att, dim=-1)
        att   = self.drop(att)

        out = att @ v                                        # (B, n_head, T, head_dim)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(out)


class FeedForward(nn.Module):
    """MLP con activación GELU (expansión ×4, estándar en GPT)."""

    def __init__(self, config: Config):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformerBlock(nn.Module):
    """
    Bloque Transformer con Pre-LayerNorm y conexiones residuales.

    Pre-LN (en lugar de Post-LN) estabiliza el entrenamiento
    en modelos pequeños al normalizar antes de cada sublayer.
    """

    def __init__(self, config: Config):
        super().__init__()
        self.ln1  = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln2  = nn.LayerNorm(config.n_embd)
        self.ff   = FeedForward(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))   # residual + atención
        x = x + self.ff(self.ln2(x))     # residual + FF
        return x


# ── Modelo principal ─────────────────────────────────────────

class TinyLM(nn.Module):

    def __init__(self, config: Config):
        super().__init__()
        self.config = config

        self.wte    = nn.Embedding(config.vocab_size, config.n_embd)   # token embeddings
        self.wpe    = nn.Embedding(config.block_size, config.n_embd)   # position embeddings
        self.drop   = nn.Dropout(config.dropout)
        self.blocks = nn.Sequential(
            *[TransformerBlock(config) for _ in range(config.n_layer)]
        )
        self.ln_f    = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Weight tying: el embedding de salida comparte pesos con wte.
        # Reduce parámetros y mejora convergencia (press & wolf, 2017).
        self.wte.weight = self.lm_head.weight

        self.apply(self._init_weights)

        n_params = sum(p.numel() for p in self.parameters())
        print(f"TinyLM — {n_params:,} parámetros | "
              f"vocab={config.vocab_size} | "
              f"layers={config.n_layer} | "
              f"heads={config.n_head} | "
              f"embd={config.n_embd}")

    # ── Inicialización de pesos ──────────────────────────────

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    # ── Forward ──────────────────────────────────────────────

    def forward(
        self,
        idx:     torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Parámetros
        ----------
        idx     : (B, T) tensor de ids de tokens
        targets : (B, T) tensor de ids objetivo para calcular la pérdida

        Retorna
        -------
        logits  : (B, T, vocab_size)
        loss    : escalar o None
        """
        B, T = idx.shape
        assert T <= self.config.block_size, (
            f"Secuencia de longitud {T} supera block_size={self.config.block_size}"
        )

        positions = torch.arange(T, device=idx.device)
        x = self.drop(self.wte(idx) + self.wpe(positions))
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            # ignore_index=-1 permite enmascarar posiciones de padding
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1,
            )
        return logits, loss

    # ── Generación ───────────────────────────────────────────

    @torch.no_grad()
    def generate(
        self,
        idx:            torch.Tensor,
        max_new_tokens: int,
        eos_id:         int | None = None,
        temperature:    float = 0.8,
        top_k:          int   = 20,
    ) -> torch.Tensor:
        """
        Genera tokens autoregresivamente.

        Parámetros
        ----------
        idx            : (B, T) prompt inicial
        max_new_tokens : límite de tokens a generar
        eos_id         : si se proporciona, detiene la generación al encontrarlo
        temperature    : >1 → más aleatorio | <1 → más determinista
        top_k          : muestreo restringido a los k tokens más probables

        Retorna
        -------
        idx : (B, T + tokens_generados)
        """
        self.eval()
        for _ in range(max_new_tokens):
            # Recortar contexto si supera block_size
            idx_cond = idx[:, -self.config.block_size:]

            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature         # (B, vocab_size)

            # Top-k: poner -inf a todo lo que no esté en el top-k
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")

            probs    = torch.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            idx      = torch.cat((idx, idx_next), dim=1)

            # Parar si todos los elementos del batch han emitido <eos>
            if eos_id is not None and (idx_next == eos_id).all():
                break

        return idx