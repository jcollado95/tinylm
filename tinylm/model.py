"""
model.py — TinyLM con RoPE (Rotary Positional Embeddings)
==========================================================
Cambios respecto a la versión anterior:
  - [NUEVO]    RoPE sustituye a wpe (nn.Embedding de posición)
  - [ELIMINADO] wpe — ahorra block_size * n_embd parámetros
  - [REDUCIDO] n_embd: 128 → 96 (−43% parámetros totales)
  - [LIMPIADO] bias=False en todas las Linear (FF incluido)
 
¿Por qué RoPE mejora al embedding aprendido?
  El embedding posicional aprendido (wpe) asigna un vector fijo
  a cada posición 0..block_size-1. RoPE en cambio *rota* los
  vectores Q y K en el espacio complejo según su posición relativa,
  de forma que el producto escalar Q·Kᵀ captura automáticamente la
  distancia entre tokens. Ventajas:
    1. Sin parámetros extra (la rotación es determinista).
    2. Generaliza a secuencias más largas que las vistas en
       entrenamiento (extrapolación posicional).
    3. La información posicional está integrada en la atención,
       no sumada al embedding antes de los bloques.
"""

import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from dataclasses import dataclass


# ─────────────────────────────────────────────────────────────
# CONFIGURACIÓN
# ─────────────────────────────────────────────────────────────
 
@dataclass
class Config:
    vocab_size:  int   = 512
    n_embd:      int   = 96
    n_head:      int   = 4
    n_layer:     int   = 4
    block_size:  int   = 128
    dropout:     float = 0.1

# ─────────────────────────────────────────────────────────────
# ROPE — Rotary Positional Embeddings
# ─────────────────────────────────────────────────────────────
 
def build_rope_cache(
    seq_len:  int,
    head_dim: int,
    device:   torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Precalcula las tablas de cos/sin para RoPE.
 
    Matemática
    ----------
    Para cada posición t y cada par de dimensiones (2i, 2i+1):
 
        θᵢ = 10000^(−2i / head_dim)      ← frecuencia base
        cos(t·θᵢ),  sin(t·θᵢ)            ← rotación en el plano i
 
    La rotación se aplica a pares consecutivos de dimensiones:
        [x₀, x₁] → [x₀·cos − x₁·sin,  x₀·sin + x₁·cos]
 
    Retorna
    -------
    cos, sin : (seq_len, head_dim/2)  — se broadcastean en apply_rope
    """
    assert head_dim % 2 == 0, "head_dim debe ser par para RoPE"
 
    # θᵢ para i = 0, 1, ..., head_dim/2 − 1
    i     = torch.arange(0, head_dim, 2, device=device).float()
    theta = 1.0 / (10000.0 ** (i / head_dim))           # (head_dim/2,)
 
    # t · θᵢ  para cada posición t
    t     = torch.arange(seq_len, device=device).float() # (seq_len,)
    freqs = torch.outer(t, theta)                         # (seq_len, head_dim/2)
 
    return freqs.cos(), freqs.sin()                       # dos tensores (T, head_dim/2)
 
 
def apply_rope(
    x:   torch.Tensor,          # (B, n_head, T, head_dim)
    cos: torch.Tensor,          # (T, head_dim/2)
    sin: torch.Tensor,          # (T, head_dim/2)
) -> torch.Tensor:
    """
    Aplica la rotación RoPE a un tensor Q o K.
 
    La rotación opera sobre pares consecutivos de dimensiones:
        dimensiones pares   (0, 2, 4, ...) → x1
        dimensiones impares (1, 3, 5, ...) → x2
 
    Resultado:
        x1_rot = x1 * cos − x2 * sin
        x2_rot = x1 * sin + x2 * cos
    """
    # Separar pares e impares: ambos (B, n_head, T, head_dim/2)
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
 
    # Añadir dims de batch y cabeza para broadcast: (1, 1, T, head_dim/2)
    cos = cos[None, None, :, :]
    sin = sin[None, None, :, :]
 
    # Rotación en cada plano 2D
    x1_rot = x1 * cos - x2 * sin
    x2_rot = x1 * sin + x2 * cos
 
    # Intercalar de vuelta: (B, n_head, T, head_dim)
    # stack sobre dim=-1 → (B, n_head, T, head_dim/2, 2), flatten → head_dim
    return torch.stack([x1_rot, x2_rot], dim=-1).flatten(-2)

# ─────────────────────────────────────────────────────────────
# BLOQUES INTERNOS
# ─────────────────────────────────────────────────────────────

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps   = eps
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        rms = x.pow(2).mean(-1, keepdim=True).add(self.eps).sqrt()
        return x / rms * self.scale
 
class CausalSelfAttention(nn.Module):
    """
    Multi-head self-attention causal con RoPE.
 
    Diferencias respecto a la versión anterior:
      - Recibe (cos, sin) precalculados en lugar de posiciones absolutas.
      - apply_rope() se aplica a Q y K antes del producto escalar.
      - V NO se rota (solo Q y K necesitan información posicional).
    """
 
    def __init__(self, config: Config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
 
        self.n_head   = config.n_head
        self.n_embd   = config.n_embd
        self.head_dim = config.n_embd // config.n_head
 
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=False)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd,     bias=False)
        self.drop   = nn.Dropout(config.dropout)
 
        # Máscara causal — buffer estático, no es un parámetro
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(config.block_size, config.block_size))
            .view(1, 1, config.block_size, config.block_size)
        )
 
    def forward(
        self,
        x:   torch.Tensor,    # (B, T, C)
        cos: torch.Tensor,    # (T, head_dim/2)  — precalculado en TinyLM
        sin: torch.Tensor,    # (T, head_dim/2)
    ) -> torch.Tensor:
 
        B, T, C = x.shape
 
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
 
        # (B, T, C) → (B, n_head, T, head_dim)
        def split_heads(t: torch.Tensor) -> torch.Tensor:
            return t.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
 
        q, k, v = split_heads(q), split_heads(k), split_heads(v)
 
        # ── RoPE: rotar Q y K con las frecuencias posicionales ──
        # V no se rota: solo necesitamos posición en similitud Q·K,
        # los valores V transportan contenido, no posición.
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)
 
        # Scaled dot-product attention con máscara causal
        scale = self.head_dim ** -0.5
        att   = (q @ k.transpose(-2, -1)) * scale
        att   = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        att   = torch.softmax(att, dim=-1)
        att   = self.drop(att)
 
        out = att @ v
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(out)


class FeedForward(nn.Module):
    """MLP con activación GELU. bias=False en todas las Linear."""
 
    def __init__(self, config: Config):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd, bias=False),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd, bias=False),
            nn.Dropout(config.dropout),
        )
 
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformerBlock(nn.Module):
    """Pre-LayerNorm + conexiones residuales. Recibe cos/sin para RoPE."""
 
    def __init__(self, config: Config):
        super().__init__()
        self.ln1  = RMSNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln2  = RMSNorm(config.n_embd)
        self.ff   = FeedForward(config)
 
    def forward(
        self,
        x:   torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        x = x + self.attn(self.ln1(x), cos, sin)
        x = x + self.ff(self.ln2(x))
        return x


# ─────────────────────────────────────────────────────────────
# MODELO PRINCIPAL
# ─────────────────────────────────────────────────────────────
 
class TinyLM(nn.Module):
 
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
 
        # ── Embeddings ───────────────────────────────────────
        self.wte  = nn.Embedding(config.vocab_size, config.n_embd)
        # wpe ELIMINADO — RoPE no necesita embedding de posición aprendido
 
        self.drop   = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList(
            [TransformerBlock(config) for _ in range(config.n_layer)]
        )
        self.ln_f    = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
 
        # Weight tying: wte y lm_head comparten la misma matriz
        self.wte.weight = self.lm_head.weight
 
        self.apply(self._init_weights)
 
        n_params = sum(p.numel() for p in self.parameters())
        print(f"TinyLM+RoPE — {n_params:,} parámetros | "
              f"vocab={config.vocab_size} | "
              f"layers={config.n_layer} | "
              f"heads={config.n_head} | "
              f"embd={config.n_embd} | "
              f"head_dim={config.n_embd // config.n_head}")
 
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
 
    def forward(
        self,
        idx:     torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
 
        B, T = idx.shape
        assert T <= self.config.block_size, (
            f"Secuencia de longitud {T} supera block_size={self.config.block_size}"
        )
 
        # Token embeddings — SIN sumar wpe
        x = self.drop(self.wte(idx))                         # (B, T, n_embd)
 
        # Precalcular cos/sin UNA sola vez para todos los bloques.
        # Todos los bloques de la misma secuencia usan las mismas
        # frecuencias: la posición del token no cambia entre capas.
        head_dim = self.config.n_embd // self.config.n_head
        cos, sin = build_rope_cache(T, head_dim, idx.device) # (T, head_dim/2)
 
        # Pasar por los bloques Transformer (nn.ModuleList, no Sequential,
        # porque ahora forward necesita argumentos extra cos y sin)
        for block in self.blocks:
            x = block(x, cos, sin)
 
        x      = self.ln_f(x)
        logits = self.lm_head(x)                             # (B, T, vocab_size)
 
        loss = None
        if targets is not None:
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
        Generación autoregresiva con parada en <eos>.
 
        RoPE y el contexto deslizante funcionan igual que antes:
        en cada paso se recorta a block_size y se recalculan cos/sin
        para la longitud actual del contexto.
        """
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.config.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
 
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")
 
            probs    = torch.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx      = torch.cat((idx, idx_next), dim=1)
 
            if eos_id is not None and (idx_next == eos_id).all():
                break
 
        return idx