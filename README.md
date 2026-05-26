# ToyLM

Building a tiny language model from scratch. We take inspiration from the
[AlgoMonster course](https://algo.monster/courses/llm/llm_course_introduction) 
and [Andrej's Karpathy nanoGPT](https://github.com/karpathy/nanoGPT).

The objective of this project is to deeply understand the inner mechanisms of these
architectures, beggining with a basic GPT2-like model and implementing improvements
step by step according to the literature (e.g. RoPE, SwiGLU, etc.)

# History of improvements

| Date | Version | Description | Model size | Vocab size | Num layers | Num att. heads | Emb size | Head dim | Best val_loss |
|---|---|---|---|---|---|---|---|---|---|
| 25-05-2026 | 1.0 | Initial model | 873,216 param | vocab=2048 | layers=4 | heads=4 | embd=128 | head_dim=32 |-|
| 25-05-2026 | 2.0 | RoPE | 1,050,880 param | vocab=2048 | layers=4 | heads=4 | embd=128 | head_dim=32 |-|
| 25-05-2026 | 3.0 | RMSNorm | 1,049,856 param | vocab=2048 | layers=4 | heads=4 | embd=128 | head_dim=32 |-|
| 26-05-2026 | 3.1 | Vocab and embedding size reduction | 492,480 param | vocab=512 | layers=4 | heads=4 | embd=96 | head_dim=32 | 3.0685 (step 20000) |

# Ideas for future improvements

- (Done) Rotary Positional Embeddings (RoPE)
- (Done) RMSNorm instead of LayerNorm
- SwiGLU instead of GELU
- Grouped Query Attention (GQA)
- Scheduler: Warmup + cosine decay
- KV cache to speed up inference
