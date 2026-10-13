# Training on 2x Gaudi 2

The next cycle after Lagutenko-4B (qwen3-4b-h200, step 318). Its fresh
holdout put it 0.10 behind Claude Sonnet on verdicts, and its test-split score
had been optimistic: the split that scored it also chose it, and reruns of the
same recipe varied by about 0.1. So this cycle measures the noise before
changing the recipe, and adds one change of size:

| run | model | seed | what it answers |
| --- | --- | --- | --- |
| qwen3-4b-gaudi-s0/s1/s2 | Qwen3-4B-Instruct-2507 | 0, 1, 2 | how far apart the same recipe lands |
| qwen3-8b-gaudi-s0 | Qwen3-8B (thinking off) | 0 | whether twice the size closes the gap |

All four use qwen3-4b-h200's recipe and data (v3: 1266 train, 33 valid;
LoRA r 64, lr 2e-4, 5 epochs, effective batch 8), one run per card, two at a
time. Qwen3-8B is Qwen3-4B's architecture and tokenizer at twice the size;
Qwen3.5 (4B, 9B) is newer but mostly linear attention plus a vision tower,
two changes at once and no tuned Gaudi kernels for it.

## What changed in train.py

- `device: hpu`: HCCL instead of NCCL, `mark_step` after each backward and
  optimizer step (lazy mode), no float64, no temperature pauses.
- `attention: fused`: hpu_attention.py runs Habana's FusedSDPA in recompute
  mode. Plain SDPA would hold heads x length^2 scores: 64 GB a layer at 32k.
- `loss: static` and `bucket: 512`: Gaudi compiles a graph for every shape.
  Each sequence is padded to one of 20 lengths (+10% tokens on v3), and the
  loss runs the head over every position in fixed chunks, so shapes follow the
  length alone. Checked on the CPU: the same loss and gradients as the sliced
  loss, padded or not (2e-8), and the same per-step losses in a 2-process run.
- `chat_template_kwargs`: Qwen3-8B renders with `enable_thinking: false`, so
  the prompt ends in an empty think block and the completion is the plan alone.

## Steps

Everything runs from this machine, with `GAUDI=user@host` (and `GAUDI_PORT`)
and `GAUDI_ROOT` set to the box's 512 GB NVMe. Everything on the box lives
under that root: the code, `venv-wir`, the HF cache `hf`, llama.cpp's converter,
and `runs`. Runs stay there. This machine takes only logs, plus the adapter
and GGUF of each checkpoint being scored.

1. `training/gaudi.sh push`, then `training/gaudi.sh setup`. Setup needs a
   Python with Intel Gaudi's PyTorch (Intel's PyTorch container, or set
   `SYS_PY`). It makes `venv-wir` beside it with transformers 5.17, peft 0.21
   and sentencepiece, torch pinned so pip cannot replace it. It also fetches
   llama.cpp's converter at the commit Lagutenko-4B's GGUF came from (b11177)
   and both base models, and prints the drive's free space and the RAM.
2. `training/gaudi.sh check`: HCCL between the cards, bf16 matmul TFLOPS, and
   FusedSDPA against plain causal attention, forward and backward. **If the
   attention check fails, nothing trains on `fused`**: hpu_attention passes
   FusedSDPA's arguments by position, and that order is from optimum-habana,
   not tested here. Fall back to `attention: sdpa` (if it fits) or `chunked`.
3. Smoke tests, both cards at once:
   `CARD=0 training/gaudi.sh start configs/gaudi-smoke-4b.yaml` and
   `CARD=1 … gaudi-smoke-8b.yaml`. Three steps on the longest sequences
   (~30k). Read from `status` / the .out: peak_gib (must leave headroom under
   96), tokens_per_second, and how long the first step took (graph compiles).
   If compiling dominates, try `PT_HPU_LAZY_MODE=0` (eager) on the same smoke.
4. The runs: cards 0 and 1 take s0 and s1; then s2 and the 8B. Compiled graphs
   are cached a card (`runs/recipes-card<N>` under the root), so a resume or the next seed
   on that card skips the compiles. `stop CONFIG` stops one cleanly.
5. `training/gaudi.sh pull CONFIG` brings a run's logs here.
   `training/gaudi.sh export RUN/checkpoint-N` runs export.py on the box,
   writing the Q8_0 GGUF beside the adapter (once). The 8B merges in fp32 on
   the CPU and needs about 32 GB of RAM. Then the adapter and the GGUF come to
   /scratch/runs/RUN/checkpoint-N, without optimizer state. Delete a scored
   GGUF here when it is done (the box keeps its copy): /scratch has about
   19 GB free.

gaudi.sh was run end to end on this machine, with a stand-in for ssh and a
scratch directory as the box: push, start (a CPU run through gaudi-run.sh),
status, pull, export (the converter at b11177, then both files back, then no
second conversion), and stop (only that config's processes, not its own shell).

## Choosing, then the holdout

- Score the epoch 2 and 3 checkpoints of each run (steps 318 and 477) on the
  **dev bench**, with run feedback: the 127-key test split plus holdout 1's 60
  (/scratch/runs/holdout). Lagutenko-4B on the same bench is the baseline.
- Serve the 8B with `llama-server --chat-template-kwargs '{"enable_thinking":false}'`,
  as it was trained.
- The best 4B and the best 8B, and nothing else, then run once on holdout 2
  (/scratch/runs/holdout2/PLAN.md, drawn and pre-registered 2026-10-13), with
  Lagutenko-4B and Claude Sonnet as anchors.
