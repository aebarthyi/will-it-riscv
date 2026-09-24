# Training plan: a small model that writes build plans

Goal: fine-tune a small open model to write the plan JSON a teacher writes,
from the same evidence pack, and measure it with `will-it-riscv-eval` against
the verified plans of the held-out repositories. Everything below runs on the
training server: 2× Radeon Pro VII (16 GB HBM2 each, gfx906), 64 GB RAM.

## Where things stand (2026-09-24)

- **Dataset:** 352 verified examples, labelled by Opus through `claude -p`
  (~$41 API-equivalent) and verified by running each plan in the pretend
  environment.
  - train: 316 (38 script-driven repositories, 278 PyPI packages)
  - test: 36 (7 repositories, 29 PyPI), held out by a hash of the name,
    with gdal, sundials, su2, hdf5 and llama.cpp pinned there
- **Files:**
  - `data/sft/{train,test}.jsonl`: chat JSONL with the system prompt, the
    pack and the plan, the plan in the wire format `modelplan.to_wire`
    writes
  - `data/full/<key>/`: every example's pack, plan, transcript and run
- **Lengths**, measured with Qwen3's tokenizer (median 5.2k; the longest is
  MFC at 30k):

  | max tokens | train kept | test within |
  | --- | --- | --- |
  | 6k | 203 / 316 | 17 / 36 |
  | 8k | 251 / 316 | 24 / 36 |
  | 12k | 300 / 316 | 32 / 36 |
  | 16k | 306 / 316 | 34 / 36 |

- **Baselines** on the 36 held-out entries (`will-it-riscv-eval --out data/full`):

  | planner | precision | recall | F1 | optional | order |
  | --- | --- | --- | --- | --- | --- |
  | autoplan (reads build files) | 0.75 | 0.38 | 0.45 | 0.92 | 0.33 |
  | teacher's first attempt | 0.99 | 0.99 | 0.99 | 1.00 | 1.00 |

  The student lands somewhere between these two.

## 0. Before the transfer (on the Mac)

- [ ] Commit the uncommitted code: the Meson/Bazel slice and the
  model-planning stack (`evidence.py`, `modelplan.py`, `dataset.py`,
  `evaluate.py`, `examples/corpus/`).
- [ ] Copy to the server:
  - the repository, **including `data/`** (git-ignored, ~40 MB; `data/pilot*`
    is superseded by `data/full` and can stay behind);
  - `~/Library/Caches/will-it-riscv/dataset/` (19 GB: the source trees
    every example was labelled from) to `~/.cache/will-it-riscv/dataset/`
    on the server, or set `WILL_IT_RISCV_CACHE`. The rest of the cache
    (~3 GB: index pages, sdists, pip wheels) downloads again on its own.
  - Claude Code's project directory,
    `~/.claude/projects/-Users-abarthyi-will-it-riscv/`, which holds this
    session and its memory. It is keyed by the repository's path, so put it
    under the key for the new path (for `/home/abarthyi/will-it-riscv`, that
    is `-home-abarthyi-will-it-riscv`).

## 1. Check the server before choosing anything

gfx906 is the MI50/MI60 chip, and recent ROCm releases have been retiring
it. Every choice below depends on these answers, so get them first.

- [ ] `rocminfo | grep -m2 gfx` shows gfx906 twice; note the ROCm version
  (`cat /opt/rocm/.info/version`).
- [ ] A PyTorch ROCm build whose `torch.cuda.get_arch_list()` includes
  `gfx906`, and a matmul that runs on both devices.
- [ ] **fp16 or bf16:** Vega 20 is not expected to do bf16 in hardware, so
  time an fp16 and a bf16 matmul, and use fp16 unless bf16 is as fast.
- [ ] **Attention memory, the deciding test.** flash-attention and
  PyTorch's memory-efficient SDPA are not expected on gfx906, so attention
  probably materialises the full score matrix (it did on the Mac: 32 heads ×
  12k² in fp16 is 9.5 GB). `training/memory_probe.py` (to write) runs one
  LoRA forward and backward of each candidate model at 4k, 6k, 8k, 12k and
  16k with gradient checkpointing, and records peak memory.
- [ ] `rocm-smi --showtopo`: is there an Infinity Fabric link between the
  cards? RCCL all-reduce works either way; the link only makes it faster.
- [ ] The tool itself on Linux: install the system tools (git, cmake,
  meson, ninja, pkg-config, gcc, g++, gfortran), run
  `pip install -e '.[dev]'`, then `pytest`. Everything so far ran on macOS,
  so fix whatever Linux turns up. Evaluation with `--execute`, and any later
  reinforcement learning, run the executor here.

## 2. Code to write on the server, in this order

- [ ] **Portable dataset paths.** `meta.json` records `root` as an absolute
  Mac path. Resolve it relative to the cache (the part after
  `will-it-riscv/`), and failing that, fetch the tree again: git at the
  commit in `origin`, PyPI at the recorded version.
- [ ] **The CMake loop must not write into the source tree.** It configures
  in place today; opencv's configure left `.cache/` in its tree and
  docling-parse left `externals/`. Configure a copy-on-write clone the way
  `pseudomeson.py` does. Stored examples are unaffected, because each pack
  was built before its plan ran.
- [ ] `training/prepare.py`: `data/sft` to TRL's conversational
  prompt-completion format (`prompt` = system + pack, `completion` = the
  plan), so the loss falls on the plan only. Drop, don't truncate, anything
  over the chosen length (a truncated example teaches a plan with its end
  cut off); carve a validation set of 1 in 16 from train; keep test exactly
  as exported. Count tokens by rendering the chat template and then
  tokenizing: `apply_chat_template(tokenize=True)` returns a dict in current
  transformers, and the MLX attempt measured every example as 2 tokens that
  way.
- [ ] `training/memory_probe.py`: the test in step 1.
- [ ] `training/train.py` with a YAML config per run: transformers, peft
  and trl, with accelerate / torchrun for DDP across both cards.
  - LoRA r=16, alpha=32, dropout 0, on every linear projection
  - fp16 (or bf16, if step 1 says so) with gradient checkpointing
  - batch 1, gradient accumulation 8, lr 1e-4, cosine schedule, 3% warmup,
    3 epochs, checkpoint and evaluation every ~100 steps
  - no bitsandbytes or flash-attention unless step 1 shows they work on
    gfx906
- [ ] If the probe caps us below 8k: a chunked attention registered through
  transformers' attention interface. Queries go in blocks of ~1k, each
  block checkpointed, so memory grows with block × length instead of
  length². That recovers 12–16k at the cost of some speed.
- [ ] `training/export.py`: merge the LoRA into the base model
  (`merge_and_unload`), save it, convert to GGUF with llama.cpp's
  `convert_hf_to_gguf.py`, and quantise to Q8_0.
- [ ] `evaluate.py`: break the table down by kind as well (script-driven
  repositories vs PyPI). The 7 repositories are where a model beats
  autoplan, and 29 PyPI rows will swamp them in an average.

## 3. Train

Choose the model from the probe's numbers:

| if the probe shows | then |
| --- | --- |
| Qwen3-4B fits at ≥12k on one card | **Qwen3-4B-Instruct-2507**, fp16 LoRA, one full copy per card (DDP) at 12k: 300 of 316 training examples |
| 4B only fits at ~8k | 4B at 8k (251 of 316), **and** Qwen3-1.7B at 12k (16 heads, half the attention memory): train both, keep the better |
| 4B doesn't fit at 6k | chunked attention first, then retry 4B |
| a later 8B run | Qwen3-8B's fp16 weights are ~16 GB, a whole card, so FSDP across both cards at short lengths; only worth it if 4B falls short |

Why the Instruct-2507 variant: it doesn't think, so its chat template adds
no `<think>` blocks to train around, and it handles long context natively.

Rough cost: 281 training examples at ~5.8k tokens is ~1.6M tokens an epoch.
At about a third of gfx906's ~26 fp16 TFLOPS, that's 45–60 min an epoch for
4B across both cards, so about 3 h for 3 epochs; 1.7B takes about 40% of
that. Watch validation loss; stop early once it rises.

## 4. Serve and evaluate

- [ ] Build llama.cpp with HIP for gfx906
  (`cmake -B build -DGGML_HIP=ON -DAMDGPU_TARGETS=gfx906`); its HIP backend
  is the dependable way to serve on these cards. vLLM does not target
  gfx906.
- [ ] `llama-server -m plans-4b-q8_0.gguf --ctx-size 32768 --port 8080`.
  Check that it constrains output with `response_format` json_schema:
  `modelplan.OpenAICompatible` sends `WIRE_SCHEMA` that way, and one curl
  with a small schema settles it.
- [ ] Score each of these with
  `will-it-riscv-eval --out data/full --planner openai --planner-url http://localhost:8080/v1 --planner-model <name>`:
  - the base Qwen3-4B-Instruct-2507, zero-shot, with the same prompt and
    schema: what fine-tuning adds;
  - the fine-tuned model with `--rounds 1`: what it writes unaided;
  - the fine-tuned model with `--rounds 3`: with the executor's feedback,
    which is how it would run in `--plan-from-model`;
  - the best of them again with `--execute`: does its plan reach the same
    verdict and require the same things when run.
- [ ] **What counts as working:** parsed 1.0 (the schema guarantees it);
  cited and grounded ≥ 0.95; F1 ≥ 0.8 overall, clearly above autoplan's
  0.45 on the 7 script-driven repositories. Read the misses one by one:
  whether the pack lacked the evidence or the model misread it decides
  what to fix next.

## 5. Then

- [ ] **More script-driven repositories.** Only 38 are in training, against
  278 PyPI packages, and they are the kind that matters. Another 100–150
  from HPC codes and projects with conda-forge or Spack recipes costs about
  $25–35 of teacher time at the observed $0.24 each, through the same
  `will-it-riscv-dataset label`, which resumes and stops cleanly at a
  usage limit.
- [ ] **Reinforcement learning** (GRPO) with the executor as the reward:
  cheap checks on every sample (schema, citations, grounding), execution on
  a subset, and results cached per (repository, step).
- [ ] **Known gaps the teacher itself flagged:**
  - no step kind for `git submodule update`;
  - the pack shows only the top-level `CMakeLists.txt` and `meson.build`;
  - Cargo and setuptools builds are read, not configured.
