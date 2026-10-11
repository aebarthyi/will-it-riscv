---
license: apache-2.0
base_model: Qwen/Qwen3-4B-Instruct-2507
base_model_relation: finetune
pipeline_tag: text-generation
library_name: gguf
language:
- en
tags:
- gguf
- llama.cpp
- riscv
- riscv64
- build-systems
- cmake
- lora
---

# Lagutenko-4B (Q8_0)

This is the model behind [will-it-riscv](https://github.com/aebarthyi/will-it-riscv)'s
`plan` command. It reads an evidence pack that the tool builds from a
repository: install docs, build scripts, manifests, build files and CI. From
that it writes the repository's **build plan** as JSON: every step the build
runs, in order, with the options each configure takes and the system packages
it installs. The tool then runs every configure in the plan as a linux/riscv64
build confined to an empty sysroot. Whatever the run shows the plan got wrong
goes back to the model.

It is a fine-tune of [Qwen3-4B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507)
for this one task. It is not a general chat model, and it expects the tool's
system prompt and pack format.

It is named after Vitaly Lagutenko, the Soviet architect of plain, utilitarian
housing. The model aims at the same thing for builds: the smallest set of
dependencies that a build actually needs.

## Use

You don't need to download it by hand:

```console
$ pip install git+https://github.com/aebarthyi/will-it-riscv
$ will-it-riscv plan ~/src/nlopt
```

The first run fetches this file (4.3 GB) and llama.cpp's prebuilt
`llama-server` into `~/.local/share/will-it-riscv`, checks the SHA-256 of
each, and serves the model on this machine. With a Vulkan GPU (AMD, NVIDIA or
Intel), a plan takes well under a minute. On the CPU alone it takes about two
minutes and 11 GB of memory.

To serve it yourself, run any recent llama.cpp:

```console
$ llama-server -m lagutenko-4B-Q8_0.gguf --jinja -c 40960 -fa on -ngl 99
$ will-it-riscv plan ~/src/nlopt --planner-url http://127.0.0.1:8080/v1
```

The context is 40,960 tokens: the longest packs run to about 32k tokens, and
the plan and the feedback rounds need the rest. The tool samples at
temperature 0 and constrains the reply to the plan's JSON schema.

## Results

The test set is 40 repositories (30 git, 10 PyPI sdists), drawn at random from
the held-out split. None of them, and no other repository from that split,
appear in the training or validation data. Each was planned for Ubuntu 26.04
riscv64 (`ubuntu:resolute`).

- **verdict**: the share of repositories where the planner's answer (yes,
  probably, unknown or no) agreed with the reference's.
- **required**: the share of what the reference's configure proved the build
  needs that the planner's plan also required. It is scored on the 31
  repositories where the reference's run proved something.

| planner | loop | verdict | required | tokens / repo |
| --- | --- | --- | --- | --- |
| **this model** | one shot | 0.825 | 0.845 | 7k |
| **this model** | run feedback, 3 rounds | **0.925** | **0.942** | 10k |
| Qwen3-4B-Instruct-2507 (the base) | one shot | 0.100 | 0.006 | 7k |
| Qwen3-4B-Instruct-2507 (the base) | run feedback, 3 rounds | 0.400 | 0.397 | 21k |
| gpt-oss-20b | run feedback, 3 rounds | 0.850 | 0.831 | 17k |
| Claude Sonnet | run feedback, 3 rounds | 0.900 | 0.937 | 14k |
| Claude Sonnet | tool-using agent | 0.950 | 0.979 | 31k |

In the same loop, this model and Claude Sonnet differ by no significant
amount on verdict, required or F1 (paired 95% bootstrap). The feedback loop adds 0.100 to
this model's verdict score over one shot, which is significant.

**These numbers are optimistic.** The same held-out split was used to choose
this checkpoint from about 14 candidates, and to tune the tool around it.
Training runs vary a lot: a rerun of the same recipe, on the same data except
for 21 revised labels, scored 0.83 on verdict for the split's 97 git
repositories, against this checkpoint's 0.94. A fresh holdout of
repositories that were never used for any choice is planned, and this table
will be replaced by its results.

**Caveat:** Claude wrote the reference plans, and it also labelled the
training data. Agreement with the references therefore partly measures how
closely the model imitates its teacher. The verdicts are not just imitation:
they come from running each plan's configures. But the references have not
been checked against real riscv64 builds.

## Training

- **Base:** Qwen/Qwen3-4B-Instruct-2507 (Apache-2.0).
- **Method:** LoRA, r 64, alpha 128, dropout 0, on every linear projection
  (q, k, v, o, gate, up, down). The loss falls on the plan only, not on the
  prompt.
- **Schedule:** lr 2e-4, cosine, 3% warmup, effective batch 8 (1 × 8
  accumulation), 5 epochs (795 steps), bf16, seed 0, sequences up to 32,768
  tokens with none dropped. It ran on one NVIDIA H100 80GB in about 1.5 hours
  on 2026-09-28.
- **Checkpoint:** step 318, the end of epoch 2. This is the lowest validation
  loss of the run (0.522). Validation loss rose after it, to 0.670 at the end.
- **Export:** the adapter was merged into the base in fp32, saved in fp16, and
  converted to GGUF Q8_0 by llama.cpp's `convert_hf_to_gguf.py`. The Q8_0 file
  is the one evaluated above.

### Data

1,266 training examples (5.5M tokens; median 2.8k, longest 29.9k) and 33
validation examples:

- **214 repositories and 260 PyPI packages.** Each example pairs the tool's
  evidence pack with a build plan. Claude Opus wrote the plans through Claude
  Code. Each plan was run in the tool's confined riscv64 configure, and what
  the run showed went back to Claude until the plan held up. Only plans that
  held up were kept. Where a configure settled whether a package was required
  or optional, the run's answer replaced Claude's.
- **792 quiz questions** on how real builds find their dependencies. No
  model wrote these answers. Each one is either what CMake did when the
  project was configured (whether a dependency is required, optional, or
  never asked for, plainly and with each option turned on) or which Debian
  packages make a `find_package` succeed, read from Debian's Contents index.

Train, validation and test are split by a hash of the repository's name.
Quiz questions inherit the split of the repository they come from.

## Limitations

- It plans builds. It does not compile anything, and a "yes" means the
  configure accepts a linux/riscv64 build given the listed packages. It does
  not guarantee that the compile and link succeed.
- Most of the training data is CMake, Meson, autotools and Python packaging,
  on Debian and Ubuntu. Other build systems and distros are less tested.
- The tool runs the project's configure steps. Only point it at a repository
  you trust.
- Tested on Linux x86_64, CPU-only and Vulkan. macOS and GPUs with less than
  5 GB of VRAM are untested.

## License

The weights are released under Apache-2.0, the license of the base model.
The will-it-riscv tool is MIT.
