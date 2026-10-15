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
$ will-it-riscv plan ~/src/nlopt --planner-url http://127.0.0.1:8080/v1 --model lagutenko-4B
```

The context is 40,960 tokens: the longest packs run to about 32k tokens, and
the plan and the feedback rounds need the rest. The tool samples at
temperature 0 and constrains the reply to the plan's JSON schema.

## Results

The test is 60 repositories that played no part in training, in choosing
this checkpoint, or in tuning the tool: 45 Spack CMake packages hosted on
GitHub and 15 PyPI sdists that ship platform wheels but none for riscv64,
drawn at random on 2026-10-11 from projects that no training corpus had
seen. The plan for the test was written down before any result, and each
row below ran once. Every repository was planned for Ubuntu 26.04 riscv64
(`ubuntu:resolute`) with will-it-riscv at commit 759ad6b. Claude Opus wrote
the reference plans, and each was verified by running it, as the training
data was.

- **verdict**: the share of repositories where the planner's answer (yes,
  probably, unknown or no) agreed with the reference's.
- **required**: the share of what the reference's configure proved the build
  needs that the planner's plan also required. It is scored on the 52
  repositories where the reference's run proved something.

| planner | loop | verdict | required | tokens / repo |
| --- | --- | --- | --- | --- |
| **this model** | one shot | 0.867 | 0.854 | 6k |
| **this model** | run feedback, 3 rounds | **0.867** | **0.861** | 9k |
| Qwen3-4B-Instruct-2507 (the base) | one shot | 0.050 | 0.093 | 6k |
| Qwen3-4B-Instruct-2507 (the base) | run feedback, 3 rounds | 0.367 | 0.308 | 18k |
| gpt-oss-20b | run feedback, 3 rounds | 0.817 | 0.792 | 14k |
| Claude Sonnet | run feedback, 3 rounds | 0.967 | 0.926 | 11k |
| Claude Sonnet | tool-using agent | 0.950 | 0.953 | 31k |

In the same loop, this model's verdicts trail Claude Sonnet's by 0.100
(paired 95% bootstrap: 0.033 to 0.183), which is significant; the gap on
required (0.065) is not. It is level with gpt-oss-20b, a model five times
its size. The fine-tune is the whole of the gain: the base model, in the
same loop, agrees on 0.367. Run feedback left this model's verdict score
where it was, and raised what its plans require a little.

Of the 8 repositories this model got wrong, Claude Sonnet got 6 right. In
two, the model copied a GPU backend from the README into the default build;
in two others its configure pointed at the wrong place in the tree. The
tool's feedback now names both mistakes, but that has not been measured:
this test has been used, and the next one needs a fresh draw.

An earlier version of this card reported 0.925 against Sonnet's 0.900 on
40 repositories from the held-out split. That split also chose this
checkpoint from about 14 candidates and steered the tool's development, so
those numbers were optimistic. This table replaces them.

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
  It was chosen over about 14 other runs and checkpoints on the held-out
  split, not on the test above.
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
