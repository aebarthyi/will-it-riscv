# will-it-riscv

Find out what a project needs in order to build on riscv64, before you have a
board. Point it at a repository and it works out what the build runs, tries
the configure steps as a linux/riscv64 build, and checks everything the build
needs against the distro's riscv64 archive. The answer is what to `apt
install`, what has to be built from source first, and what has nothing for
riscv64 at all.

Nothing is compiled and nothing is emulated.

## Install

```console
$ pip install git+https://github.com/aebarthyi/will-it-riscv
```

It needs Linux, Python 3.9 or later, and a C and C++ compiler
(`sudo apt install build-essential`). Add the `teacher` extra to use the
Claude API as the planner:

```console
$ pip install 'will-it-riscv[teacher] @ git+https://github.com/aebarthyi/will-it-riscv'
```

## Quick start

```console
$ will-it-riscv doctor              # what this machine has for it, and lacks
$ will-it-riscv plan ~/src/nlopt    # plan the build, run it, answer
```

The first `plan` downloads the planner, a fine-tuned 4B model
([`aebarthyi/lagutenko-4B`](https://huggingface.co/aebarthyi/lagutenko-4B),
4.3 GB), and llama.cpp's prebuilt server, once. The model runs on this
machine, on the GPU through Vulkan if there is one. On the CPU alone it needs
about 11 GB of memory.

**`plan` runs the project's configure steps.** Only point it at a repository
you trust, ideally in a container.

## How it works

1. **Read.** It gathers the parts of the repository that say how it builds:
   install docs, top-level scripts, manifests, build files and CI workflows.
2. **Plan.** A model writes the build as a plan: typed steps in order
   (system packages, Python installs, CMake or Meson configures, the
   project's own build driver), each citing the lines it came from. A
   citation that does not hold goes back to the model.
3. **Run.** Each step is answered without building anything for real:
   - Python packages are resolved against PyPI for riscv64 wheel tags, and
     system packages looked up in the distro's riscv64 archive.
   - Each configure runs as a cross build for linux/riscv64, confined to an
     empty sysroot. Whatever it stops on is stubbed and it runs again, so a
     dependency that stops it is required and one it does without is
     optional.
   - The project's own build driver runs with the compilers and build tools
     replaced by shims, to see which configures and imports it really makes.

   What the run shows the plan got wrong goes back to the model too.
4. **Answer.** Will it build on riscv64: yes, yes once some packages are
   built from source, probably, no, or unknown (with where it stopped).

## Usage

```console
# plan, run and answer for the repository here, against Ubuntu 26.04
$ will-it-riscv plan

# against Debian 13, keeping the plan, and drawing the dependency graph
$ will-it-riscv plan ~/src/MFC --distro debian:trixie --save-plan mfc.json --graph mfc.html

# also fetch, plan and configure what has only source for riscv64, all the way down
$ will-it-riscv plan ~/src/MFC --recurse

# write a script that does the build on the riscv64 machine (then: bash build-riscv.sh, there)
$ will-it-riscv plan ~/src/MFC --build-script build-riscv.sh

# with Claude writing the plan
$ will-it-riscv plan ~/src/MFC --planner claude

# machine-readable
$ will-it-riscv plan ~/src/MFC -f json -o report.json
```

Without a model:

```console
# read the source tree and CI configuration for what it links against
$ will-it-riscv ~/src/gromacs

# and run its configure as a linux/riscv64 build
$ will-it-riscv ~/src/gromacs --pseudobuild

# run a plan written by hand
$ will-it-riscv ~/src/MFC --plan examples/plans/mfc.json

# walk a Python dependency tree for what has no riscv64 wheel
$ will-it-riscv pyproject.toml -E all
$ will-it-riscv requirements.txt --target riscv64-musl1.2 --python 3.11
$ will-it-riscv -p 'numpy>=2' -p pandas
```

`will-it-riscv -h` and `will-it-riscv plan -h` list every option.

### Planners

| `--planner` | what writes the plan |
| --- | --- |
| `local` | the fine-tuned planner, run by llama.cpp on this machine (the default). `--model file.gguf` runs another GGUF |
| `claude` | Claude Code's `claude -p` if it is installed (`--model sonnet` or `opus`), else the Claude API (`ANTHROPIC_API_KEY`, `--model` a model id) |
| `openai` | any OpenAI-compatible server, such as vLLM, llama.cpp or Ollama: `--planner-url` and `--model` |

### Distros

`--distro` picks the archive to check against: `ubuntu:resolute` (Ubuntu
26.04, the default), `ubuntu:questing`, `ubuntu:plucky`, `ubuntu:noble`,
`debian:trixie`, `debian:forky` or `debian:sid`. Ubuntu's riscv64 packages
come from ports, `main` and `universe`. `--no-distro` skips the check.

### The build script

`--build-script FILE` writes a bash script to run on the riscv64 machine, from
the checkout. It installs what the archive has in one `apt-get`, makes a venv
for the Python steps, builds what has only source in dependency order, then
configures, builds and installs each step into `$PREFIX`. `DRY_RUN=1` prints
every command without running it, `SKIP_APT=1` skips the packages, and an
optional step runs with `WITH_<STEP>=1`. The script's header lists the rest.

### Exit codes

| code | `plan` | reading a project or manifest |
| --- | --- | --- |
| 0 | yes | everything installs from wheels or is pure Python |
| 1 | probably, or yes once something is built from source | something, or the project itself, is built from source |
| 2 | no, or unknown | something is blocked or has no distribution |
| 3 | the package index was unusable | the same |

### Files

The planner and llama.cpp's server live in `~/.local/share/will-it-riscv`
(`$WILL_IT_RISCV_DATA`). Index pages, sdists and distro package lists are
cached in the platform cache directory (`$WILL_IT_RISCV_CACHE`);
`will-it-riscv --clear-cache` empties it. `$LLAMA_SERVER` uses your own
llama.cpp server, and `$LLAMA_SERVER_ARGS` adds to its arguments
(`-ngl 0` for the CPU only).

## Limits

- **A yes is not a compile.** It means the configure accepts a linux/riscv64
  build given those packages. Compile checks are answered by this machine's
  compiler, and link checks cannot be answered at all.
- **It runs configures, not builds.** `plan`, `--plan` and `--pseudobuild`
  run the project's configure steps (and a plan's build driver, with its
  tools shimmed); `--recurse` runs the configures of every package it
  fetches.
- **Bazel, Cargo and setuptools builds are read, not configured.**
- **Fedora names are suggestions.** Only Debian and Ubuntu archives are
  checked. A library missing from the curated name map gets a guessed
  `libfoo-dev`, marked as a guess.

## Development

```console
$ pip install -e '.[dev]'
$ pytest -q
```

The planner's training data is made and scored with three more commands:
`will-it-riscv-dataset` labels repositories with verified plans,
`will-it-riscv-quiz` writes questions answered by real configures, and
`will-it-riscv-eval` scores planners against held-out plans. `training/`
holds the fine-tuning setup.

## License

MIT
