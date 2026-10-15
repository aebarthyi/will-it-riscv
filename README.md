# will-it-riscv

Find out what a project needs in order to build on riscv64 — before you get
there. It reads a repository's source tree and its CI configuration to work out
what the project itself links against, walks its Python dependency tree to find
what has no riscv64 wheel, and checks every resulting system package against the
distro's actual riscv64 archive.

```console
$ git clone https://github.com/gromacs/gromacs && cd gromacs
$ will-it-riscv
```

Point it at a repository and it reads the source tree; point it at a
`pyproject.toml` or `requirements.txt` and it walks the dependency graph. Most
repositories want both, and it does both.

It answers three questions for every package in the transitive closure:

1. **Is there a wheel for the target?** Matched properly against PEP 425/600/656
   tags, so `manylinux_2_39_riscv64` counts and `manylinux_2_28_x86_64` does not.
2. **If not, can pip build it?** The source distribution is downloaded and read —
   never executed — to find out whether it compiles anything, with what, and
   whether its own build backend is available for the target.
3. **What does that build need from the system?** Libraries and toolchain scraped
   out of `setup.py`, `CMakeLists.txt`, `meson.build`, `configure.ac`, `Cargo.toml`
   and PEP 725 `[external]`, mapped to Debian/Fedora package names, and checked
   against the distro's actual riscv64 archive.

The output is the list you need to hand to whoever is provisioning the board: what
to `apt install`, what has no distro package and must be built by hand, and what
the distro already ships so you can skip the build entirely.

## Quick start

```console
$ pip install git+https://github.com/aebarthyi/will-it-riscv
$ will-it-riscv doctor              # what this machine has for it, and lacks
$ will-it-riscv plan ~/src/nlopt    # write the build plan, run it, answer
$ will-it-riscv plan ~/src/nlopt --graph deps.html --build-script build-riscv.sh
```

The second form also draws the dependency graph as a page to explore, and
writes a script that builds the project on the riscv64 machine itself:
`bash build-riscv.sh` from the checkout there.

`plan` has a fine-tuned 4B model write the repository's build plan:
everything its build runs, in order. It runs every configure in that plan as
a linux/riscv64 build confined to an empty sysroot, and sends what the run
shows the plan got wrong back to the model. Then it answers: will it build on
riscv64, and what does it need from Ubuntu 26.04's archive?

On 60 fresh repositories that played no part in training or in choosing
the model, the planner's verdicts agreed with the references' on 52 (Claude
Sonnet's, in the same loop: 58). It named 86% of what those builds require
(Sonnet: 93%). The [model card](https://huggingface.co/aebarthyi/lagutenko-4B)
has the full table.

The first run fetches the planner (4.3 GB) and llama.cpp's prebuilt server,
once, into `~/.local/share/will-it-riscv`. The model runs on this machine:
with a GPU (Vulkan, so AMD, NVIDIA or Intel) a plan takes well under a
minute; on the CPU alone, about two minutes and 11 GB of memory.

It needs Linux, and a C and C++ compiler (`sudo apt install build-essential`).
`--planner claude` uses Claude instead, through Claude Code or the API. **It
runs the project's configure steps**: only point it at a repository you trust.

## Why

Wheel availability for riscv64 is real but thin. [manylinux gained
`manylinux_2_39_riscv64` images in 2025][manylinux] and packages like `aiohttp`,
`regex`, `markupsafe`, `charset-normalizer` and `rpds-py` publish riscv64 wheels
today — but most of PyPI does not, and `pip install` on the target is a slow,
serialized way to discover that. [uv can resolve for riscv64][uv-riscv]
(`--python-platform riscv64-unknown-linux`), but a resolution names versions;
which of them must be compiled, and what each compile needs from the system,
is still left to find out.

The alternative is finding out one traceback at a time on an emulated board.

[manylinux]: https://github.com/pypa/manylinux
[uv-riscv]: https://github.com/astral-sh/uv/pull/15630

## Two modes

**Repository mode** runs automatically when you point it at a directory. It
reads the source tree to find what *this* project needs in order to compile —
scraping `CMakeLists.txt`, `meson.build`, `configure.ac`, `Makefile`,
hand-written `configure` scripts, `Cargo.toml`, `setup.py` and the `#include`
directives in the C and C++ sources — and then
reads the CI configuration, where projects usually write their system
dependencies down outright. Those declared lists are authoritative in a way
inference is not, so they are never reported as guesses.

It understands the shapes real repositories use: Pillow's bash array
(`packages=( … )` installed as `"${packages[@]}"`), psycopg2's interpolated
version pin, GROMACS's [HPC Container Maker][hpccm] `ospackages=[…]`, plain
Dockerfiles, `apt.txt`, and nix `buildInputs`. What it cannot resolve it says
so about, rather than silently reporting nothing. It also flags third-party
package sources — a PPA or a vendor apt repo that has no builds for your
architecture is a finding, not a detail.

### Pseudobuilds: configure it, watch, throw it away

Static reading has a ceiling. GDAL wraps every driver in its own
`gdal_check_package()` macro, so no regex and no `if`/`else` walk will ever
tell you those are optional — but the macro is perfectly visible while it
runs.

`--pseudobuild` runs the project's **configure** step (never its build) in a
scratch directory, with CMake tracing every command and its arguments already
expanded, and every `pkg-config` query denied. A configure told that nothing
is installed, which still insists on something, genuinely needs it.

```console
$ will-it-riscv ~/src/gromacs --pseudobuild
```
```
Pseudobuild
  configured as linux/riscv64, confined to an empty sysroot; configure completed over 2 rounds
  in 31s — 22 dependency probes observed
  rounds: FFTW → completed
  blame by experiment: FFTW ✓   (✓ the error moved once it existed; ✗ it did not, so the stub
  was taken back out)

  Will it riscv?  YES — every dependency a default build demands is in Ubuntu 26.04 LTS
                  (resolute) for riscv64
    sudo apt install libfftw3-dev    # the minimum a default configure demanded
    shown by configuring, not compiling: the configure accepts a linux/riscv64 build given only these

  hard requirements, in the order the build demanded them:
    1.  FFTW  shown by experiment  cmake/gmxManageFFTLibraries.cmake:67  libfftw3-dev ✓
  proven optional — absent, and the configure carried on (4): HDF5, ImageMagick, LATEX, MPI
  claimed by compile-only checks, unverifiable without a target linker: BLAS, LAPACK, OpenMP
  graph: 8 nodes, 8 edges — -f dot | dot -Tsvg > deps.svg
```

**It configures for riscv64, not for the machine you run it on.** The
configure runs as a cross build — `CMAKE_SYSTEM_NAME Linux`,
`CMAKE_SYSTEM_PROCESSOR riscv64` — with every library, header and package
search re-rooted into an empty scratch sysroot. Otherwise the host answers
questions meant for the target. On a Mac with Homebrew, 24 of GDAL's
dependencies were simply *found*, which proves nothing about whether the
build needed them, and `if(APPLE)` sends the configure down branches a
riscv64 build never takes. Confined, the only things a configure can find are
the stubs this tool puts there, so every probe ends in a provable state.
OpenCV, confined, follows its RISC-V branches and checks for RVV.

| the configure said | meaning |
| --- | --- |
| `-- Could NOT find X`, carried on, and **completed** | **X is optional.** Demonstrated, not inferred. |
| `-- Could NOT find X`, carried on, then stopped | nothing — it may have stopped *because* of X. AdaptiveCpp misses LLVM this way. |
| `Could NOT find X` inside a `CMake Error` | **X is required**, and it is the first thing that stops the build. |
| `-- Found X` at a host program | a build tool on this host; whether the build needs it is untested |
| `-- Found X`, a library, confined | a compile-only check fooled — nothing real was there to find |

And when the configure runs to the end, silence counts too: anything it
never asked about is not part of a default build. That inference is applied only to
dependencies sighted solely in CMake files, since the trace has no view of a
Makefile or a CI config.

**Unblock and rerun.** A configure that stops tells you one thing: what
stopped it. So the blocker is satisfied with a stub and the configure runs
again, until it either completes or stops somewhere stubbing cannot reach.
Everything is driven by what the configure itself said:

| it said | the stub |
| --- | --- |
| `(missing: PROJ_LIBRARY PROJ_INCLUDE_DIR)` | an empty library and an include directory |
| `Could NOT find HDF5`, however it is worded | whatever HDF5's own Find module looked for this round — `hdf5.h`, `libhdf5.so` — put where it looked |
| `Could not find FYPP_EXE using the following names: fypp` | a runnable `fypp` — `find_program(… REQUIRED)`, and the same for libraries and paths |
| `(missing: … SSL Crypto)`, or `links to: OpenSSL::SSL but the target was not found` | the component's `OPENSSL_SSL_LIBRARY` |
| `The following required packages were not found: - libpsl` | a `libpsl.pc` in the only directory pkg-config may search |
| `Could not find a package configuration file provided by "Boost"` | a `BoostConfig.cmake` in the sysroot, with `Boost::headers`, `Boost::boost` and a target for each component asked for; it grows any other imported target the build links once the configure names it missing |
| `file failed to open for reading: …/proj.h` | that header, carrying every common spelling of a version macro |
| `… are set to NOTFOUND … FOO_LIBRARY linked by target` | the variable a target links |
| `linux/fs.h header not found` | the header — reported as a **host gap**, not a dependency: every riscv64 Linux system has it; the Mac SDK does not |
| `(missing: DOXYGEN_EXECUTABLE)`, then `doxygen -s -g CMakeDoxyfile.tpl` | a program that answers `--version` (`99.9.9`; `git version 99.9.9` for git) and writes the file it is told to write, inside the scratch directory only |
| `Found unsuitable version "0.0.0"`, after the module grepped a stub header with `REGEX "#define ARMA_VERSION_MAJOR …"` | a line that the regex matches, added to the stub: `#define ARMA_VERSION_MAJOR 99` (`patchlevel.h` gets the interpreter's own version). If the module read the header only `if(EXISTS)`, it is written first |
| `links to: BML::bml but the target was not found`, with no config to grow it | an empty imported target, made after the top directory is done, so a real one always comes first |
| `… provided by "QT" with any of the following names: Qt6Config.cmake …` | `Qt6Config.cmake`, the first name searched. A stub whose name ends in its major version (Qt6, Eigen3, SDL2) claims that major, `6.99.9`, and each component gets its own `Qt6Core_VERSION` |

**Blame by experiment.** Some configures die without naming anything.
GROMACS's FindFFTW never says "Could NOT find" at all; the configure fails
in GROMACS's own words, `Cannot find FFTW 3`. So the loop suspects the
misses nearest the error — first any the error message mentions by name, then
the find_packages the trace saw it name — and stubs the likeliest the way its
own Find module looked for it: the trace records every `find_library`,
`find_path` and `pkg_check_modules` each module ran. A lookup the error names
in the project's own words is a suspect too, after the named packages:
`find_program(EXE_SQLITE3 sqlite3)`, then `sqlite3 binary not found!`. CMake's
own lookups (`CMAKE_MAKE_PROGRAM`, which every quoted try_compile log names)
and the pinned interpreter are never suspects. If the error moves, the
suspect is a hard requirement, shown by experiment. If it does not, every
trace of the stub is taken back out, so a later round cannot find it, and the
next suspect is tried — at most three per stuck point.

Status misses are never stubbed on sight — faking an optional dependency
would erase the very evidence that it is one — and nothing a configure sees
is ever written outside the scratch directory.

**An agent for the long tail (opt-in, experimental).** When the rules run out
of suspects, `WILL_IT_RISCV_UNBLOCK=claude` (Sonnet, or `claude:<model>`)
hands the stuck point to Claude through `claude -p`, with a workbench served
over MCP.
It reads: the error, the trace around the line that raised it, what each
Find module searched for, the source tree and CMake's modules. It writes stub
files inside the sysroot, and sets cache variables a lookup would have
filled: a find result, a check's `HAVE_*`, a version, a package's `_DIR`, or
an install location the error asks for by name — never a path on the host.
It is never allowed the source tree, a build option, the plan's `-D` flags
or `CMAKE_*`. It can rerun the configure up to eight times. Its stubs come
back as a trial like any other: if the executor's next round shows the error
moved, they stay, and if not, they are taken back out. An agent that finds
the plan at fault (the configure insists on a flag the plan does not give)
or something no stub can be (a CUDA toolkit, which is then a hard
requirement) leaves a note instead.

A configure gets at most six sessions, each capped at $2
(`$WILL_IT_RISCV_UNBLOCK_BUDGET`) and 20 minutes, and their time is not
counted against `--pseudobuild-timeout`. Each session's prompt, transcript
and stubs are kept under `$WILL_IT_RISCV_UNBLOCK_LOG` (default
`~/.cache/will-it-riscv/unblock`), and so is what it found: a later run that
meets the same stuck point gets the same stubs back without a new session,
and a trial that failed is forgotten. `WILL_IT_RISCV_UNBLOCK=replay` uses
only what earlier sessions found, and never starts one.

### The dependency graph, and the answer

Every probe in the trace carries who asked: the project line that decided to
(for GDAL, each `gdal_check_package(…)` call, not the macro body they all go
through), the project macro it went through, the package whose Find module
asked it, and the round that first reached it. That is the graph — the
project at the root, an edge from whatever asked to what it asked for — and
`-f dot` draws it, grouped by round, so the drawing shows the unblock loop
too: what the first configure asked, and what only came into view once PROJ
existed.

```console
$ will-it-riscv ~/src/gdal --pseudobuild -f dot | dot -Tsvg > gdal.svg
```

Hover a node for its proof. `-f json` carries the same graph under
`pseudobuild.graph`.

The answer comes from checking the hard requirements — and the build tools
the configure REQUIRED — against the distro's riscv64 archive:

| | |
| --- | --- |
| **yes** | the configure completes as a linux/riscv64 build, and the archive has everything it demanded |
| **probably** | it completes, but a name could not be matched to a package, the archive was not checked, or it only configured for the host |
| **no** | a hard requirement has no riscv64 package: it has to be built first |
| **unknown** | the configure stopped somewhere no stub gets past, or ran out of time, and says where |

| repo | static | pseudobuild | hard requirements | will it riscv? |
| --- | --- | --- | --- | --- |
| gdal | 315 required / 4 optional | **221 / 98** | PROJ, and SWIG to build with | yes — `libproj-dev swig` |
| curl | 48 / 1 | **30 / 19** | OpenSSL, libpsl | yes — `libssl-dev libpsl-dev` |
| opencv | 60 / 13 | **22 / 51** | none — it bundles every codec | yes |
| gromacs | 20 / 0 | **13 / 7** | FFTW (by experiment) | yes — `libfftw3-dev` |
| adaptivecpp | 13 / 6 | **12 / 8** | LLVM, llc (by experiment) | yes — `llvm-dev` |

"Yes" means the configure accepts a riscv64 build given only those packages.
It does not mean the code compiles: the pseudobuild never runs the compiler
on the project.

**This runs the project's build scripts.** Everything else in this tool only
reads. Use it on repositories you trust, ideally in a container. It is
opt-in, time-bounded (`--pseudobuild-timeout`, 20 minutes, bounds the whole
loop, agent sessions aside), confined to a temporary directory that is
deleted afterwards, and never invokes the compiler on the project itself.

**Its compile checks are the host's.** There is no riscv64 compiler in the
loop: the host's compiler answers `check_include_file` and friends, against
the host's SDK. Checks that would need linking cannot be answered at all —
the Linux platform rules and the host's linker do not mix — so try-compiles
only compile, and a library a link check "found" is reported as unverified
rather than present. Headers the host lacks and any riscv64 Linux system has
are stubbed and listed as host gaps. If a project will not configure as a
cross build at all, it is configured for the host instead, and the report
says so.

### Build plans: everything the build runs, in order

A repository's build is rarely one configure. `./mfc.sh build` installs a
Python toolchain, builds FFTW, HDF5, SILO and LAPACK from source through
`toolchain/dependencies`, and then configures four targets with flags that
`toolchain/mfc/build.py` assembles. `--plan` runs a plan: that sequence written
down as data, one typed step at a time.

```console
$ will-it-riscv ~/src/MFC --plan examples/plans/mfc.json
```
```
Steps
  ✓ ci-packages              system-packages  12 packages
  ✓ fortran                  system-packages  1 package
  ✓ toolchain                python-install   114 packages resolved, 1 with nothing for riscv64
  ✓ dep-fftw                 cmake-configure  1 round: nothing required
  ...
  ✓ simulation               cmake-configure  4 rounds: fypp → MPI → FFTW
  ✓ post_process             cmake-configure  7 rounds: fypp → MPI → SILO → HDF5 → FFTW → LAPACK
  ✓ syscheck                 cmake-configure  3 rounds: fypp → MPI
  ✗ simulation-gpu-acc       cmake-configure  CMake Error at CMakeLists.txt:92 (message): ERROR: MFC
                                              with GPU processing is not currently compatible with GNU
                                              compilers.  Please use NVIDIA or Cray compilers.
                                              (optional: ./mfc.sh build --gpu acc)
  ✓ build                    python-run       imported 14 of the 114 packages the plan installs;
                                              called build tools 24 times
    plan check: ran 8 configures; 8 of the plan's 8 cmake steps match

  Will it riscv?  NO — jaxlib: required by the default build, and nothing for the target in the
                  index or archive checked

  required, and nothing public for the target (1)
    • pypi:jaxlib    publishes binary wheels, but none for riscv64 (...)
      toolchain → pyrometheus → jaxlib   · installed by default with pyrometheus; the build step
                                           itself never loads it
  required, and only source for the target (17)
    • pypi:numpy     publishes binary wheels, but none for riscv64 (...); Ubuntu 26.04 LTS
                     (resolute) ships python3-numpy for riscv64
      toolchain → numpy   · the build step imports it
    ...
    of these, the build step itself imports 3: cantera, numpy, pyyaml — the rest are installed
    by default all the same
  provided by the plan itself (5)
    • FFTW           provided by step 'dep-fftw'
    • fypp           provided by step 'toolchain'
    ...
  sudo apt install cmake g++ gcc gfortran hdf5-tools libblas-dev libfftw3-dev libhdf5-dev
    liblapack-dev libopenmpi-dev openmpi-bin python3 python3-dev
  how the 133 requirements are known: 6 shown by a configure, 114 declared by the project, 0 asked
  for by a configure that stopped, 13 only named by the plan's package lists
  note — optional: only with an extra flag or choice
    ./mfc.sh build --gpu acc:  step 'simulation-gpu-acc' failed: ... Please use NVIDIA or Cray compilers.
  note — optional in the configures: absent, and they carried on (4): blas, event_core, hwloc, libz
```

The answer is the pseudobuild's yes, probably, no or unknown, with one
more: *yes, after source builds*, when every default step completes once
what has only source for the target is built first.

| step kind | how it is answered |
| --- | --- |
| `python-install` | resolved against the index for the target's wheel tags — never installed. A manifest, or `packages` named in the plan itself |
| `system-packages` | looked up in the distro's riscv64 archive |
| `cmake-configure` | configured for real, with the plan's `-D` flags, confined to an empty sysroot that grows a stub for whatever the configure insists on |
| `meson-setup` | the same for Meson: `meson setup` as a linux/riscv64 cross build, with the plan's `-D` options and the Meson the project ships, if it names one |
| `python-run` | the project's own build driver, run on the host from a copy-on-write clone, with its build tools shimmed and every import it makes recorded |

**Nothing is emulated.** The sandbox is a pretend environment: whatever a
configure looks for, it is made to see, and whatever it complains about next
is added in turn. MPI is answered the way FindMPI asks its own questions — a
library, a header and a module directory per language, and the
`MPI_<LANG>_WORKS` it would have cached — rather than by an MPI anyone
built. HDF5 gets the `hdf5.h` its own `find_path` looked for, where it
looked.

Every step's asks land in one graph, keyed by ecosystem so that PyPI's numpy
and Debian's python3-numpy stay two things. A requirement that an earlier
step provides — the CMake configure wants `fypp`, which the Python toolchain
installs; it wants FFTW, which the dependency target builds from source — is
resolved to that step rather than looked for in the archive, and it stays off
the install line. `--graph deps.html` draws it: a node per dependency,
coloured by how it can be had, with a dashed edge from whichever step
provides it (`-f dot` gives the same graph to Graphviz).

**What the build step imports is worth knowing.** `./mfc.sh build` installs
114 Python packages and then hands over to `toolchain/main.py`, which loads
its modules lazily. So which of them the build step itself loads has an
exact answer: run the driver and watch. It runs in a copy-on-write clone of the repository,
with `HOME` in scratch, and with cmake, make, the compilers and MPI wrappers
replaced by shims that record their arguments and report success — nothing
is compiled. An import it cannot satisfy is installed for the host at the
version the plan resolved, or, if the plan provides nothing for it, stubbed;
then it runs again, the way a configure is unblocked. It sees only the
standard library and what it was given, never this tool's own packages.

MFC's build step imports 14 of the 114 — pyrometheus among them, which it
uses to generate `m_thermochem.f90` — and never jax or jaxlib; its driver ran
exactly the 8 configures the plan lists, with the same `-DMFC_*` flags.

**The default build is the minimal spec.** Everything the default install
brings in is required, whether or not the build step itself imports it:
jaxlib is required, because `./mfc.sh build` installs it — and without it
the flamelet examples that ship with MFC cannot run, which makes a build
without it a smaller thing than the default one. What the
trace shows is reported beside each requirement — *the build step itself
never loads it* — and never used to drop one. What only comes with an extra
flag or a choice left to the user is optional: a plan step marked
`"optional": true` says what turns it on (`"enabled_by": "./mfc.sh build
--gpu acc"`), and whatever only it needs is listed apart, under that flag.
MFC's OpenACC build is one: its configure refuses GNU compilers, and nothing
else exists for riscv64, so it is closed — and it does not change the
answer for the default build.

A plan is what a model is for. Reading a README, a CI workflow and a
bootstrap script, and writing down what they run, needs no knowledge of
riscv64. So the format is strict: typed steps, every one citing the lines it
came from — `{"at": "toolchain/bootstrap/python.sh:233", "quote":
"uv_install_with_retry \"$(pwd)/toolchain\""}` — and a citation that does not
say what the plan claims is reported before anything runs. `plan.PLAN_SCHEMA`
is the same shape as a JSON Schema, for constraining whatever writes one.
`examples/plans/mfc.json` is written by hand; every citation in it holds.

### Plans written by a model

`examples/plans/mfc.json` is written by hand. The plan format is built so a
model can write one, and `--plan-from-model` has one do it:

```console
$ will-it-riscv ~/src/MFC --plan-from-model --save-plan mfc.json
```

**The model is shown an evidence pack, not the repository.** `--show-pack`
prints it. It holds the parts of the tree that say how the project builds,
chosen the same way every time and numbered line by line:
- the install sections of the README and docs;
- the scripts at the top of the tree, what they source, and the Python
  drivers they hand over to (`mfc.sh` → `toolchain/bootstrap/python.sh` →
  `toolchain/main.py`);
- the manifests (for `pyproject.toml`, only the tables a build reads);
- the top of `CMakeLists.txt` and `meson.build`;
- the CI workflows that build it, and the scripts they call.

Linting and formatting scripts go last. Anything that doesn't fit the
budget (about 20k tokens) is named as not shown.

**Every answer is checked, and what is wrong goes back.** The answer might
not be JSON, the plan format might refuse a step, a quote might be nowhere
in the pack, or a citation might point to a line the pack never showed.
Each problem goes back as a sentence, and the model writes the plan again,
with three attempts in all. A quote the pack does show, on other lines than
the ones cited, is not a problem: the citation is moved to where the quote
is. The model only plans; the executor judges, so a bad plan costs a round,
not a wrong verdict.

**Every plan that holds is run, and what the run shows goes back too**, as
it does in labelling. A run can show these mistakes, and each goes back the
same way:
- a manifest that isn't there, or a configure pointed at a directory with
  nothing to configure (saying where the project's `CMakeLists.txt` is);
- a package name the distro has never had on any architecture, with the
  nearest names it does have;
- a compiler left as a placeholder (`${FC}`) that stopped the configure;
- a driver that configured something the plan does not list, or a plan
  step the driver never configured;
- a default step that moves an option from its default without citing why,
  or turns on a GPU backend;
- a `CMakeLists.txt` or `meson.build` at the root that no default step
  configures.

A configure that stops for any other reason is the build's answer, so it is
reported, not sent back. The report is the run the last plan was accepted
on. On the held-out split this feedback took the
fine-tuned 4B planner's verdicts from 0.825 to 0.925; on 60 fresh
repositories it left them at 0.867 and raised what its plans require a
little. It matters most for weaker planners: the base Qwen3-4B goes from
0.050 to 0.367 on those 60.

It writes a wire format: the plan format with every object closed,
evidence always `{at, quote}`, and `-D` flags as name/value pairs. That is
strict enough for schema-constrained decoding, and the same schema
constrains every backend:

| `--planner` | what writes the plan |
| --- | --- |
| `local` | the fine-tuned planner, run by llama.cpp on this machine (**the default**; both fetched the first time). `--model planner.gguf` runs another GGUF instead |
| `claude` | Claude Code's `claude -p` when it is installed (tools off, `--json-schema`, no API key needed; `--model sonnet` or `opus`, the default), else the Claude API (the `teacher` extra, `ANTHROPIC_API_KEY`; `--model` takes a model id) |
| `openai` | any OpenAI-compatible server, such as vLLM, llama.cpp or Ollama, with the schema as `response_format` (`--planner-url` and `--model`; `$PLANNER_API_KEY` if it wants a key) |

`claude-cli`, `anthropic` and `llama-server` name one backend exactly.
Without `--planner`, a `--planner-url` means `openai`, and a `--model` that
is not a `.gguf` file means `claude`.

The fine-tuned planner is one GGUF file, which `training/export.py` writes
from the adapter: Qwen3-4B-Instruct-2507, tuned on the teacher's verified
plans and on a quiz about how real builds find their dependencies (both
below). It is published as
Lagutenko-4B, [`aebarthyi/lagutenko-4B`](https://huggingface.co/aebarthyi/lagutenko-4B).
The first `plan` downloads it into the data directory
(`$WILL_IT_RISCV_DATA`, else `~/.local/share/will-it-riscv`), picking up
where it stopped if it is interrupted, and checks it against its SHA-256.

The server is llama.cpp's own prebuilt release, pinned, fetched the same
way: the Vulkan build where there is a Vulkan loader (it uses the GPU when
it finds one, and the CPU when not), the CPU build where there is none. A
server of your own wins: `$LLAMA_SERVER`, then `llama-server` on the PATH.
It is started with the settings the planner was evaluated with: a 40k
context, every layer on the GPU, flash attention, one slot and the model's
chat template. `$LLAMA_SERVER_ARGS` is added at the end, so
`LLAMA_SERVER_ARGS='-ngl 0'` runs on the CPU only.

**A dataset of verified plans.** `will-it-riscv-dataset` labels a corpus
with a teacher and verifies each label by running the plan. Anything that
fails because of the plan goes back to the teacher: the same mistakes a run
sends back to a planner, above. A configure that stops is the build's own
answer, not the plan's fault. Each example keeps its pack, plan, transcript
and run, and the verified ones export as chat-format JSONL: the pack in,
the plan out.

```console
$ will-it-riscv-dataset seed-pypi --top 3000 --want 350 --to examples/corpus/pypi.jsonl
$ will-it-riscv-dataset label --corpus examples/corpus/repos.jsonl --out data/repos --jobs 2
$ will-it-riscv-dataset export --out data/repos --to data/sft
```

The other subcommands look after a labelled set. `relabel` lets the default
configures mark each package a plan states as required or optional;
`revise` sends verified labels back to the teacher with what running them
shows; `rerun` runs them again on today's executor, or against another
`--distro`; `check` finds each example's tree and rebuilds its pack; and
`status` says how far a run has got and what it cost. Labelling runs
against Debian 13 (`debian:trixie`), whatever `plan`'s default.

`examples/corpus` holds four corpora:
- `pypi.jsonl`: 309 of the 3000 most-downloaded PyPI projects that compile
  and have no riscv64 wheel;
- `repos.jsonl`: 45 source repositories whose build is a script, a
  toolchain or a CI recipe (MFC, GROMACS, LAMMPS, CP2K, GDAL, OpenCV, jax,
  XLA, CPython, …);
- `repos-large.jsonl`: 166 more, from desktop applications to HPC codes
  (Audacity, Blender, darktable, …);
- `repos-spack-2.jsonl`: 120 projects from Spack's package repository.

One in ten entries is held out by a hash of its name, unless the entry
names its split: five script-driven repositories in `repos.jsonl` and four
in `repos-large.jsonl` are pinned to the held-out set, and every entry in
`repos-spack-2.jsonl` names its own (70 held out, 50 for training).

**A quiz, answered by the builds themselves.** `will-it-riscv-quiz build`
configures the dataset's CMake projects, plainly and then with each option
that guards a lookup turned on, and writes down what CMake did with each
dependency: required, optional, or never asked for. It also asks which
Debian packages make a `find_package` succeed, answered from Debian's
Contents index. No model writes an answer. `ask` puts the questions to a
served model and scores it, `export` writes them as chat JSONL, and
`sample` draws a small balanced set for people to take. Questions inherit
the split of their repository. 792 of them are among the planner's 1,266
training examples.

A pilot of eight repositories, with Opus as the teacher through
`claude -p`, gave 8 verified plans for $2.45. Six were right first time.
MFC's first plan ran `./mfc.sh build` but listed none of the eight
configures the driver runs; the plan check named them, and the second
plan had them all, `--gpu acc` included, marked optional. numpy's first
plan tripped on the plan format rejecting a Meson option with a dash in
its name, and that was fixed in the format, not left to the model.

**Evaluation.** `will-it-riscv-eval` scores planners on the held-out
repositories. Each planner gets the same pack the reference was labelled
from:
- `parsed`: whether it produced a plan at all;
- `cited` and `grounded`: how many of its citations hold, and how many are
  lines the pack showed;
- precision, recall and F1 of its steps against the verified plan, matched
  softly (package lists by overlap, configures by directory and `-D`
  flags);
- agreement on which steps are optional, and on ordering;
- with `--execute`, whether running it reaches the same verdict (`verdict`)
  and requires the same things: what the runs show is needed (`required`),
  and everything the plans name, package lists included (`req_all`);
- what it cost: thousands of tokens, the same less what a prompt cache gave
  back, and dollars where the backend says.

Each planner gets a row over every repository, then one for script-driven
repositories and one for PyPI packages. The baselines are `autoplan`, the
build-file reader recursion uses, and `teacher-first`, the teacher's plan
before any of it was sent back. On the pilot's three held-out repositories,
autoplan finds 19% of the steps. The teacher's first plans were all
accepted as written, so they match their references by construction; that
baseline only means something where revisions happened.

```console
$ will-it-riscv-eval --out data/repos --planner openai --planner-url http://localhost:8000/v1 --planner-model qwen3-8b-plans
```

A planner gets one attempt unless `--rounds` gives it more, and a run's
mistakes go back only with `--execute --run-feedback`: `--rounds 3
--execute --run-feedback` is what `plan` does.
`--planner` takes `claude-cli`, `anthropic` and `openai` as above, `agent`
for a model that plans with tools (Claude through `claude -p`, or a model
at `--planner-url`), and `saved` to rescore the answers an earlier run kept
with `--transcripts`. `--unblock claude` hands the configures the rules
cannot get past to the agent, and counts what it spends. Runs go against
`--distro debian:trixie` unless told otherwise, and the references must
have been run against the same archive (`will-it-riscv-dataset rerun
--distro`).

`bench/board.py` sets `--json` files from several evals side by side, over
only the examples every planner scored, and `--against PLANNER` adds each
other planner's paired difference from it with a 95% bootstrap interval.
`bench/dev.txt` is a fixed dev set from the training split (30 repositories,
10 PyPI packages), for working on the harness without touching the
held-out set.

### Recursion: down the tree, and back up

A plan's answer stops at its frontier: what the default build requires that
has only source, or nothing at all, for riscv64. `--recurse` gives each of
those the root's treatment. It fetches the source at the version the plan
resolved: the sdist, or, when there is none (jaxlib publishes only wheels),
the repository the package's metadata names, checked out at the release tag.
It plans the build from the package's own files, runs that plan in the same
pretend environment, and repeats for whatever that leaves unresolved. Then
it builds the answer back up from the leaves.

```console
$ will-it-riscv ~/src/MFC --plan examples/plans/mfc.json --recurse
```
```
Recursion
  fetched 19 packages and configured 9, in 181s
  MFC
  ├─ ✓ cantera 3.2.0  buildable — its configure completed: Python → Boost → BLAS → HDF5 → LAPACK   [sdist cantera-3.2.0.tar.gz]
  │  │    installed first: libboost-all-dev, libhdf5-dev, liblapack-dev, libopenblas-dev, the target's Python (python3-dev)
  │  └─ ✓ numpy 2.5.4  buildable — its configure completed: cython   [sdist numpy-2.5.4.tar.gz]
  ├─ ✓ contourpy 1.4.0  buildable — its configure completed: pybind11   [sdist contourpy-1.4.0.tar.gz]
  ├─ ~ ffmt 0.4.5  probably — read, not configured: builds with cargo   [sdist ffmt-0.4.5.tar.gz]
  │       installed first: cargo, rustc
  ├─ ...
  ├─ ~ jaxlib 0.11.2  probably — read, not configured: builds with bazel   [git https://github.com/jax-ml/jax@jax-v0.11.2]
  │  │    hermetic Python: rules_python 2.2.0 has CPython 3.12.13 for riscv64-unknown-linux-gnu ✓
  │  │    its hermetic C++ toolchains are for linux aarch64 and x86_64 only; --config=clang_local builds with
  │  │    the machine's own compiler instead (clang)
  │  │    its Python packages are downloaded as wheels for aarch64 and x86_64 only; for riscv64, local_wheels
  │  │    takes ml_dtypes, numpy, scipy from dist/, built first
  │  │    installed first: clang
  │  ├─ ~ bazel-bootstrap 8.7.0  probably — bootstraps with compile.sh, which builds Bazel with itself: read,
  │  │                           not run   [git https://github.com/bazelbuild/bazel@8.7.0]
  │  │       installed first: g++, openjdk-21-jdk-headless, python3, unzip
  │  ├─ ✓ ml-dtypes 0.6.0  buildable — its configure completed: Python   [sdist ml_dtypes-0.6.0.tar.gz]
  │  │  │    installed first: the target's Python (python3-dev)
  │  │  └─ ✓ numpy 2.5.4  (above)
  │  ├─ ✓ numpy 2.5.4  (above)
  │  └─ ✓ scipy 1.18.1  buildable — its configure completed: cython → pythran → numpy → pybind11 → OpenBLAS
  │     │    installed first: libopenblas-dev
  │     └─ ✓ numpy 2.5.4  (above)
  ├─ ✓ matplotlib 3.11.2  buildable — its configure completed: pybind11   [sdist matplotlib-3.11.2.tar.gz]
  ├─ ...
  ├─ ✓ pandas 3.0.6  buildable — its configure completed: cython   [sdist pandas-3.0.6.tar.gz]
  ├─ ...
  ├─ ~ ruff 0.6.5  probably — read, not configured: builds with cargo   [sdist ruff-0.6.5.tar.gz]
  │       installed first: cargo, rustc
  ├─ ...
  └─ ~ typos 1.51.1  probably — read, not configured: builds with cargo   [sdist typos-1.51.1.tar.gz]
          installed first: cargo, rustc (rustup)

  build order, dependencies first:
    numpy 2.5.4 → cantera 3.2.0 → contourpy 1.4.0 → ffmt 0.4.5 → h5py 3.16.0 → imageio-ffmpeg 0.6.0 →
    bazel-bootstrap 8.7.0 → ml-dtypes 0.6.0 → scipy 1.18.1 → jaxlib 0.11.2 → matplotlib 3.11.2 → ...

  Will it riscv, all the way down?  PROBABLY — after building 19 packages from source, some of them only
  read, not configured
```

| status | meaning |
| --- | --- |
| `buildable` ✓ | its own configure or setup ran to the end, and everything under it is buildable |
| `probably` ~ | its build could only be read (Cargo, setuptools and Bazel builds are not configured), and nothing under it is blocked |
| `blocked` ✗ | something it requires has nothing public for the target, or reading its build found something the target cannot have |
| `unknown` ? | a configure stopped, a fetch failed, a bootstrap cycle, or the budget ran out |

The build order is a post-order walk of the tree, dependencies first, and
that order is the port plan. *Installed first* is what each package's own
plan found it needs before it can be built, beyond the Python packages
already in the tree. That means archive packages, the target's Python, and
any tool the archive has too old. typos declares `rust-version = "1.95"`
and Ubuntu 26.04 has rustc 1.93, so it gets rustc from rustup. ruff asks
for 1.76, and Ubuntu's rustc is enough.

Nine of the nineteen configure for real: cantera and ml-dtypes with CMake,
and numpy, scipy, matplotlib, pandas, contourpy, scikit-image and siphash24
with Meson (below). jaxlib builds with Bazel, which is read rather than run,
and the Bazel it needs is followed to its own source (below that).

A fetched package's configure is where the pretend environment has to be
most accommodating, because these configures lean on what pip's build
isolation would have given them:

- **Their Python is the build host's.** scikit-build-core pins the
  interpreter running the build, and a configure asks it questions:
  ml-dtypes runs `import numpy; print(numpy.get_include())` and stops if
  that fails. So every configure is given this host's interpreter, run
  without its own site-packages. When an import kills it, the module is
  installed for the host at the version the plan resolved (from wheels
  only, so no fetched `setup.py` ever runs), or stubbed if nothing says
  which version. Then the configure runs again. An import the configure
  catches and does without is left alone. This is a build tool on the
  build host; nothing is run for riscv64.
- **What they fetch is theirs.** cantera's configure fetches yaml-cpp,
  Eigen and SUNDIALS with FetchContent. Each is downloaded once per run,
  not once per round. yaml-cpp asks for a CMake older than 3.5, which
  CMake 4 refuses. The configures run Debian 13's CMake, 3.31, which this
  tool installs and which still accepts it (`$WILL_IT_RISCV_CMAKE` names
  another). Where the CMake that runs is 4 or later, the configure is rerun
  with `-DCMAKE_POLICY_VERSION_MINIMUM=3.5` (CMake's own suggested fix) and
  a note says so. Ubuntu 26.04 ships CMake 4.2, so a build there meets the
  same refusal.
- **What the host only pretends to have is not there.** pyenv's and asdf's
  shims are dropped from the configure's PATH: a shim exists for every
  program any installed Python has, and fails when the active one has none,
  so a configure would find a `cython` that only says "command not found".
- **What steers the host's own builds stays with the host.** A
  configure's PATH holds only the system's directories and this Python's.
  With `/opt/rocm/bin` on it, PyTorch's configure said "ROCm enabled"; a
  vendor's `clang++` ahead of `/usr/bin` stopped saltfm. `CC`, `CXX`,
  `CFLAGS`, `CPATH`, `LIBRARY_PATH`, `CMAKE_PREFIX_PATH` and the like are
  left out as well. If the system has no compiler at all, as on an HPC
  login node with a compiler module loaded, that compiler's directory is
  kept, and the same goes for `make`, `ninja`, `pkg-config` and `git`.
- **What needs a linker is answered for the target.** FortranCInterface
  links a program of C and Fortran to prove the two compilers agree, and
  nothing links here. On the target gcc and gfortran do agree, so the
  toolchain file says so.
- **What they build is on the target's side.** Confinement re-roots every
  search into the empty sysroot. A path that already points into the
  build tree or the source tree is searched as written, though. SUNDIALS
  compiles a Fortran library in its build tree to work out name mangling,
  then looks for it there; the host's own libraries stay out of reach.

**This runs the configures of everything it fetches.** They are third-party
build scripts, run on this host: confined to a scratch sysroot, never
compiled, never emulated, but run. `--recurse` is opt-in and bounded by
`--recurse-depth` (3) and `--recurse-limit` (40 packages); `will-it-riscv
plan --recurse` keeps those defaults.

### The graph, to explore

`--graph deps.html` writes the plan's dependency graph as one page that
needs nothing installed and fetches nothing when it opens: the steps on the
left, what each needs flowing right, coloured as the report's tiers are
(in the archive, made by a step of the plan, only source for the target,
not settled, nothing public). Optional dependencies are dashed and can be
hidden; so is what a configure looked for and did without. Click a node for
why the build needs it — the chain from a step down to it — what settled it,
and the lines the plan cited. Python packages that have wheels for the
target, and lead to nothing that does not, fold into one node a step: MFC's
toolchain brings in 84 of them. With `--recurse`, a package built from
source carries its recursion: its status, its source, its place in the build
order. It pans, zooms, searches, and follows the system's light or dark
theme. `--graph deps.svg` writes the same picture, still.

The layout is the tool's own (layered, after Sugiyama), so there is no
Graphviz to install.

### A build script, from the plan

`--build-script build-riscv.sh` writes the build itself, for the target:
the plan's steps, each followed by the compile and install its build system
does after a configure. It is written by code from the plan the run held up,
not by the model, and every value the plan carries goes into it quoted. Run
it on the riscv64 machine, from the checkout. In order, it:

1. checks the machine is the one planned for (`ANY_ARCH=1` to run it
   elsewhere), says so if the distro is not, and checks it was started
   from the checkout;
2. installs everything the build needs from the archive, in one
   `apt-get install`;
3. installs Rust with rustup where the archive's is too old;
4. makes a virtual environment for the Python steps, since Debian and
   Ubuntu refuse pip installs into the system's Python;
5. builds what the target has only as source, dependencies first. When the
   build needs any, writing the script turns `--recurse` on to find them and
   the order. pip builds the Python packages; a tool such as Bazel, whose
   bootstrap was read and not run, has no recipe, so the script stops
   until it is on the PATH;
6. runs the plan's steps in order: pip installs, CMake and Meson configures
   each built and installed into `$PREFIX`, and the project's own driver.
   A driver runs the configures it was seen to run, and pip's build runs
   the configure a meson-python or scikit-build backend owns, so the script
   leaves those to them.

`DRY_RUN=1` prints every command and runs none; `SKIP_APT=1` skips the
packages; `SRC` is the checkout (the current directory), `WORK` the build
directories (`$SRC/_riscv`), `PREFIX` where everything is installed
(`$WORK/install`), `VENV` the virtual environment (`$WORK/venv`), and
`JOBS` how wide (`nproc`); an optional step runs with `WITH_<STEP>=1`
(MFC's `WITH_SIMULATION_GPU_ACC=1`). nlopt's script, run on x86_64 with
`ANY_ARCH=1 SKIP_APT=1`, configures, builds and installs `libnlopt.so` in 11
seconds.

### Meson builds are set up the same way

numpy, scipy, matplotlib and pandas build with Meson, through meson-python.
A plan's `meson-setup` step, and `--pseudobuild` on a tree with a
`meson.build` and no `CMakeLists.txt`, runs `meson setup` the way a CMake
configure is run. It is a cross build for linux/riscv64, confined to an
empty sysroot, set up in a copy-on-write clone of the tree because Meson
writes into it, and unblocked round by round. The Meson is the one the
project ships, if it names one, else the build requirement's version, else
Debian 13's (1.7, a dependency of this tool): qemu wants 1.6, libvips 1.4,
and an older host Meson stops both before they ask for anything. A project
that wants newer still (vlc asks for 1.10) gets a Meson from PyPI that
meets its `meson_version`, and a note says so.

```console
$ will-it-riscv ~/src/numpy --pseudobuild
```
```
Pseudobuild
  configured as linux/riscv64, confined to an empty sysroot; configure completed over 2 rounds in 16s
  — 8 dependency probes observed
  rounds: cython → completed
  note: configured with the Meson it ships (vendored-meson/meson/meson.py)

  Will it riscv?  YES — every dependency a default build demands is in Ubuntu 26.04 LTS (resolute)
                  for riscv64
    cython comes with its Python build requirements (pip)
    shown by configuring, not compiling: the configure accepts a linux/riscv64 build given only these

  hard requirements, in the order the build demanded them:
    1.  cython  stopped the configure    comes with its Python build requirements
  proven optional — absent, and the configure carried on (4): blas, blis, flexiblas, lapack
  graph: 5 nodes, 5 edges — -f dot | dot -Tsvg > deps.svg
```

| Meson said | what it is given |
| --- | --- |
| `Dependency "OpenBLAS" not found` | an `OpenBLAS.pc` where pkg-config is confined to look, and an empty `libOpenBLAS.a` |
| `C shared or static library 'foo' not found` | an empty archive the host's linker takes (an empty file crashes ld64) |
| `Unknown compiler(s): [['cython']]`, `Program 'pythran' not found` | the build requirement, installed for the host from a wheel at the plan's version; a stub that answers its version otherwise |
| `C header 'foo.h' not found` | that header, carrying version macros |
| `ModuleNotFoundError` in a script Meson ran | the module: pandas' `generate_version.py` imports versioneer, matplotlib runs `python -m setuptools_scm` |
| `Problem encountered: …` | blame by experiment, as for CMake |
| `Meson version is 1.7.2 but project requires >= 1.10` | a Meson from PyPI that meets it, installed for the host, once |

The cross file gives Meson this host's compilers with `needs_exe_wrapper =
false`, which is the Meson equivalent of the CMake loop's pass-through
emulator. What a check compiles runs here, and nothing riscv64 runs at all.
The Meson is the one the build uses: numpy's vendored fork when
`[tool.meson-python]` names it, otherwise meson at the version the plan
resolved, installed from a wheel. A link check still runs against this
host's own SDK. On a Mac, Accelerate answers numpy's `dependency('blas')`,
and that is reported as claimed and unverifiable, not as found. What a
build compiles from its own subprojects, like matplotlib's freetype and
qhull, is noted, not listed as something to install.

### Bazel builds are read for what they download

Bazel is not run. Analysing jaxlib's build would mean fetching XLA and LLVM
(gigabytes) to answer questions its `MODULE.bazel` already answers.
Whatever a Bazel build fetches as source, Bazel builds for any platform
with a C++ compiler. The things that can be missing on riscv64 are the
things it downloads prebuilt:

| what it downloads | what is checked |
| --- | --- |
| hermetic Python, `python.toolchain(python_version = "3.12")` | rules_python's own manifest, read from a sparse checkout of the version the build pins: 2.2.0 has CPython 3.12.13 for `riscv64-unknown-linux-gnu` |
| hermetic C++, `register_toolchains(...linux_x86_64...)` | whether any is for the target; if not, whether a `.bazelrc` config builds with the machine's own compiler (jax's `clang_local`) |
| wheels, `pip.parse(download_only = True, target_platforms = [...])` | whether the target is listed; if not, what `local_wheels` lets `dist/` supply. Those wheels are built first: numpy, scipy and ml-dtypes, already in the tree. The wheels the repository builds itself, jax and jaxlib, are left out |

The build Bazel itself needs is the other half. jax pins Bazel 8.7.0 in
`.bazelversion`, and Ubuntu 26.04, like Debian 13, has 4.2.3 for riscv64.
So the recursion follows `bazel-bootstrap` upstream: a 2 MB sparse checkout
of the `8.7.0` tag, just `compile.sh`, `scripts/bootstrap/`, `MODULE.bazel`
and `.bazelversion`. It reads what the bootstrap needs from those scripts:
`JAVA_VERSION=${JAVA_VERSION:-21}`, the `unzip` in its tool check,
rules_python's autodetecting toolchain, `-std=c++17`. Each of those is
checked against the archive. compile.sh builds Bazel with itself, so it is
read, not run. jax's `build.py` downloads Bazel only for x86_64 and aarch64
Linux, macOS and Windows; anywhere else it takes `--bazel_path`, which is
where the bootstrapped one goes.

### Meson projects are asked, not guessed at

`meson introspect --scan-dependencies` walks a project's `meson.build` files,
recursing through `subdir()`, and reports every dependency with whether it is
required and whether it sits behind a condition — exactly the classification
this tool reconstructs by hand everywhere else, from the parser that owns the
language. When it works, its answer wins: a regex over `configure.ac` that
knows nothing about `AC_ARG_WITH` must not outvote Meson about Meson's files.
On PostgreSQL that moves ldap, libcurl, libxml2, libxslt, lz4, numa and pam
out of the install line, where they belong.

It is an enhancement, never a dependency. Meson resolves the project's
languages first, so a project declaring Rust makes it run `rustc --version`
and fail without a Rust toolchain — QEMU does exactly that. On any failure the
regex readers carry on unchanged, and the report says which happened.
`--no-meson-introspect` turns it off.

### Optional dependencies are separated from required ones

Most `find_package` calls in a large project sit inside a branch nobody
enables. AdaptiveCpp is the clean case: its CUDA, ROCm and Level Zero backends
each live in `if(WITH_..._BACKEND)`, and those default to whatever
autodetection found — which on a riscv64 machine is nothing. Reported as
requirements, a project whose minimal build needs LLVM and a C++ compiler
looks like it needs three vendor GPU stacks.

So the CMake option defaults are read first, then the `if`/`elseif`/`else`
structure is walked to decide what a build with **no `-D` flags** reaches.
Evaluation is three-valued — true, false and *unknown* — and unknown counts as
reachable, because guessing a real dependency away is the error that matters.

| understood | |
| --- | --- |
| `option(X "" OFF)` | and the bare `option(X "")`, which CMake defaults to OFF |
| `set(X OFF CACHE BOOL …)` | the other way projects declare a switch |
| `set(X ${CUDA_FOUND} CACHE …)` | defaults to autodetection, so off unless asked for |
| `if(CUDA_FOUND)` | false when the package has no build for the target at all |
| `NOT` / `AND` / `OR`, nesting, `elseif`, `else` | three-valued throughout |
| `WIN32`, `APPLE`, `MSVC` | false for a Linux target, so those branches are dead |
| `find_package(X QUIET)` without `REQUIRED` | a probe, not a requirement |
| `STREQUAL`, `MATCHES`, `DEFINED`, `EXISTS` | *unknown* — kept, not guessed away |

Beyond CMake: Meson's `dependency('x', required: false)` says so outright, and
FFmpeg-style `enabled libx264 && require_pkg_config …` reports the
`--enable-libx264` that would turn it on. Accelerator packages a CI job
installs (`rocm-dev`, `nvidia-cuda-toolkit`, `intel-oneapi-*`) are treated the
same way — the same judgement, for the half of the evidence that has no
conditions to walk.

Required anywhere beats optional elsewhere: a dependency found unconditionally
in one file is required, whatever another file does with it.

### Build, test and documentation dependencies are separated

CI installs more than a build needs. The install line covers only what is
required to *compile*; test harnesses and documentation toolchains are listed
separately and left out of it.

Classification is by package name first, because a single command routinely
mixes purposes — git installs `gcc`, `libcurl4-openssl-dev`, `apache2` and
`subversion` in one `apt-get`, and no amount of surrounding context separates
those. The step name is consulted only for packages the name map does not
recognise, so a build tool stays a build tool even when the step installing it
is called "run tests". Anything still unclassified counts as a build
dependency: dropping a real one is a worse error than keeping a test one.

```
git      45 build   12 test (apache2, cvs, subversion, valgrind…)   4 docs (asciidoc, xmlto…)
redis    13 build    9 test (tcl, tclx, valgrind, lcov…)
qemu    109 build    0 test
```

Two repository-specific things it knows:

- **Bundled libraries are optional.** A project that ships its own copy of a
  library in `external/`, `third_party/` or `vendor/` can build without the
  system package. GROMACS bundles sixteen; those are listed separately from
  the ones you actually have to install.
- **An uninitialised submodule is a loud warning.** It scans perfectly
  cleanly and reports nothing, which is the most dangerous way for this tool
  to be wrong.

**Manifest mode** is the dependency walk described above. A manifest that is
not at the repository root usually describes something else — documentation,
language bindings, a test harness — so it is reported rather than silently
adopted, and you point at it directly if you want it analysed.

**Unless the project's own scripts install it.** MFC is built with
`./mfc.sh build`, which sources `toolchain/bootstrap/python.sh`, which
pip-installs `toolchain/` into a venv before any CMake runs. So the shell
scripts at the root are read too: `source` chains and invoked scripts are
followed, the usual spellings of the repository root are resolved (`$(pwd)`,
`$(dirname "$0")`, `${BASH_SOURCE%/*}`, `git rev-parse --show-toplevel`, and
variables set from them), and a wrapper function that forwards `"$@"` to
`pip install` counts as an installer — MFC's goes through two. Whatever they
install is analysed along with the declared dependencies, and the report says
how it was reached:

```
This project
  installs  toolchain/pyproject.toml
            before building, by mfc.sh:53 → toolchain/bootstrap/python.sh:233 — analysed below

Nothing installable (1)
  jaxlib   0.11.2
```

That is MFC's riscv64 answer, and it is not in CMake: `jaxlib` has no riscv64
build at all, and both jax and pyrometheus pull it in, so `./mfc.sh build`
stops at the venv bootstrap. The scripts are read, never run, and conditions
are not evaluated — an install anywhere in them counts.

[hpccm]: https://github.com/NVIDIA/hpc-container-maker

## Install

It is not on PyPI yet; install it from the repository:

```console
$ pip install git+https://github.com/aebarthyi/will-it-riscv
$ pip install 'will-it-riscv[teacher] @ git+https://github.com/aebarthyi/will-it-riscv'   # with the Claude API
```

## Use

```console
# have the planner write the build plan, run it, and answer (see Quick start)
$ will-it-riscv plan ~/src/MFC

# the same, drawn, against Debian 13, with the plan kept
$ will-it-riscv plan ~/src/MFC --distro debian:trixie --save-plan mfc.json --graph mfc.html

# and a script that builds it on the riscv64 machine (then: bash build-riscv.sh, there)
$ will-it-riscv plan ~/src/MFC --build-script build-riscv.sh

# with Claude writing the plan
$ will-it-riscv plan ~/src/MFC --planner claude

# what this machine has for `plan`, and lacks
$ will-it-riscv doctor

# the project in the current directory, read rather than configured
$ will-it-riscv

# a specific file, with extras
$ will-it-riscv pyproject.toml -E all

# a requirements file, targeting musl and Python 3.11
$ will-it-riscv requirements.txt --target riscv64-musl1.2 --python 3.11

# ad-hoc packages, no file needed
$ will-it-riscv -p 'numpy>=2' -p pandas

# a source repository: scans the tree and its CI configuration
$ will-it-riscv ~/src/gromacs

# only resolve declared dependencies, do not read the source tree
$ will-it-riscv --no-scan

# machine-readable, for CI
$ will-it-riscv -f json -o riscv-report.json

# configure it as linux/riscv64 and draw what it asked for
$ will-it-riscv --pseudobuild -f dot | dot -Tsvg > deps.svg

# run a whole build plan, and draw the one graph all its steps make
$ will-it-riscv ~/src/MFC --plan examples/plans/mfc.json -f dot | dot -Tsvg > mfc.svg

# then fetch and configure whatever it needs from source, all the way down
$ will-it-riscv ~/src/MFC --plan examples/plans/mfc.json --recurse

# have a model write the plan instead, then run it (what `plan` does)
$ will-it-riscv ~/src/MFC --plan-from-model --save-plan mfc.json

# just the names of everything that is not pure Python
$ will-it-riscv -f list
```

### Exit codes

| code | reading a project, a manifest or packages | `plan`, `--plan`, `--plan-from-model` |
| --- | --- | --- |
| 0 | everything installs from wheels or is pure Python | yes: the build has everything it needs from the archive |
| 1 | some packages, or the project itself, must be built from source | probably, or yes once some packages are built from source |
| 2 | something is blocked, unresolvable, or has no distribution at all | no, or unknown: a step could not be got past |
| 3 | the index itself was unusable | the same |

So `will-it-riscv || exit 1` is a reasonable CI gate. A project that
compiles code of its own always exits 1 or more when its tree is read: it is
built from source whatever its dependencies do.

### Targets

`--target` takes an architecture, a Rust-style triple, or an explicit libc floor:

```
riscv64                        riscv64 glibc >= 2.39 (the manylinux baseline)
riscv64-unknown-linux-gnu      the same
riscv64-musl1.2                musllinux_1_2_riscv64
riscv64-glibc2.36              an older glibc than manylinux images target
aarch64                        nothing here is riscv-specific
```

### Distros

`--distro` selects the archive to check system packages against:

| identifier | |
| --- | --- |
| `debian:trixie` | Debian 13 — riscv64 is an official release architecture as of 13 |
| `debian:sid` | Debian unstable |
| `debian:forky` | Debian 14 |
| `ubuntu:noble` | Ubuntu 24.04 LTS |
| `ubuntu:plucky` | Ubuntu 25.04 |
| `ubuntu:questing` | Ubuntu 25.10 |
| `ubuntu:resolute` | Ubuntu 26.04 LTS (**default**) |

A bare family name resolves to the current Debian stable (`debian` →
`debian:trixie`) or the newest Ubuntu LTS (`ubuntu` → `ubuntu:resolute`).
Ubuntu's riscv64 packages live on ports, and both `main` and `universe` are
searched. The package index is downloaded once and cached for a week;
`--no-distro` skips it.

Fedora does not carry riscv64 in its primary repositories, so Fedora package names
are reported as suggestions without an availability check.

## What it does not do

**It does not build anything.** Every conclusion about a source distribution is
static inference from reading the archive. That is a deliberate trade: it runs
anywhere in seconds, needs no emulator, and cannot execute a hostile `setup.py`.
It also means a package reported as buildable can still fail on a detail no
static read would catch. Treat the output as a work list, not a guarantee.
`--pseudobuild`, `--plan` and `--recurse` go further. They run configures
and build drivers, and `--recurse` runs those of every package it fetches.
Nothing is compiled or emulated, but those scripts do run, so use them on
code you would be willing to configure yourself.

### Build systems

| | read from |
| --- | --- |
| CMake | `find_package`, `pkg_check_modules`, `find_library` |
| Meson | `dependency()`, `find_library()`, `project()` languages |
| Autotools | `AC_CHECK_LIB`, `PKG_CHECK_MODULES`, `AC_SEARCH_LIBS` |
| Make | `-l` flags in `*LIBS` / `*LDFLAGS` variables, `pkg-config` calls |
| Hand-written `configure` | FFmpeg-style `require_pkg_config`, `-l` flags (a generated autoconf script is skipped — its `configure.ac` already said it) |
| Cargo | `*-sys` crates, `pkg_config::probe` in `build.rs` |
| setuptools | `libraries=[…]`, `find_library`, `pkgconfig` calls |
| any C/C++ | `#include` directives, against a 130-entry header map |

SCons is detected but not scraped, and Bazel is read only for what it
downloads prebuilt; the report says so rather than implying the project has
no dependencies.

**Repository scanning is inference too.** The option and `if` walk above
decides what a build with no `-D` flags reaches, but a condition it cannot
evaluate counts as reachable, so an optional backend can still appear in
the list. Vendor GPU stacks (CUDA, ROCm, SYCL, oneAPI) are filtered out
entirely, since none of them exists for riscv64 and all are opt-in.

**It is not a resolver.** For each package it takes the highest version satisfying
the constraints seen so far, and revisits when a later edge tightens them. It does
not backtrack. Where pip and uv disagree with it, they are right, and `uv pip
compile --python-platform riscv64-unknown-linux` is a way to check it. What this
adds is which of those versions must be compiled, and what the compile needs.

**Guessed package names are marked.** A library not in the curated map gets a
Debian name guessed from convention (`libfoo-dev`). The report says which ones
those are. Do not paste them into a provisioning script unchecked.

## Cache

Index pages, wheel metadata, source distributions and distro package lists are
cached under your platform cache directory (override with `WILL_IT_RISCV_CACHE`).
`will-it-riscv --clear-cache` empties it.

The planner and llama.cpp's server are not cache: they go in the data
directory (`$WILL_IT_RISCV_DATA`, else `~/.local/share/will-it-riscv`), and
`--clear-cache` leaves them alone. Delete that directory to remove them.

## License

MIT
