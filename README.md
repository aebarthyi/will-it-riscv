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

## Why

Wheel availability for riscv64 is real but thin. [manylinux gained
`manylinux_2_39_riscv64` images in 2025][manylinux] and packages like `aiohttp`,
`regex`, `markupsafe`, `charset-normalizer` and `rpds-py` publish riscv64 wheels
today — but most of PyPI does not, and `pip install` on the target is a slow,
serialized way to discover that. [uv cannot resolve for riscv64 at all][uv-issue].

The alternative is finding out one traceback at a time on an emulated board.

[manylinux]: https://github.com/pypa/manylinux
[uv-issue]: https://github.com/astral-sh/uv/issues/8889

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
  configured as linux/riscv64, confined to an empty sysroot; configure completed over 3 rounds
  in 23s — 24 dependency probes observed
  rounds: OpenMP → FFTW → completed
  blame by experiment: OpenMP ✓, FFTW ✓   (✓ the error moved once it existed; ✗ it did not,
  so the stub was taken back out)

  Will it riscv?  YES — every dependency a default build demands is in Debian 13 (trixie) for riscv64
    sudo apt install libfftw3-dev    # the minimum a default configure demanded
    OpenMP comes with the compiler (GCC's libgomp)
    shown by configuring, not compiling: the configure accepts a linux/riscv64 build given only these

  hard requirements, in the order the build demanded them:
    1.  OpenMP  shown by experiment  cmake/gmxManageOpenMP.cmake:46        comes with the compiler
    2.  FFTW    shown by experiment  cmake/gmxManageFFTLibraries.cmake:67  libfftw3-dev ✓
  proven optional — absent, and the configure carried on (3): HDF5, ImageMagick, MPI
  claimed by compile-only checks, unverifiable without a target linker: BLAS, LAPACK
  located on this host, need untested: LATEX, UnixCommands
  graph: 9 nodes, 9 edges — -f dot | dot -Tsvg > deps.svg
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
| `(missing: … SSL Crypto)`, or `links to: OpenSSL::SSL but the target was not found` | the component's `OPENSSL_SSL_LIBRARY` |
| `The following required packages were not found: - libpsl` | a `libpsl.pc` in the only directory pkg-config may search |
| `file failed to open for reading: …/proj.h` | that header, carrying every common spelling of a version macro |
| `… are set to NOTFOUND … FOO_LIBRARY linked by target` | the variable a target links |
| `linux/fs.h header not found` | the header — reported as a **host gap**, not a dependency: every riscv64 Linux system has it; the Mac SDK does not |

**Blame by experiment.** Some configures die without naming anything.
GROMACS narrates `-- Could NOT find OpenMP` and then fails in its own words;
its FindFFTW never says "Could NOT find" at all. So the loop suspects the
misses nearest the error — first any the error message mentions by name, then
the find_packages the trace saw it name — and stubs the likeliest the way its
own Find module looked for it: the trace records every `find_library`,
`find_path` and `pkg_check_modules` each module ran. If the error moves, the
suspect is a hard requirement, shown by experiment. If it does not, every
trace of the stub is taken back out, so a later round cannot find it, and the
next suspect is tried — at most three per stuck point.

Status misses are never stubbed on sight — faking an optional dependency
would erase the very evidence that it is one — and nothing is ever written
outside the scratch directory.

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
| **unknown** | the configure stopped somewhere no stub gets past, and says where |

| repo | static | pseudobuild | hard requirements | will it riscv? |
| --- | --- | --- | --- | --- |
| gdal | 297 required / 4 optional | **202 / 99** | PROJ | yes — `libproj-dev` |
| curl | 47 / 1 | **29 / 19** | OpenSSL, libpsl | yes — `libssl-dev libpsl-dev` |
| opencv | 60 / 13 | **22 / 51** | none — it bundles every codec | yes |
| gromacs | 20 / 0 | **13 / 7** | OpenMP, FFTW (by experiment) | yes — `libfftw3-dev` |
| adaptivecpp | 13 / 6 | 13 / 6 | — | unknown: it stops wanting clang's own headers |

"Yes" means the configure accepts a riscv64 build given only those packages.
It does not mean the code compiles: the pseudobuild never runs the compiler
on the project.

**This runs the project's build scripts.** Everything else in this tool only
reads. Use it on repositories you trust, ideally in a container. It is
opt-in, time-bounded (`--pseudobuild-timeout` bounds the whole loop), confined to a
temporary directory that is deleted afterwards, and never invokes the compiler
on the project itself.

**Its compile checks are the host's.** There is no riscv64 compiler in the
loop: the host's compiler answers `check_include_file` and friends, against
the host's SDK. Checks that would need linking cannot be answered at all —
the Linux platform rules and the host's linker do not mix — so try-compiles
only compile, and a library a link check "found" is reported as unverified
rather than present. Headers the host lacks and any riscv64 Linux system has
are stubbed and listed as host gaps. If a project will not configure as a
cross build at all, it is configured for the host instead, and the report
says so.

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

```console
$ pip install will-it-riscv
```

## Use

```console
# the project in the current directory
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

# just the names of everything that is not pure Python
$ will-it-riscv -f list
```

### Exit codes

| code | meaning |
| --- | --- |
| 0 | everything installs from wheels or is pure Python |
| 1 | some packages must be built from source |
| 2 | something is blocked, unresolvable, or has no distribution at all |
| 3 | the index itself was unusable |

So `will-it-riscv || exit 1` is a reasonable CI gate.

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
| `debian:trixie` | Debian 13 — riscv64 is an official release architecture as of 13 (**default**) |
| `debian:sid` | Debian unstable |
| `debian:forky` | Debian 14 |
| `ubuntu:noble` | Ubuntu 24.04 LTS |
| `ubuntu:plucky` | Ubuntu 25.04 |
| `ubuntu:questing` | Ubuntu 25.10 |
| `ubuntu:resolute` | Ubuntu 26.04 LTS |

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

SCons and Bazel are detected but not scraped; the report says so rather than
implying the project has no dependencies.

**Repository scanning is inference too.** A CMake option you never enable is
indistinguishable, statically, from one you always do — so an optional
backend can appear in the list. Vendor GPU stacks (CUDA, ROCm, SYCL, oneAPI)
are filtered out entirely, since none of them exists for riscv64 and all are
opt-in.

**It is not a resolver.** For each package it takes the highest version satisfying
the constraints seen so far, and revisits when a later edge tightens them. It does
not backtrack. Where pip and uv disagree with it, they are right — but neither can
target riscv64 today, which is the whole reason this exists.

**Guessed package names are marked.** A library not in the curated map gets a
Debian name guessed from convention (`libfoo-dev`). The report says which ones
those are. Do not paste them into a provisioning script unchecked.

## Cache

Index pages, wheel metadata, source distributions and distro package lists are
cached under your platform cache directory (override with `WILL_IT_RISCV_CACHE`).
`will-it-riscv --clear-cache` empties it.

## License

MIT
