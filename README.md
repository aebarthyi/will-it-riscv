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
