# will-it-riscv

Walk a Python project's dependency tree and find out what will **not** install on
riscv64 without a compiler — before you get there.

```console
$ will-it-riscv pyproject.toml
```

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
