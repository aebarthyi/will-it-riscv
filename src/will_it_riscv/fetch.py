"""What a run fetches the first time it needs it: the planner, and a server for it.

``will-it-riscv plan`` has its plan written by the fine-tuned planner, one
GGUF file, which llama.cpp's server runs. Neither comes with the package --
the model is 4.3 GB -- so the first run downloads both into the data
directory (``$WILL_IT_RISCV_DATA``, else the platform's:
~/.local/share/will-it-riscv on Linux), checks each against the SHA-256
pinned here, and every run after that uses what is there. ``--clear-cache``
leaves them alone; they are not a cache.

The server is llama.cpp's own prebuilt release, pinned. On Linux it is the
Vulkan build wherever there is a Vulkan loader -- it runs on AMD, NVIDIA and
Intel GPUs alike, and on the CPU when it finds no GPU -- and the CPU build
where there is none. A llama-server of your own wins: ``$LLAMA_SERVER``,
then one on the PATH.
"""

from __future__ import annotations

import contextlib
import ctypes.util
import hashlib
import os
import platform
import shutil
import tarfile
import tempfile
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from platformdirs import user_data_dir


@dataclass(frozen=True)
class Download:
    url: str
    sha256: str
    size: int

    @property
    def name(self) -> str:
        return self.url.rsplit("/", 1)[-1]


#: Where the planner is published.
PLANNER_REPO = "aebarthyi/lagutenko-4B"
#: The repository commit the file is fetched from, so that later weights
#: pushed to the repository leave this release's download as it was.
PLANNER_REVISION = "6dd533b2854316597e079c5c03aaca2cf98dfdd9"
#: The fine-tuned planner, Lagutenko-4B: Qwen3-4B-Instruct-2507 tuned on the
#: teacher's plans (checkpoint 318), q8_0 -- the file the held-out bench scored.
PLANNER = Download(
    f"https://huggingface.co/{PLANNER_REPO}/resolve/{PLANNER_REVISION}/lagutenko-4B-Q8_0.gguf",
    "9bf58f89a5cd03449b348ae2dc68954f6f74010e7136f70497c5067d6a7d692c",
    4280403456,
)

#: The llama.cpp release the server comes from.
LLAMA_CPP = "b11191"
_RELEASE = (f"https://github.com/ggml-org/llama.cpp/releases/download/{LLAMA_CPP}/"
            f"llama-{LLAMA_CPP}-bin-")
#: (platform.system(), platform.machine(), build) -> that release's archive.
SERVERS = {
    ("Linux", "x86_64", "vulkan"): Download(
        _RELEASE + "ubuntu-vulkan-x64.tar.gz",
        "0e2018de56a2e23dcdf71c5a4f7ee01889820c29bcb7a7d5db9318bc7c2c6251", 30956068),
    ("Linux", "x86_64", "cpu"): Download(
        _RELEASE + "ubuntu-x64.tar.gz",
        "ecf99c49b7626fed1d699e789e31d7cd5a20ddd08222b0fdc65f61208f20e3bc", 17015522),
    ("Linux", "aarch64", "vulkan"): Download(
        _RELEASE + "ubuntu-vulkan-arm64.tar.gz",
        "0245bbc0d71631f279824f78639ad20d5bd1f2f8e4a997c0ec6882ace926caf1", 24775871),
    ("Linux", "aarch64", "cpu"): Download(
        _RELEASE + "ubuntu-arm64.tar.gz",
        "ce12531c1e1f3ab2a8fa3305731a103de792449a420698d74253c56e73bcb34f", 13605391),
    # Metal is in the arm64 build.
    ("Darwin", "arm64", "cpu"): Download(
        _RELEASE + "macos-arm64.tar.gz",
        "18a5342183b0f4bad7290195b64b4efdcb577917d187ff5955127f2117afe77b", 11766532),
    ("Darwin", "x86_64", "cpu"): Download(
        _RELEASE + "macos-x64.tar.gz",
        "93a3fa704d37177c34fe69245b86d58337802331e57b6c37a462f2ddec3b771a", 11249545),
}


def home() -> Path:
    env = os.environ.get("WILL_IT_RISCV_DATA")
    if env:
        return Path(env).expanduser()
    return Path(user_data_dir("will-it-riscv", "tactcomplabs"))


def planner_path() -> Path:
    return home() / "models" / PLANNER.name


def planner(console=None, client=None) -> Path:
    """The fine-tuned planner's GGUF, downloaded the first time."""
    dest = planner_path()
    if not dest.is_file() and console is not None:
        console.print(f"fetching the planner, {PLANNER.size / 1e9:.1f} GB, once, "
                      f"into {dest.parent}")
    return download(PLANNER, dest, console, client)


def server_build(system: Optional[str] = None, machine: Optional[str] = None,
                 vulkan: Optional[bool] = None) -> tuple[str, Optional[Download]]:
    """Which prebuilt server suits this machine: (build, its archive or None)."""
    system = system or platform.system()
    machine = machine or platform.machine()
    machine = {"AMD64": "x86_64", "amd64": "x86_64"}.get(machine, machine)
    if system == "Linux" and machine == "arm64":
        machine = "aarch64"
    elif system == "Darwin" and machine == "aarch64":
        machine = "arm64"
    if vulkan is None:
        vulkan = system == "Linux" and ctypes.util.find_library("vulkan") is not None
    build = "vulkan" if vulkan and (system, machine, "vulkan") in SERVERS else "cpu"
    return build, SERVERS.get((system, machine, build))


def llama_server(console=None, client=None) -> str:
    """llama.cpp's server: one of your own, else the pinned release, fetched once."""
    from . import serve

    found = serve.found()
    if found:
        return found
    build, item = server_build()
    if item is None:
        raise RuntimeError(
            f"no prebuilt llama-server for {platform.system()} {platform.machine()}: build "
            "llama.cpp, then put llama-server on the PATH or set LLAMA_SERVER to it"
        )
    where = home() / "llama.cpp" / f"{LLAMA_CPP}-{build}"
    server = where / "llama-server"
    if server.is_file():
        return str(server)
    if console is not None:
        console.print(f"fetching llama.cpp's server ({LLAMA_CPP}, {build}), once, "
                      f"into {where.parent}")
    archive = download(item, where.parent / item.name, console, client)
    _unpack(archive, where)
    archive.unlink()
    if not server.is_file():
        raise RuntimeError(f"{item.name} has no llama-server")
    return str(server)


def download(item: Download, dest: Path, console=None, client=None) -> Path:
    """``item`` at ``dest``, fetched unless it is there, checked against its SHA-256.

    A download that stops part way picks up where it stopped the next time:
    4 GB is a lot to start again.
    """
    import httpx

    if dest.is_file() and dest.stat().st_size == item.size:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    have = part.stat().st_size if part.is_file() else 0
    if have > item.size:
        part.unlink()
        have = 0
    if have < item.size:
        own = client is None
        if own:
            client = httpx.Client(follow_redirects=True, timeout=httpx.Timeout(30, read=120))
        try:
            headers = {"Range": f"bytes={have}-"} if have else {}
            with client.stream("GET", item.url, headers=headers) as response:
                if response.status_code == 200:
                    have = 0   # the whole file, whether or not a part was asked for
                elif response.status_code != 206:
                    raise RuntimeError(f"{item.url}: HTTP {response.status_code}")
                with open(part, "ab" if have else "wb") as out, \
                        _progress(console, item.name, item.size, have) as advance:
                    for chunk in response.iter_bytes(1 << 20):
                        out.write(chunk)
                        advance(len(chunk))
        except httpx.HTTPError as exc:
            raise RuntimeError(f"fetching {item.name} stopped ({exc}); run again to "
                               "pick up where it stopped") from exc
        except KeyboardInterrupt:
            if console is not None:
                console.print(f"stopped at {part.stat().st_size / 1e9:.1f} of "
                              f"{item.size / 1e9:.1f} GB; the next run picks up from there")
            raise
        finally:
            if own:
                client.close()
    if _sha256(part) != item.sha256:
        part.unlink()
        raise RuntimeError(f"{item.name} is not the file it should be (its SHA-256 differs); "
                           "run again to fetch it afresh")
    part.replace(dest)
    return dest


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


@contextlib.contextmanager
def _progress(console, label: str, total: int, done: int) -> Iterator[Callable[[int], None]]:
    if console is None or console.quiet:
        yield lambda _: None
        return
    from rich.progress import (
        BarColumn,
        DownloadColumn,
        Progress,
        TextColumn,
        TimeRemainingColumn,
        TransferSpeedColumn,
    )

    with Progress(TextColumn("{task.description}"), BarColumn(), DownloadColumn(),
                  TransferSpeedColumn(), TimeRemainingColumn(), console=console,
                  transient=True) as progress:
        task = progress.add_task(label, total=total, completed=done)
        yield lambda size: progress.advance(task, size)


def _unpack(archive: Path, where: Path) -> None:
    """The release's one top directory, unpacked as ``where``.

    Nothing in it may land outside, or link outside: the data filter where
    tarfile has one, the same checks by hand where it does not (3.9 before
    3.9.17).
    """
    scratch = Path(tempfile.mkdtemp(prefix=".unpack-", dir=where.parent))
    try:
        with tarfile.open(archive) as tar:
            if hasattr(tarfile, "data_filter"):
                try:
                    tar.extractall(scratch, filter="data")
                except tarfile.TarError as exc:
                    raise RuntimeError(f"{archive.name}: {exc}") from exc
            else:
                inside = str(scratch.resolve()) + os.sep
                for member in tar.getmembers():
                    target = (scratch / member.name).resolve()
                    link = target
                    if member.issym():
                        link = (target.parent / member.linkname).resolve()
                    elif member.islnk():
                        link = (scratch / member.linkname).resolve()
                    if not (str(target).startswith(inside) and str(link).startswith(inside)) \
                            or member.isdev():
                        raise RuntimeError(f"{archive.name}: {member.name} reaches outside")
                tar.extractall(scratch)
        tops = list(scratch.iterdir())
        if len(tops) != 1 or not tops[0].is_dir():
            raise RuntimeError(f"{archive.name}: not one directory")
        if where.exists():
            shutil.rmtree(where)
        tops[0].replace(where)
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
