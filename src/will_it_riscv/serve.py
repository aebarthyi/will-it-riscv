"""A local model, served by llama.cpp for as long as one run needs it.

A fine-tuned planner is one GGUF file (``training/export.py`` writes it).
``llama_server`` starts llama.cpp's server on it, on a free local port,
waits until the model is loaded, and stops the server when the block ends --
so ``--planner llama-server --planner-model planner.gguf`` needs nothing
running beforehand.

The server is found as ``$LLAMA_SERVER``, else ``llama-server`` on the PATH,
else it is given (``fetch.llama_server`` downloads llama.cpp's own build).
It runs with what the planner was evaluated with: a 40k context (the pack,
then up to three answers and what was sent back about them), every layer on
the GPU, flash attention, one slot, the model's own chat template.
``$LLAMA_SERVER_ARGS`` is appended, so ``LLAMA_SERVER_ARGS='-ngl 0'`` runs on
the CPU alone.
"""

from __future__ import annotations

import contextlib
import os
import shlex
import shutil
import socket
import subprocess
import tempfile
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Optional

#: What the planner was evaluated with.
CONTEXT = 40960
ARGS = ("-ngl", "99", "-fa", "on", "-np", "1", "--jinja")


def found() -> Optional[str]:
    """A llama-server of your own: ``$LLAMA_SERVER``, else one on the PATH."""
    return os.environ.get("LLAMA_SERVER") or shutil.which("llama-server")


def binary() -> str:
    server = found()
    if not server:
        raise RuntimeError(
            "no llama-server: build llama.cpp, then put llama-server on the PATH "
            "or set LLAMA_SERVER to it"
        )
    return server


def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def command(model: Path, port: int, server: Optional[str] = None) -> list[str]:
    return [
        server or binary(), "-m", str(model), "-a", model.stem,
        "--host", "127.0.0.1", "--port", str(port), "--ctx-size", str(CONTEXT),
        *ARGS, *shlex.split(os.environ.get("LLAMA_SERVER_ARGS", "")),
    ]


@contextlib.contextmanager
def llama_server(model: Path, ready_timeout: float = 600.0,
                 server: Optional[str] = None) -> Iterator[str]:
    """Serve ``model`` (with ``server``, else ``binary()``); yields the base URL."""
    import httpx

    model = Path(model)
    if not model.is_file():
        raise RuntimeError(f"{model}: no such model file")
    port = _free_port()
    with tempfile.TemporaryFile("w+") as log:
        process = subprocess.Popen(
            command(model, port, server), stdin=subprocess.DEVNULL, stdout=log,
            stderr=subprocess.STDOUT,
        )
        try:
            deadline = time.monotonic() + ready_timeout
            while True:
                if process.poll() is not None:
                    log.seek(0)
                    tail = "".join(log.readlines()[-15:]).rstrip()
                    raise RuntimeError(
                        f"llama-server stopped (exit {process.returncode}) "
                        f"before {model.name} was loaded:\n{tail}"
                    )
                try:
                    if httpx.get(f"http://127.0.0.1:{port}/health", timeout=5).status_code == 200:
                        break
                except httpx.HTTPError:
                    pass   # not listening yet
                if time.monotonic() > deadline:
                    raise RuntimeError(
                        f"llama-server did not load {model.name} in {ready_timeout:.0f} s"
                    )
                time.sleep(1)
            yield f"http://127.0.0.1:{port}/v1"
        finally:
            process.terminate()
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
