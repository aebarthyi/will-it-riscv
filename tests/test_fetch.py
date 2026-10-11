import hashlib
import io
import tarfile
from pathlib import Path

import httpx
import pytest

from will_it_riscv import fetch, serve

BODY = bytes(range(256)) * 64


def item(body=BODY, url="https://example.org/files/planner.gguf"):
    return fetch.Download(url, hashlib.sha256(body).hexdigest(), len(body))


def client(body=BODY, ranges=True, seen=None):
    def respond(request):
        if seen is not None:
            seen.append(request.headers.get("range"))
        asked = request.headers.get("range")
        if asked and ranges:
            start = int(asked.removeprefix("bytes=").rstrip("-"))
            return httpx.Response(206, content=body[start:])
        return httpx.Response(200, content=body)
    return httpx.Client(transport=httpx.MockTransport(respond))


def test_a_download_is_checked_and_put_in_place(tmp_path):
    dest = tmp_path / "models" / "planner.gguf"
    assert fetch.download(item(), dest, client=client()) == dest
    assert dest.read_bytes() == BODY and not (tmp_path / "models" / "planner.gguf.part").exists()
    seen = []
    fetch.download(item(), dest, client=client(seen=seen))
    assert seen == []   # there already: nothing fetched


def test_a_download_that_is_not_the_file_is_thrown_away(tmp_path):
    dest = tmp_path / "planner.gguf"
    with pytest.raises(RuntimeError, match="SHA-256 differs"):
        fetch.download(item(), dest, client=client(body=bytes(len(BODY))))
    assert not dest.exists() and not (tmp_path / "planner.gguf.part").exists()


def test_a_stopped_download_picks_up_where_it_stopped(tmp_path):
    dest = tmp_path / "planner.gguf"
    (tmp_path / "planner.gguf.part").write_bytes(BODY[:5000])
    seen = []
    fetch.download(item(), dest, client=client(seen=seen))
    assert seen == ["bytes=5000-"] and dest.read_bytes() == BODY


def test_a_server_that_will_not_resume_sends_it_all_again(tmp_path):
    dest = tmp_path / "planner.gguf"
    (tmp_path / "planner.gguf.part").write_bytes(b"x" * 5000)
    fetch.download(item(), dest, client=client(ranges=False))
    assert dest.read_bytes() == BODY


def test_a_download_that_fails_says_so_and_keeps_what_it_has(tmp_path):
    def broken(request):
        raise httpx.ConnectError("no route", request=request)
    with pytest.raises(RuntimeError, match="run again"):
        fetch.download(item(), tmp_path / "planner.gguf",
                       client=httpx.Client(transport=httpx.MockTransport(broken)))


def test_the_prebuilt_server_suits_the_machine():
    build, archive = fetch.server_build("Linux", "x86_64", vulkan=True)
    assert build == "vulkan" and archive.name.endswith("ubuntu-vulkan-x64.tar.gz")
    build, archive = fetch.server_build("Linux", "x86_64", vulkan=False)
    assert build == "cpu" and archive.name.endswith("ubuntu-x64.tar.gz")
    assert fetch.server_build("Linux", "arm64", vulkan=False)[1].name.endswith("ubuntu-arm64.tar.gz")
    assert fetch.server_build("Darwin", "arm64", vulkan=False)[1].name.endswith("macos-arm64.tar.gz")
    assert fetch.server_build("Linux", "riscv64", vulkan=True) == ("cpu", None)
    for archive in fetch.SERVERS.values():
        assert fetch.LLAMA_CPP in archive.url and len(archive.sha256) == 64


def release(members):
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
        for name, data, link in members:
            info = tarfile.TarInfo(name)
            if link:
                info.type, info.linkname = tarfile.SYMTYPE, link
            else:
                info.size, info.mode = len(data), 0o755
            tar.addfile(info, io.BytesIO(data) if data else None)
    return buffer.getvalue()


def test_the_server_is_fetched_once_and_unpacked(tmp_path, monkeypatch):
    archive = release([("llama-b1/llama-server", b"#!/bin/sh\n", None),
                       ("llama-b1/libggml.so.0", b"lib", None),
                       ("llama-b1/libggml.so", None, "libggml.so.0")])
    monkeypatch.setenv("WILL_IT_RISCV_DATA", str(tmp_path))
    monkeypatch.setattr(serve, "found", lambda: None)
    monkeypatch.setattr(fetch, "server_build", lambda: ("cpu", item(archive, "https://x/b1.tar.gz")))
    server = fetch.llama_server(client=client(archive))
    assert server == str(tmp_path / "llama.cpp" / f"{fetch.LLAMA_CPP}-cpu" / "llama-server")
    assert Path(server).is_file() and (Path(server).parent / "libggml.so").resolve().name == "libggml.so.0"
    assert not (tmp_path / "llama.cpp" / "b1.tar.gz").exists()
    seen = []
    assert fetch.llama_server(client=client(archive, seen=seen)) == server and seen == []
    monkeypatch.setattr(serve, "found", lambda: "/mine/llama-server")
    assert fetch.llama_server() == "/mine/llama-server"


def test_an_archive_that_reaches_outside_is_refused(tmp_path):
    archive = tmp_path / "bad.tar.gz"
    archive.write_bytes(release([("llama-b1/llama-server", b"x", None),
                                 ("llama-b1/../../escape", b"x", None)]))
    (tmp_path / "llama.cpp").mkdir()
    with pytest.raises(RuntimeError):
        fetch._unpack(archive, tmp_path / "llama.cpp" / "b1-cpu")
    assert not (tmp_path / "escape").exists()


def test_no_prebuilt_server_says_what_to_do(monkeypatch):
    monkeypatch.setattr(serve, "found", lambda: None)
    monkeypatch.setattr(fetch, "server_build", lambda: ("cpu", None))
    with pytest.raises(RuntimeError, match="build llama.cpp"):
        fetch.llama_server()


def test_an_interrupted_download_keeps_what_it_has_and_says_so(tmp_path):
    from rich.console import Console

    body = BODY * 128   # 2 MB: the download writes whole 1 MB chunks

    def chunks():
        yield body[:1 << 20]
        raise KeyboardInterrupt
    out = io.StringIO()
    stopped = httpx.Client(transport=httpx.MockTransport(
        lambda request: httpx.Response(200, content=chunks())))
    with pytest.raises(KeyboardInterrupt):
        fetch.download(item(body), tmp_path / "planner.gguf", Console(file=out), stopped)
    assert (tmp_path / "planner.gguf.part").read_bytes() == body[:1 << 20]
    assert "the next run picks up from there" in out.getvalue()
    seen = []
    done = fetch.download(item(body), tmp_path / "planner.gguf", client=client(body, seen=seen))
    assert seen == [f"bytes={1 << 20}-"] and done.read_bytes() == body


def test_the_planner_goes_in_the_data_directory(tmp_path, monkeypatch):
    monkeypatch.setenv("WILL_IT_RISCV_DATA", str(tmp_path))
    monkeypatch.setattr(fetch, "PLANNER", item())
    assert fetch.planner(client=client()) == tmp_path / "models" / "planner.gguf"


def test_plan_is_plan_from_model_with_only_what_it_takes(tmp_path, monkeypatch):
    from will_it_riscv import cli

    args = cli._plan_arguments(["src/proj", "--model", "x.gguf", "-f", "dot"])
    assert args.plan_from_model and args.path == "src/proj" and args.planner_model == "x.gguf"
    assert args.distro == "ubuntu:resolute" and args.format == "dot"
    assert args.workers == 8 and not args.recurse   # the rest of main's defaults
    with pytest.raises(SystemExit):
        cli._plan_arguments(["--pseudobuild"])   # not one of plan's


def test_doctor_says_what_is_missing(tmp_path, monkeypatch, capsys):
    import shutil

    from will_it_riscv import cli, doctor

    monkeypatch.setenv("WILL_IT_RISCV_DATA", str(tmp_path))
    monkeypatch.delenv("LLAMA_SERVER", raising=False)
    real = shutil.which
    monkeypatch.setattr(shutil, "which", lambda name, *a, **k: None if name in (
        "cc", "gcc", "clang", "gfortran", "llama-server") else real(name, *a, **k))
    found = {c.name: c for c in doctor.checks()}
    assert found["C compiler"].status == doctor.FAIL and "build-essential" in found["C compiler"].detail
    assert found["Fortran compiler"].status == doctor.WARN
    assert found["planner"].status == doctor.OK and "fetched on the first plan" in found["planner"].detail
    assert cli.main(["doctor"]) == 1
    assert "cannot do without" in capsys.readouterr().out
