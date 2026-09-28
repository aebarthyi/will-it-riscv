"""The evidence pack is the same on every filesystem."""

from will_it_riscv import evidence


def test_names_are_found_whatever_their_case_and_cited_as_spelled(tmp_path):
    (tmp_path / "makefile").write_text("all:\n\tcc -o hello hello.c -lz\n")
    (tmp_path / "Docs").mkdir()
    (tmp_path / "Docs" / "install.md").write_text(
        "# Installing\n\n## Building\n\n```bash\nmake\n```\n")
    pack = evidence.build(tmp_path, "hello")
    shown = {e.path for e in pack.excerpts}
    assert "makefile" in shown
    assert "Docs/install.md" in shown
    assert not shown & {"Makefile", "docs/install.md"}
