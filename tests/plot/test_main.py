import subprocess
import sys

from plot.__main__ import main

OUTPUTS = {
    "fig1.pdf",
    "fig1.png",
    "fig2.pdf",
    "fig2.png",
    "fig2-R1-1h.pdf",
    "fig2-R1-1h.png",
    "fig3.pdf",
    "fig3.png",
    "fig3-R1.pdf",
    "fig3-R1.png",
    "fig4.pdf",
    "fig4.png",
    "fig5.pdf",
    "fig5.png",
    "table1.csv",
    "table1.tex",
    "table2.csv",
    "table2.tex",
    "table3.csv",
    "table3.tex",
    "table4.csv",
    "table4.tex",
    "cost.txt",
    "methods.txt",
    "captions.txt",
}


def test_main_writes_every_output(results_dir, tmp_path):
    out = tmp_path / "out"
    assert main(["--results", str(results_dir), "--out", str(out)]) == 0
    assert {p.name for p in out.iterdir()} == OUTPUTS
    assert all(p.stat().st_size > 0 for p in out.iterdir())


def test_main_captions_lists_figures_before_tables(results_dir, tmp_path):
    out = tmp_path / "out"
    main(["--results", str(results_dir), "--out", str(out)])
    captions = (out / "captions.txt").read_text()
    labels = [
        paragraph.split(".")[0] for paragraph in captions.strip().split("\n\n")
    ]
    assert labels == [
        "Figure 1",
        "Figure 2",
        "Figure 2",
        "Figure 3",
        "Figure 3",
        "Figure 4",
        "Figure 5",
        "Table 1",
        "Table 2",
        "Table 3",
        "Table 4",
    ]
    assert captions.isascii()


def test_main_uses_the_recording_option(results_dir, tmp_path):
    out = tmp_path / "out"
    main(
        ["--results", str(results_dir), "--out", str(out), "--recording", "R1"]
    )
    names = {p.name for p in out.iterdir()}
    assert "fig3-R3.pdf" in names
    assert "fig2-R1-1h.pdf" in names
    assert (
        "on R1"
        in (out / "table3.csv").parent.joinpath("captions.txt").read_text()
    )


def test_main_uses_the_format_recording_option(results_dir, tmp_path):
    out = tmp_path / "out"
    main(
        [
            "--results",
            str(results_dir),
            "--out",
            str(out),
            "--format-recording",
            "R1-1h",
        ]
    )
    names = {p.name for p in out.iterdir()}
    assert "fig2-R3-1h.pdf" in names
    assert "fig2-R1-1h.pdf" not in names
    assert "fig3-R1.pdf" in names


def test_main_skips_figure_2_for_a_format_recording_that_is_absent(
    results_dir, tmp_path, capsys
):
    out = tmp_path / "out"
    main(
        [
            "--results",
            str(results_dir),
            "--out",
            str(out),
            "--format-recording",
            "R4",
        ]
    )
    assert "skipped Figure 2" in capsys.readouterr().err
    assert "fig2.pdf" not in {p.name for p in out.iterdir()}


def test_main_skips_outputs_whose_csv_is_absent(results_dir, tmp_path, capsys):
    (results_dir / "format-reads.csv").unlink()
    (results_dir / "resampling-requests.csv").unlink()
    out = tmp_path / "out"
    assert main(["--results", str(results_dir), "--out", str(out)]) == 0
    names = {p.name for p in out.iterdir()}
    assert "fig1.pdf" not in names
    assert "fig3.pdf" in names
    assert "cost.txt" not in names
    assert "format-reads.csv: file not found" in capsys.readouterr().err


def test_main_skips_a_recording_that_is_absent(results_dir, tmp_path, capsys):
    out = tmp_path / "out"
    main(
        ["--results", str(results_dir), "--out", str(out), "--recording", "R4"]
    )
    assert "skipped Figure 3" in capsys.readouterr().err
    assert "fig3.pdf" not in {p.name for p in out.iterdir()}


def test_main_returns_1_and_writes_only_table_1_without_any_csv(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    out = tmp_path / "out"
    assert main(["--results", str(empty), "--out", str(out)]) == 1
    assert {p.name for p in out.iterdir()} == {
        "table1.csv",
        "table1.tex",
        "captions.txt",
    }


def test_module_runs_as_a_command(results_dir, tmp_path):
    out = tmp_path / "out"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "plot",
            "--results",
            str(results_dir),
            "--out",
            str(out),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert (out / "fig1.png").exists()
