#!/usr/bin/env python3
"""Assemble the LaTeX report in report/ from the figures in output/.

Each main figure gets its own folder report/figures/<name>/ holding the
panels PDF and figure.tex (the figure environment with the caption written
by maps.py / controls.py); all supplementary figures go to report/si/.
report/main.tex inputs these files. Compiles main.pdf with tectonic.
"""
import re
import shutil
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "output"
REPORT = HERE / "report"

MAIN = [  # (folder, output stem)
    ("fig1_windows", "dorsal_windows"),
    ("fig2_coverage", "dorsal_counts"),
    ("fig3_tuning", "dorsal_tuning_unfolded"),
    ("fig4_decoding_n10", "dorsal_decoding_n10"),
    ("fig5_decoding_n10_scatter", "dorsal_decoding_n10_scatter"),
    ("fig6_decoding_all", "dorsal_decoding_all"),
    ("fig7_decoding_all_scatter", "dorsal_decoding_all_scatter"),
]
SI = [
    "dorsal_tuning",            # S1 folded tuning maps
    "si_tuning_sessions",       # S2
    "si_tuning_variance",       # S3
    "si_tuning_corrected",      # S4
    "si_tuning_strength",       # S5
    "si_hex1mm_counts",         # S6
    "si_hex1mm_tuning",         # S7
    "si_hex1mm_decoding",       # S8
    "si_windows",               # S9
    "si_layers",                # S10
]


def caption_of(stem):
    tex = (OUT / f"{stem}.tex").read_text()
    m = re.search(r"\\caption\{(.*)\}\s*\n\s*\\label", tex, re.S)
    return m.group(1)


def width_of(stem):
    tex = (OUT / f"{stem}.tex").read_text()
    return float(re.search(r"\\begin\{minipage\}\{([\d.]+)in\}", tex).group(1))


def figure_tex(stem, path_prefix, label):
    w = min(width_of(stem), 7.0)
    return (f"\\begin{{figure}}[p]\n  \\centering\n"
            f"  \\includegraphics[width={w:.2f}in,height=0.78\\textheight,keepaspectratio]{{{path_prefix}{stem}_panels.pdf}}\n"
            f"  \\caption{{{caption_of(stem)}}}\n  \\label{{{label}}}\n\\end{{figure}}\n")


def build():
    (REPORT / "figures").mkdir(parents=True, exist_ok=True)
    (REPORT / "si").mkdir(exist_ok=True)
    for folder, stem in MAIN:
        d = REPORT / "figures" / folder
        d.mkdir(exist_ok=True)
        shutil.copy(OUT / f"{stem}_panels.pdf", d)
        (d / "figure.tex").write_text(figure_tex(stem, f"figures/{folder}/", f"fig:{folder}"))
    for stem in SI:
        if not (OUT / f"{stem}_panels.pdf").exists():
            print("missing", stem)
            continue
        shutil.copy(OUT / f"{stem}_panels.pdf", REPORT / "si")
        (REPORT / "si" / f"{stem}.tex").write_text(figure_tex(stem, "si/", f"fig:{stem}"))
    subprocess.run(["tectonic", "-X", "compile", "main.tex"], cwd=REPORT, check=True)


if __name__ == "__main__":
    build()
