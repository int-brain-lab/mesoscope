"""
Shared plotting conventions for this subfolder.

Default for every figure made here:
- top/right spines removed
- at most 3 major ticks per numeric axis (categorical/labeled axes are
  exempt -- reducing to 3 ticks there would delete most of the labels)
- PETH / trial-type names typeset as LaTeX math (matching the paper's own
  $L_sL_cR_b$-style subscript notation), rotated 45 degrees, wherever they
  appear as axis tick labels
- Nature-journal figure standards: figure width snapped to a standard
  column width (single/1.5/double column), sans-serif font, small point
  sizes (axis text 6-7pt, panel letters 8pt bold, caption 8pt), applied via
  `apply_nature_style()` (called automatically on import)
- final deliverable is a single PDF with the caption rendered into it
  (compiled from LaTeX via `tectonic`), one bold panel letter per panel in
  the caption text (matching the paper's own Figure 3/4 caption style)
"""
import shutil
import subprocess
from pathlib import Path
import matplotlib as mpl
import matplotlib.ticker as mticker

# ---------------------------------------------------------------------
# Nature journal figure-width standards (mm -> inches)
# https://www.nature.com/documents/natrevmicro-artwork-guide.pdf and
# similar Springer Nature "Preparing figures" guides:
#   single column = 89 mm, 1.5 column = 136 mm, double column = 183 mm,
#   max whole-page height (incl. legend) = 247 mm.
# ---------------------------------------------------------------------
NATURE_SINGLE_COL_IN = 89 / 25.4    # ~3.50 in
NATURE_ONEHALF_COL_IN = 136 / 25.4  # ~5.35 in
NATURE_DOUBLE_COL_IN = 183 / 25.4   # ~7.20 in
NATURE_MAX_HEIGHT_IN = 247 / 25.4   # ~9.72 in


def apply_nature_style():
    """Sans-serif font, small point sizes, thin lines -- Nature house
    style for figure text (5-7pt body text, up to 8pt for panel labels)."""
    mpl.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
        'font.size': 7,
        'axes.labelsize': 7,
        'axes.titlesize': 7,
        'xtick.labelsize': 6,
        'ytick.labelsize': 6,
        'legend.fontsize': 6,
        'axes.linewidth': 0.6,
        'xtick.major.width': 0.6,
        'ytick.major.width': 0.6,
        'xtick.major.size': 2.5,
        'ytick.major.size': 2.5,
        'lines.linewidth': 1.0,
        'patch.linewidth': 0.6,
    })


apply_nature_style()

# ---------------------------------------------------------------------
# PETH / trial-type name -> neat LaTeX math label (no surrounding $ $),
# following the paper's own notation: subscript s = stimulus side,
# c = choice side, b = block side; trailing ", s"/", m" = stim-aligned vs.
# movement-aligned segment (see concat_PETHs' `tts` dict in dmn_bwm.py).
# ---------------------------------------------------------------------
PETH_LABEL = {
    'inter_trial':    r'\mathrm{ITI}',
    'blockL':         r'\mathrm{block}_L',
    'blockR':         r'\mathrm{block}_R',
    'quiescence':     r'\mathrm{quiesc.}',
    'block_change_s': r'\mathrm{change}_b,\,s',
    'stimLbLcL':      r'L_sL_cL_b,\,s',
    'stimLbRcL':      r'L_sL_cR_b,\,s',
    'stimRbRcR':      r'R_sR_cR_b,\,s',
    'stimRbLcR':      r'R_sR_cL_b,\,s',
    'mistake_s':      r'\mathrm{mistake},\,s',
    'motor_init':     r'\mathrm{move\ init}',
    'block_change_m': r'\mathrm{change}_b,\,m',
    'sLbLchoiceL':    r'L_sL_cL_b,\,m',
    'sLbRchoiceL':    r'L_sL_cR_b,\,m',
    'sRbRchoiceR':    r'R_sR_cR_b,\,m',
    'sRbLchoiceR':    r'R_sR_cL_b,\,m',
    'mistake_m':      r'\mathrm{mistake},\,m',
    'choiceL':        r'\mathrm{choice}_L',
    'choiceR':        r'\mathrm{choice}_R',
    'fback1':         r'\mathrm{fback}^{+}',
    'fback0':         r'\mathrm{fback}^{-}',
}


def peth_math(name: str) -> str:
    """Bare LaTeX math source for a PETH/trial-type name (no $ $)."""
    return PETH_LABEL.get(name, name.replace('_', r'\_'))


def peth_tex(name: str) -> str:
    """LaTeX math snippet for a PETH/trial-type name, incl. $ $."""
    return f'${peth_math(name)}$'


def set_top_peth_labels(ax2, ticks, names, fontsize=6, rotation=45):
    """Neatly typeset PETH/trial-type names on a (secondary, top) x-axis:
    LaTeX math notation, rotated 45 degrees, matching the other figures."""
    ax2.set_xticks(ticks)
    ax2.set_xticklabels([peth_tex(n) for n in names], rotation=rotation,
                         ha='left', va='bottom', rotation_mode='anchor',
                         fontsize=fontsize)


def style_axes(ax, max_xticks=3, max_yticks=3, categorical_x=False,
                categorical_y=False):
    """Strip top/right spines; cap tick count on numeric axes."""
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    if not categorical_x:
        ax.xaxis.set_major_locator(mticker.MaxNLocator(nbins=max_xticks))
    if not categorical_y:
        ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=max_yticks))


def panel_label(ax, letter, x=-0.12, y=1.05, fontsize=8):
    """Bold lowercase panel letter, Nature style (8pt bold, top-left)."""
    ax.text(x, y, letter, transform=ax.transAxes, fontsize=fontsize,
            fontweight='bold', va='bottom', ha='right')


def save_with_caption(fig, out_stem: Path, panel_captions: list[str],
                       label: str, intro: str = ''):
    """
    Produce the final deliverable for one figure:
      - `<out_stem>_panels.pdf`: the raw matplotlib panels (no caption)
      - `<out_stem>.tex`: standalone LaTeX source (image + rendered
        caption, Nature-style sans-serif 8pt caption text), with a
        `figure` block that can be copied straight into a manuscript
      - `<out_stem>.pdf`: that source compiled with `tectonic` -- image
        and LaTeX-typeset caption together in one PDF. This is the
        deliverable to look at / share.

    panel_captions: list of strings, one per panel, in order a, b, c, ...
    Falls back to leaving only the .tex (uncompiled) if `tectonic` is not
    on PATH.
    """
    out_stem = Path(out_stem)
    panels_pdf = out_stem.parent / f'{out_stem.name}_panels.pdf'
    tex_path = out_stem.with_suffix('.tex')
    final_pdf = out_stem.with_suffix('.pdf')

    fig.savefig(panels_pdf, bbox_inches='tight')
    fig_width_in = fig.get_size_inches()[0]

    letters = [chr(ord('a') + i) for i in range(len(panel_captions))]
    body = ' '.join(f'\\textbf{{{L}}} {c}' for L, c in zip(letters, panel_captions))
    caption = (intro + ' ' + body).strip() if intro else body

    tex = (
        "\\documentclass[border=4pt]{standalone}\n"
        "\\usepackage[T1]{fontenc}\n"
        "\\usepackage{helvet}\n"
        "\\renewcommand{\\familydefault}{\\sfdefault}\n"
        "\\usepackage{graphicx}\n"
        "\\usepackage{caption}\n"
        "\\usepackage{amsmath}\n"
        "\\captionsetup{font={sf,footnotesize},labelfont={sf,bf,footnotesize},"
        "justification=justified,singlelinecheck=false}\n"
        "\\begin{document}\n"
        f"\\begin{{minipage}}{{{fig_width_in:.2f}in}}\n"
        "\\centering\n"
        "\\begin{figure}[htbp]\n"
        "    \\centering\n"
        f"    \\includegraphics[width=\\linewidth]{{{panels_pdf.name}}}\n"
        f"    \\caption{{{caption}}}\n"
        f"    \\label{{fig:{label}}}\n"
        "\\end{figure}\n"
        "\\end{minipage}\n"
        "\\end{document}\n"
    )
    tex_path.write_text(tex)

    if shutil.which('tectonic') is None:
        print('[plotstyle] tectonic not on PATH; wrote .tex but did not '
              'compile the captioned PDF.')
        return panels_pdf, tex_path, None

    try:
        subprocess.run(
            ['tectonic', tex_path.name],
            cwd=str(out_stem.parent), check=True,
            capture_output=True, text=True,
        )
    except subprocess.CalledProcessError as e:
        print('[plotstyle] tectonic compile failed:\n', e.stdout, e.stderr)
        return panels_pdf, tex_path, None

    return panels_pdf, tex_path, final_pdf
