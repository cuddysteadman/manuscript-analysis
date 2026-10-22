from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

import matplotlib as mpl


PAPER_STYLE = {
    "figure.facecolor": "#ffffff",
    "figure.edgecolor": "#ffffff",
    "figure.dpi": 100,
    "savefig.facecolor": "#ffffff",
    "savefig.edgecolor": "#ffffff",
    "savefig.transparent": False,
    "savefig.dpi": 600,
    "savefig.bbox": "tight",
    "axes.facecolor": "#ffffff",
    "axes.edgecolor": "#000000",
    "axes.labelcolor": "#000000",
    "axes.titlecolor": "#000000",
    "axes.grid": False,
    "axes.axisbelow": False,
    "text.color": "#000000",
    "font.family": "Arial",
    "font.size": 9,
    "axes.labelsize": 10,
    "axes.titlesize": 10,
    "axes.titleweight": "normal",
    "xtick.color": "#000000",
    "ytick.color": "#000000",
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.major.size": 3.5,
    "ytick.major.size": 3.5,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "legend.facecolor": "#ffffff",
    "legend.edgecolor": "#000000",
    "legend.framealpha": 1.0,
    "legend.labelcolor": "#000000",
    "legend.frameon": True,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}


def apply_paper_style() -> None:
    mpl.rcParams.update(PAPER_STYLE)


@contextmanager
def paper_style_context() -> Iterator[None]:
    style = {**mpl.rcParamsDefault, **PAPER_STYLE}
    with mpl.rc_context(style):
        yield
