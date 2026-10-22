from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.table import Table
from matplotlib.text import Text


EXPORT_FORMATS = ("png", "svg", "pdf")
LABEL_VARIANTS = (("with_labels", True), ("without_labels", False))


def save_formats(figure: Figure, output_dir: Path, stem: str, **savefig_kwargs: Any) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = [output_dir / f"{stem}.{file_format}" for file_format in EXPORT_FORMATS]
    for path in paths:
        figure.savefig(path, **savefig_kwargs)
    return paths


def save_label_variants(
    figure: Figure,
    output_dir: Path,
    stem: str,
    **savefig_kwargs: Any,
) -> list[Path]:
    paths = save_formats(figure, output_dir, f"{stem}_with_labels", **savefig_kwargs)

    # ``Table`` does not expose its cell text through the normal artist tree,
    # so include those labels explicitly as well.
    text_artists = list(figure.findobj(match=Text))
    for table in figure.findobj(match=Table):
        text_artists.extend(cell.get_text() for cell in table.get_celld().values())
    text_artists = list(dict.fromkeys(text_artists))
    original_visibility = [(artist, artist.get_visible()) for artist in text_artists]
    try:
        for artist, _ in original_visibility:
            artist.set_visible(False)
        paths += save_formats(figure, output_dir, f"{stem}_without_labels", **savefig_kwargs)
    finally:
        for artist, visible in original_visibility:
            artist.set_visible(visible)
    return paths


def render_label_variants(
    make_figure: Callable[[bool], Figure],
    output_dir: Path,
    stem: str,
    **savefig_kwargs: Any,
) -> list[Path]:
    paths = []
    for suffix, include_labels in LABEL_VARIANTS:
        figure = make_figure(include_labels)
        paths += save_formats(figure, output_dir, f"{stem}_{suffix}", **savefig_kwargs)
        plt.close(figure)
    return paths
