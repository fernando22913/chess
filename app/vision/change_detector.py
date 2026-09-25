"""Lightweight image-change detection for the watch loop.

The watch loop must not run YOLO on every captured frame. This module provides
the cheap gate that runs first: a compact grayscale signature of the board and
a comparison that costs microseconds, so unchanged frames are discarded before
any neural inference happens.

Signatures are compared against the last *recognized* frame rather than the
last captured frame, so a change is reported whenever the position differs from
the one that produced the current FEN.
"""

from __future__ import annotations

import cv2
import numpy as np

# Small enough to keep the comparison trivial, large enough that moving a piece
# still crosses the change ratio.
SIGNATURE_SIZE = 160

# Per-pixel grayscale difference required before a pixel counts as "changed".
PIXEL_DIFF_THRESHOLD = 20

# Fraction of signature pixels that must change for the board to count as
# changed. One moved piece rewrites roughly 3% of the signature; a cursor or a
# single blinking highlight stays well under 0.2%.
DEFAULT_CHANGE_RATIO = 0.01


def board_signature(image: np.ndarray | None) -> np.ndarray | None:
    """Build the compact signature used for change comparison."""
    if image is None or getattr(image, "size", 0) == 0:
        return None

    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    return cv2.resize(
        gray,
        (SIGNATURE_SIZE, SIGNATURE_SIZE),
        interpolation=cv2.INTER_AREA,
    )


def changed_ratio(
    current: np.ndarray | None,
    previous: np.ndarray | None,
    pixel_threshold: int = PIXEL_DIFF_THRESHOLD,
) -> float | None:
    """Fraction of signature pixels that differ, or ``None`` if not comparable."""
    if current is None or previous is None:
        return None
    if current.shape != previous.shape:
        return None

    delta = cv2.absdiff(current, previous)
    changed = int(np.count_nonzero(delta > pixel_threshold))
    return changed / float(delta.size)


def has_changed(
    current: np.ndarray | None,
    previous: np.ndarray | None,
    ratio_threshold: float = DEFAULT_CHANGE_RATIO,
    pixel_threshold: int = PIXEL_DIFF_THRESHOLD,
) -> bool:
    """Decide whether the board image changed meaningfully.

    * No baseline to compare against (``previous`` is ``None``) or no new frame
      (``current`` is ``None``) -> no change: there is nothing to report.
    * Signature size mismatch -> change: the geometry no longer lines up and the
      position must be re-read.
    * Otherwise -> change only when enough pixels differ.
    """
    if current is None or previous is None:
        return False
    if current.shape != previous.shape:
        return True

    ratio = changed_ratio(current, previous, pixel_threshold)
    if ratio is None:
        return False
    return ratio >= ratio_threshold
