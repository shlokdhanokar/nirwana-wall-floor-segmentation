"""Label space and the ADE20K (SceneParsing-150) -> {other, wall, floor} mapping.

ADE20K annotation PNGs store 0 for "not annotated" and 1..150 for the 150
classes listed in objectInfo150.csv. We collapse them into a 3-class problem:

    0 = other   (furniture, ceiling, windows, doors, people, ...)
    1 = wall
    2 = floor   (floor + rug/carpet, matching the reference GT where the rug
                 is part of the floor mask)
  255 = ignore  (pixels the annotators never labelled; excluded from the loss)
"""

import numpy as np

# input normalisation (ImageNet channel statistics, matching the encoder's init), RGB, 0..255 scale
MEAN = np.array([0.485, 0.456, 0.406], np.float32) * 255
STD = np.array([0.229, 0.224, 0.225], np.float32) * 255

OTHER, WALL, FLOOR = 0, 1, 2
IGNORE = 255
CLASS_NAMES = ("other", "wall", "floor")
NUM_CLASSES = len(CLASS_NAMES)

# ADE20K-150 ids (1-based, as stored in the annotation PNGs)
ADE_WALL = 1
ADE_SKY = 3
ADE_FLOOR = 4
ADE_CEILING = 6
ADE_RUG = 29

ADE_TO_WALLFLOOR = np.zeros(256, dtype=np.uint8)  # every labelled class -> other
ADE_TO_WALLFLOOR[0] = IGNORE
ADE_TO_WALLFLOOR[ADE_WALL] = WALL
ADE_TO_WALLFLOOR[ADE_FLOOR] = FLOOR
ADE_TO_WALLFLOOR[ADE_RUG] = FLOOR


def remap_ade(label: np.ndarray) -> np.ndarray:
    """Map an ADE20K-150 label map (uint8) to the 3-class label space."""
    return ADE_TO_WALLFLOOR[label]


def is_interior(label: np.ndarray) -> bool:
    """Heuristic indoor filter based on ADE20K label statistics.

    ADE20K mixes indoor and outdoor scenes. We keep images with (almost) no sky,
    a meaningful amount of wall/floor, and evidence of a room (ceiling or floor).
    """
    n = label.size
    frac = lambda *ids: float(np.isin(label, ids).sum()) / n  # noqa: E731
    sky = frac(ADE_SKY)
    wall = frac(ADE_WALL)
    floor = frac(ADE_FLOOR, ADE_RUG)
    ceiling = frac(ADE_CEILING)
    return sky < 0.005 and (wall + floor) > 0.08 and (ceiling > 0.002 or floor > 0.01)
