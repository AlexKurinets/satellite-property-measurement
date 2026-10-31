"""Class lists for the two label sets.

The base model was trained on public land-cover classes. The fine-tuned model predicts the classes a quote needs.
The indices line up so the base model can be scored on the same benchmark, but the definitions differ: base
"bare_soil" covers mulch beds and dirt alike, and base "grass" has never seen a dormant lawn.
"""

BASE_CLASSES = ["other", "grass", "bare_soil", "impervious", "building", "tree"]
CLIENT_CLASSES = ["other", "lawn", "bed", "pavement", "roof", "canopy"]

OTHER, LAWN, BED, PAVEMENT, ROOF, CANOPY = range(6)
N_CLASSES = 6

# RGBA used for overlays
CLASS_COLORS = {
    OTHER: (0, 0, 0, 0),
    LAWN: (64, 200, 90, 110),
    BED: (200, 120, 40, 140),
    PAVEMENT: (140, 140, 255, 120),
    ROOF: (240, 60, 60, 120),
    CANOPY: (20, 110, 40, 150),
}

FEATURES = ["lawn", "bed", "pavement", "roof"]
