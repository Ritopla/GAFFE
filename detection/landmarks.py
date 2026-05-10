"""
MediaPipe Face Mesh landmark index constants.

All indices refer to the 478-point Face Mesh model.
Grouped by anatomical region for clarity.

Reference: https://github.com/google/mediapipe/blob/master/mediapipe/modules/face_geometry/data/canonical_face_model_uv_visualization.png
"""

# ──────────────────────────────────────────────
# Eyebrow landmarks
# ──────────────────────────────────────────────

# Inner corners of the eyebrows (closest to the nose)
BROW_INNER_LEFT = 55
BROW_INNER_RIGHT = 285

# Mid-brow points (used for vertical displacement)
BROW_MID_LEFT = 70
BROW_MID_RIGHT = 300

# ──────────────────────────────────────────────
# Eye landmarks (for EAR calculation)
# Using the 6-point model per eye:
#   p1 = outer corner, p2 = upper-lateral,
#   p3 = upper-medial, p4 = inner corner,
#   p5 = lower-medial, p6 = lower-lateral
# ──────────────────────────────────────────────

# Left eye
EYE_LEFT_OUTER = 33       # p1
EYE_LEFT_UPPER_LAT = 160  # p2
EYE_LEFT_UPPER_MED = 158  # p3
EYE_LEFT_INNER = 133      # p4
EYE_LEFT_LOWER_MED = 153  # p5
EYE_LEFT_LOWER_LAT = 144  # p6

# Right eye
EYE_RIGHT_OUTER = 263     # p1
EYE_RIGHT_UPPER_LAT = 385 # p2
EYE_RIGHT_UPPER_MED = 387 # p3
EYE_RIGHT_INNER = 362     # p4
EYE_RIGHT_LOWER_MED = 373 # p5
EYE_RIGHT_LOWER_LAT = 380 # p6

# Upper eyelid top (for brow-eyelid distance)
EYELID_TOP_LEFT = 159
EYELID_TOP_RIGHT = 386

# ──────────────────────────────────────────────
# Face contour (for normalization)
# ──────────────────────────────────────────────

FACE_LEFT_CONTOUR = 234    # Left temple / jaw
FACE_RIGHT_CONTOUR = 454   # Right temple / jaw
FACE_TOP = 10              # Top of forehead
FACE_BOTTOM = 152          # Chin

# ──────────────────────────────────────────────
# Convenience tuples for batch access
# ──────────────────────────────────────────────

LEFT_EYE_INDICES = (
    EYE_LEFT_OUTER,
    EYE_LEFT_UPPER_LAT,
    EYE_LEFT_UPPER_MED,
    EYE_LEFT_INNER,
    EYE_LEFT_LOWER_MED,
    EYE_LEFT_LOWER_LAT,
)

RIGHT_EYE_INDICES = (
    EYE_RIGHT_OUTER,
    EYE_RIGHT_UPPER_LAT,
    EYE_RIGHT_UPPER_MED,
    EYE_RIGHT_INNER,
    EYE_RIGHT_LOWER_MED,
    EYE_RIGHT_LOWER_LAT,
)
