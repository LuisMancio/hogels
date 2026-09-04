"""
rainbow_hologram_parameters.py

Purpose
-------
Stage 1 of the rainbow (dot-matrix) hologram pipeline.

This script does NOT draw any grating geometry and does NOT touch GDS at all.
Its only job is to turn 5 RGB perspective images of the object into a flat
numeric table: for every (macro-pixel position, source image, color channel)
combination, it stores the grating PERIOD (fixed per image+channel) and the
grating DUTY CYCLE (varies per pixel, encodes brightness).

That table is the input a later, separate script will consume to actually
draw the individual line gratings and export GDSII (e.g. following the
GDoeSII-style polygon-grouping approach discussed separately).

Physical background
--------------------
1) Grating period (fixed per image i, color channel c):

       d_i,c = lambda_c / (sin(alpha_i) + sin(beta))

   where:
       beta      = illumination angle from the substrate normal (fixed for
                   the whole hologram, single reference light source)
       alpha_i   = observation angle assigned to source image i (each of the
                   5 images is designed to diffract correctly at its own
                   viewing angle -> this is what produces the "swing" effect)
       lambda_c  = reference wavelength for channel c (R/G/B)

   NOTE ON SIGN CONVENTION: this uses the common reflection-grating form
   d*(sin(alpha) + sin(beta)) = m*lambda (1st order, m=1), with alpha and
   beta measured on the same side of the substrate normal. Verify this
   against your actual illumination/observation geometry before fabrication
   -- if your real setup has alpha and beta on opposite sides of the normal,
   flip the sign to d*(sin(alpha) - sin(beta)) = lambda instead.

2) Duty cycle from pixel intensity (varies per pixel, per image, per
   channel):

       I/I0 = 0.5 * (1 - cos(2*pi*h/d))

   Inverting for h/d (duty cycle), restricted to [0, 0.5] which is the
   physically useful branch (0 = no line/dark, 0.5 = optimum duty cycle /
   maximum diffraction efficiency / brightest):

       h/d = arccos(1 - 2*(I/I0)) / (2*pi)

   I/I0 here is simply the normalized pixel brightness (0..1) of that
   image's color channel at that pixel position.

Explicitly OUT OF SCOPE for this first version (add later, as separate
processing steps once this baseline pipeline is validated):
    - Proximity-effect dose correction (left to the e-beam operator / a
      simple empirical dose matrix for this first run, see conversation
      notes).
    - Controllable grating curvature to equalize R/G/B angular blur
      (the arc-shaped gratings from the reference paper). This script only
      ever produces straight-line gratings (implicit, since we only store
      period + duty cycle, no curvature parameter).

Output
------
A single CSV file (long format) with one row per
(macro_pixel_row, macro_pixel_col, image_index, channel), containing the
period and duty cycle for that grating, plus the fixed geometry/reference
parameters repeated for convenience. A downstream GDS-generation script can
group rows by (image_index, channel) to draw one grating "type" at a time.
"""

import numpy as np
import pandas as pd
from PIL import Image
from pathlib import Path


# ---------------------------------------------------------------------------
# CONFIG -- edit these for your actual setup
# ---------------------------------------------------------------------------

# Paths to your 5 RGB perspective images, in order from most negative to most
# positive observation angle. Any format Pillow can read (PNG, etc.).
IMAGE_PATHS = [
    "view_-10deg.png",
    "view_-05deg.png",
    "view_000deg.png",
    "view_+05deg.png",
    "view_+10deg.png",
]

# Observation angle (alpha_i, degrees) assigned to each image above, in the
# SAME order. Must have the same length as IMAGE_PATHS.
ALPHA_DEG = [-10.0, -5.0, 0.0, 5.0, 10.0]

# Illumination angle (beta, degrees), fixed for the whole hologram.
BETA_DEG = 45.0

# Reference wavelengths per color channel (nm). Standard RGB defaults.
WAVELENGTHS_NM = {
    "R": 620.0,
    "G": 540.0,
    "B": 470.0,
}

# Elementary grating size (micrometers). Determines natural diffraction
# blur gamma_nat = 2*lambda/a per channel -- not used directly by this
# script, but recorded in the output table for reference in later steps.
GRATING_SIZE_UM = 5.0

# Macro-pixel pitch (micrometers) -- the spacing of the coarse grid that
# each of the 15 (5 images x 3 channels) mini-gratings is packed into.
MACRO_PIXEL_PITCH_UM = 60.0

# Target number of macro-pixels along the WIDTH of the final pattern. The
# height is derived automatically from the aspect ratio of your actual
# source images (read from the first image in IMAGE_PATHS), so rectangular
# renders (e.g. 720x400) are NOT squashed into a square grid.
# Example: a 720x400 source image (aspect 1.8:1) with
# NUM_MACRO_PIXELS_X = 36 gives NUM_MACRO_PIXELS_Y = 20 (36/1.8 = 20).
NUM_MACRO_PIXELS_X = 36

# Output table path.
OUTPUT_CSV_PATH = "rainbow_hologram_layer_table.csv"


# ---------------------------------------------------------------------------
# PHYSICS
# ---------------------------------------------------------------------------

def grating_period_um(alpha_deg: float, beta_deg: float, wavelength_nm: float) -> float:
    """
    First-order reflection grating equation:
        d = lambda / (sin(alpha) + sin(beta))

    Returns the period in micrometers.

    See the sign-convention note in the module docstring before trusting
    this for a real fabrication run -- confirm against your actual
    illumination/observation geometry.
    """
    alpha_rad = np.radians(alpha_deg)
    beta_rad = np.radians(beta_deg)
    wavelength_um = wavelength_nm / 1000.0

    denom = np.sin(alpha_rad) + np.sin(beta_rad)
    if np.isclose(denom, 0.0):
        raise ValueError(
            f"sin(alpha)+sin(beta) is ~0 for alpha={alpha_deg}, beta={beta_deg}; "
            "grating period would be infinite (no diffraction solution here)."
        )
    return wavelength_um / denom


def natural_blur_deg(wavelength_nm: float, grating_size_um: float) -> float:
    """
    Natural (irreducible) diffraction blur of a finite grating patch:
        gamma_nat ~= 2 * lambda / a   (radians, small-angle-ish approx)

    Returned in degrees. Recorded for reference only -- not used to modify
    anything in this script (no curvature correction here).
    """
    wavelength_um = wavelength_nm / 1000.0
    gamma_rad = 2.0 * wavelength_um / grating_size_um
    return np.degrees(gamma_rad)


def intensity_to_duty_cycle(intensity_norm: np.ndarray) -> np.ndarray:
    """
    Invert I/I0 = 0.5*(1 - cos(2*pi*h/d)) for h/d, restricted to the
    physically useful branch [0, 0.5]:

        h/d = arccos(1 - 2*(I/I0)) / (2*pi)

    intensity_norm: array of values in [0, 1] (normalized pixel brightness).
    Returns duty cycle values in [0, 0.5].
    """
    intensity_norm = np.clip(intensity_norm, 0.0, 1.0)
    arg = np.clip(1.0 - 2.0 * intensity_norm, -1.0, 1.0)  # guard against fp drift
    duty_cycle = np.arccos(arg) / (2.0 * np.pi)
    return duty_cycle


# ---------------------------------------------------------------------------
# IMAGE LOADING
# ---------------------------------------------------------------------------

def derive_macro_pixel_grid_size(first_image_path: str, target_width_cells: int) -> tuple:
    """
    Reads the real width/height of the first source image and derives
    (num_macro_pixels_x, num_macro_pixels_y) preserving that aspect ratio,
    given a target width in macro-pixel cells.

    This avoids squashing rectangular renders (e.g. 720x400) into a square
    macro-pixel grid.
    """
    with Image.open(first_image_path) as img:
        src_width, src_height = img.size

    aspect_ratio = src_width / src_height  # e.g. 720/400 = 1.8
    num_x = target_width_cells
    num_y = max(1, round(target_width_cells / aspect_ratio))
    return num_x, num_y


def load_image_as_normalized_rgb(path: str, target_size_xy: tuple) -> np.ndarray:
    """
    Loads an RGB image, resizes it to target_size_xy = (num_macro_pixels_x,
    num_macro_pixels_y) -- one image pixel per macro-pixel -- and returns a
    float array of shape (height, width, 3) normalized to [0, 1].

    Resizing here effectively defines your final image resolution: each
    output pixel becomes exactly one macro-pixel cell in the hologram.
    Pass a target_size_xy that matches your source images' aspect ratio
    (see derive_macro_pixel_grid_size) to avoid distortion.
    """
    img = Image.open(path).convert("RGB")
    img = img.resize(target_size_xy, resample=Image.LANCZOS)
    arr = np.asarray(img).astype(np.float64) / 255.0
    return arr  # shape: (height, width, 3), values in [0, 1]


# ---------------------------------------------------------------------------
# MAIN PIPELINE
# ---------------------------------------------------------------------------

def build_layer_table() -> pd.DataFrame:
    if len(IMAGE_PATHS) != len(ALPHA_DEG):
        raise ValueError("IMAGE_PATHS and ALPHA_DEG must have the same length.")

    channels = ["R", "G", "B"]
    channel_index = {"R": 0, "G": 1, "B": 2}

    if not Path(IMAGE_PATHS[0]).exists():
        raise FileNotFoundError(f"Image not found: {IMAGE_PATHS[0]}")
    num_macro_pixels_x, num_macro_pixels_y = derive_macro_pixel_grid_size(
        IMAGE_PATHS[0], NUM_MACRO_PIXELS_X
    )
    target_size = (num_macro_pixels_x, num_macro_pixels_y)  # (width, height)
    print(f"Derived macro-pixel grid: {num_macro_pixels_x} x {num_macro_pixels_y} "
          f"(from aspect ratio of {IMAGE_PATHS[0]})")

    rows = []

    for img_idx, (path, alpha_deg) in enumerate(zip(IMAGE_PATHS, ALPHA_DEG), start=1):
        if not Path(path).exists():
            raise FileNotFoundError(f"Image not found: {path}")

        rgb = load_image_as_normalized_rgb(path, target_size)
        # rgb shape: (num_macro_pixels_y, num_macro_pixels_x, 3)

        for channel in channels:
            wavelength_nm = WAVELENGTHS_NM[channel]

            # One fixed period for this (image, channel) combination.
            period_um = grating_period_um(alpha_deg, BETA_DEG, wavelength_nm)
            gamma_nat_deg = natural_blur_deg(wavelength_nm, GRATING_SIZE_UM)

            channel_plane = rgb[:, :, channel_index[channel]]  # (rows, cols)
            duty_cycle_plane = intensity_to_duty_cycle(channel_plane)

            for row in range(num_macro_pixels_y):
                for col in range(num_macro_pixels_x):
                    rows.append({
                        "image_index": img_idx,
                        "alpha_deg": alpha_deg,
                        "channel": channel,
                        "wavelength_nm": wavelength_nm,
                        "period_um": period_um,
                        "natural_blur_deg": gamma_nat_deg,
                        "macro_pixel_row": row,
                        "macro_pixel_col": col,
                        "macro_pixel_pitch_um": MACRO_PIXEL_PITCH_UM,
                        "elementary_grating_size_um": GRATING_SIZE_UM,
                        "pixel_intensity_norm": channel_plane[row, col],
                        "duty_cycle": duty_cycle_plane[row, col],
                    })

    df = pd.DataFrame(rows)
    return df


def main():
    df = build_layer_table()
    df.to_csv(OUTPUT_CSV_PATH, index=False)

    grid_x = df["macro_pixel_col"].nunique()
    grid_y = df["macro_pixel_row"].nunique()
    print(f"Wrote {len(df)} rows to {OUTPUT_CSV_PATH}")
    print(f"({len(IMAGE_PATHS)} images x 3 channels x {grid_x}x{grid_y} macro-pixels)")

    # Quick sanity printout: the 15 fixed periods, one per (image, channel).
    summary = (
        df[["image_index", "alpha_deg", "channel", "wavelength_nm",
            "period_um", "natural_blur_deg"]]
        .drop_duplicates()
        .sort_values(["image_index", "channel"])
    )
    print("\nFixed periods per (image, channel):")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()