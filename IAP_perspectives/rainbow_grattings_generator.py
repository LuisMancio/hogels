"""
rainbow_grattings_generator.py

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
   image's color channel at that pixel position -- linearized from sRGB
   first (see LINEARIZE_SRGB / srgb_to_linear) since PNG pixel values are
   gamma-encoded, not physical intensity.

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

Rows are skipped whenever the resulting grating LINE WIDTH (duty_cycle *
period_um) would fall below MIN_LINE_WIDTH_UM -- this covers true black
pixels (duty cycle exactly 0) but also near-black ones. In practice, PNG
resize (LANCZOS) ringing near a sharp edge can leave isolated pixels at
raw value 1/255 where the source was meant to be pure black; that maps to
a nonzero but physically absurd sub-nanometer line width if not filtered.
Skipping by line width (rather than by raw intensity) ties the cutoff to
something a fab engineer actually cares about -- no real process can pattern
a line thinner than MIN_LINE_WIDTH_UM anyway, so drawing it would just be
dead weight in the table (and later in the GDS).

Each row also carries cell_col_index (0/1/2, position of this channel's
column -- R/G/B -- within the 60x60 um mother cell) and cell_row_index
(0..4, position of this image's row within the mother cell, following the
IMAGE_PATHS/ALPHA_DEG order). This gives the downstream GDS-generation
script the mini-grating's position inside its mother cell explicitly,
instead of having it re-derive that from channel name / image_index.

num_macro_pixels_x / num_macro_pixels_y (the full derived grid size, same
value repeated on every row) are also included so a downstream script knows
the true canvas extent even if the outer rows/columns were skipped for
being all-black.

COORDINATE CONVENTION WARNING: macro_pixel_row follows numpy/image
convention, i.e. row 0 is the TOP of the source image and row increases
DOWNWARD. GDS/gdsfactory coordinates increase UPWARD (standard Cartesian y).
A downstream script that maps macro_pixel_row directly to a y coordinate
will produce a vertically flipped (upside-down) layout -- it must first
invert the row, e.g. y = (num_macro_pixels_y - 1 - macro_pixel_row) * pitch.
macro_pixel_col needs no such flip (left-to-right matches in both
conventions).
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
NUM_MACRO_PIXELS_X = 250

# Minimum fabricable grating line width (micrometers). Any pixel whose
# duty_cycle * period_um would produce a thinner line than this is skipped
# entirely (treated the same as a black pixel -- no grating drawn), instead
# of writing a row no real process could pattern anyway.
#
# This is a conservative, GENERIC sanity floor (roughly the edge of what
# advanced e-beam can do), not a stand-in for your actual chosen process's
# resolution -- e.g. if you end up on the 1.5 um DMD/UV tool discussed in
# the project notes, you'd want this closer to 1.5 um (and likely also
# revisit the grating period itself, see fabrication summary caveats).
MIN_LINE_WIDTH_UM = 0.1

# Whether to convert pixel values from sRGB (gamma-encoded -- what PNGs,
# including Blender renders, normally store) to linear light before treating
# them as I/I0 in the duty cycle formula. Physically more correct; set to
# False to use the raw sRGB values directly instead.
LINEARIZE_SRGB = True

# Output table path.
OUTPUT_CSV_PATH = "rainbow_hologram_layer_table.csv"

# Plain-text fabrication summary path -- a human-readable rundown of period,
# duty cycle, feature width and write-area ranges, meant to be handed to
# fab technicians to help pick a process/tool (see write_fabrication_summary).
FAB_SUMMARY_TXT_PATH = "rainbow_hologram_fab_summary.txt"


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


def srgb_to_linear(channel_srgb: np.ndarray) -> np.ndarray:
    """
    Converts sRGB-encoded (gamma) values in [0, 1] to linear light intensity,
    using the standard sRGB EOTF:

        c_linear = c / 12.92                     if c <= 0.04045
                 = ((c + 0.055) / 1.055) ** 2.4   otherwise

    PNG images (e.g. Blender renders) store pixel values in sRGB space, not
    linear light. Treating them as I/I0 directly would bias the perceived
    brightness -> diffraction efficiency mapping, since sRGB's gamma curve
    is not proportional to physical intensity.
    """
    channel_srgb = np.clip(channel_srgb, 0.0, 1.0)
    low = channel_srgb <= 0.04045
    return np.where(low, channel_srgb / 12.92, ((channel_srgb + 0.055) / 1.055) ** 2.4)


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

    If LINEARIZE_SRGB is True, values are converted from sRGB to linear
    light (see srgb_to_linear) before being returned.
    """
    img = Image.open(path).convert("RGB")
    img = img.resize(target_size_xy, resample=Image.LANCZOS)
    arr = np.asarray(img).astype(np.float64) / 255.0
    if LINEARIZE_SRGB:
        arr = srgb_to_linear(arr)
    return arr  # shape: (height, width, 3), values in [0, 1]


# ---------------------------------------------------------------------------
# MAIN PIPELINE
# ---------------------------------------------------------------------------

def build_layer_table() -> tuple:
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
    physical_width_mm = num_macro_pixels_x * MACRO_PIXEL_PITCH_UM / 1000.0
    physical_height_mm = num_macro_pixels_y * MACRO_PIXEL_PITCH_UM / 1000.0
    print(f"Derived macro-pixel grid: {num_macro_pixels_x} x {num_macro_pixels_y} "
          f"(from aspect ratio of {IMAGE_PATHS[0]})")
    print(f"Resulting physical hologram size: "
          f"{physical_width_mm:.3f} mm x {physical_height_mm:.3f} mm "
          f"(= macro-pixel count x {MACRO_PIXEL_PITCH_UM} um pitch)")
    print("NOTE: this size depends only on NUM_MACRO_PIXELS_X and "
          "MACRO_PIXEL_PITCH_UM, NOT on your source image's native pixel "
          "resolution -- the source image is downscaled to fit the grid.")

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

            # Position of this (channel, image) mini-grating within the
            # 60x60 um mother cell: column by channel (R/G/B), row by image
            # (follows IMAGE_PATHS/ALPHA_DEG order).
            cell_col_index = channel_index[channel]
            cell_row_index = img_idx - 1

            for row in range(num_macro_pixels_y):
                for col in range(num_macro_pixels_x):
                    intensity = channel_plane[row, col]
                    duty_cycle = duty_cycle_plane[row, col]
                    if duty_cycle * period_um < MIN_LINE_WIDTH_UM:
                        continue  # unfabricably thin (or exactly black) -> skip

                    rows.append({
                        "image_index": img_idx,
                        "alpha_deg": alpha_deg,
                        "channel": channel,
                        "wavelength_nm": wavelength_nm,
                        "period_um": period_um,
                        "natural_blur_deg": gamma_nat_deg,
                        "macro_pixel_row": row,
                        "macro_pixel_col": col,
                        "num_macro_pixels_x": num_macro_pixels_x,
                        "num_macro_pixels_y": num_macro_pixels_y,
                        "cell_col_index": cell_col_index,
                        "cell_row_index": cell_row_index,
                        "macro_pixel_pitch_um": MACRO_PIXEL_PITCH_UM,
                        "elementary_grating_size_um": GRATING_SIZE_UM,
                        "pixel_intensity_norm": intensity,
                        "duty_cycle": duty_cycle,
                    })

    df = pd.DataFrame(rows)
    return df, num_macro_pixels_x, num_macro_pixels_y


def write_fabrication_summary(df: pd.DataFrame, num_macro_pixels_x: int,
                               num_macro_pixels_y: int, path: str) -> None:
    """
    Writes a plain-text summary of the fabrication-relevant ranges in the
    table -- period, duty cycle, resulting line/gap width, and write area --
    meant to be handed to fab technicians so they can pick a process/tool
    (e-beam, DMD/UV projection, etc.) capable of resolving the smallest
    feature and covering the total write area.
    """
    pitch_um = float(df["macro_pixel_pitch_um"].iloc[0])
    grating_size_um = float(df["elementary_grating_size_um"].iloc[0])

    line_width_um = df["duty_cycle"] * df["period_um"]
    gap_width_um = df["period_um"] - line_width_um

    # Nominal canvas: the full derived macro-pixel grid, including any
    # all-black margin that was skipped.
    nominal_width_mm = num_macro_pixels_x * pitch_um / 1000.0
    nominal_height_mm = num_macro_pixels_y * pitch_um / 1000.0

    # Actual content bounding box: the smallest rectangle (in macro-pixel
    # cells) that contains EVERY grating that was actually written, however
    # sparse. Resize antialiasing (LANCZOS) can leave a handful of isolated
    # near-black (but not exactly black) macro-pixels scattered well outside
    # the visually dense object -- this box includes those, so it's a
    # worst-case / upper-bound extent, not a good write-area estimate.
    col_min, col_max = int(df["macro_pixel_col"].min()), int(df["macro_pixel_col"].max())
    row_min, row_max = int(df["macro_pixel_row"].min()), int(df["macro_pixel_row"].max())
    content_width_mm = (col_max - col_min + 1) * pitch_um / 1000.0
    content_height_mm = (row_max - row_min + 1) * pitch_um / 1000.0

    # Dense content bounding box: trims the 0.5% sparsest columns/rows on
    # each side (i.e. keeps the middle 99% of gratings by position). This is
    # a much more realistic write-area estimate, since it isn't skewed by a
    # handful of stray outlier pixels the way the box above can be.
    col_lo, col_hi = df["macro_pixel_col"].quantile([0.005, 0.995])
    row_lo, row_hi = df["macro_pixel_row"].quantile([0.005, 0.995])
    dense_width_mm = (col_hi - col_lo) * pitch_um / 1000.0
    dense_height_mm = (row_hi - row_lo) * pitch_um / 1000.0
    num_outlier_gratings = int((
        (df["macro_pixel_col"] < col_lo) | (df["macro_pixel_col"] > col_hi) |
        (df["macro_pixel_row"] < row_lo) | (df["macro_pixel_row"] > row_hi)
    ).sum())

    periods_table = (
        df[["image_index", "alpha_deg", "channel", "wavelength_nm", "period_um"]]
        .drop_duplicates()
        .sort_values(["image_index", "channel"])
    )

    lines = []
    lines.append("RAINBOW HOLOGRAM -- FABRICATION SUMMARY")
    lines.append("=" * 60)
    lines.append("")
    lines.append("Purpose: reference numbers for choosing a lithography process/tool")
    lines.append("(minimum resolvable feature, alignment, and total write area).")
    lines.append("")
    lines.append("-- WRITE AREA --")
    lines.append(f"Macro-pixel pitch: {pitch_um:.3f} um")
    lines.append(f"Macro-pixel grid (nominal, full canvas): "
                  f"{num_macro_pixels_x} x {num_macro_pixels_y} cells "
                  f"({nominal_width_mm:.3f} mm x {nominal_height_mm:.3f} mm)")
    lines.append(f"Full content bounding box (every grating, incl. isolated outliers): "
                  f"{col_max - col_min + 1} x {row_max - row_min + 1} cells "
                  f"({content_width_mm:.3f} mm x {content_height_mm:.3f} mm)")
    lines.append(f"Dense content bounding box (middle 99% of gratings -- RECOMMENDED "
                  f"write-area estimate): {dense_width_mm:.3f} mm x {dense_height_mm:.3f} mm "
                  f"({num_outlier_gratings} of {len(df)} gratings fall outside this box, "
                  f"scattered in the sparse margin)")
    lines.append(f"Elementary grating size: {grating_size_um:.3f} um x {grating_size_um:.3f} um")
    lines.append(f"Total individual gratings to write: {len(df)}")
    lines.append("")
    lines.append("-- GRATING PERIOD (d) -- fixed per image + color channel --")
    lines.append(f"Min period: {df['period_um'].min():.4f} um")
    lines.append(f"Max period: {df['period_um'].max():.4f} um")
    lines.append("")
    lines.append("Full breakdown (15 combinations):")
    lines.append(periods_table.to_string(index=False))
    lines.append("")
    lines.append("-- DUTY CYCLE (h/d) -- varies per pixel, encodes brightness --")
    lines.append(f"Min duty cycle: {df['duty_cycle'].min():.4f}")
    lines.append(f"Max duty cycle: {df['duty_cycle'].max():.4f}")
    lines.append("(Theoretical max from the physics: 0.5)")
    lines.append("")
    lines.append("-- RESULTING FEATURE WIDTHS -- this is what the tool must resolve --")
    lines.append(f"Min line width  (duty_cycle x period): {line_width_um.min():.4f} um")
    lines.append(f"Max line width  (duty_cycle x period): {line_width_um.max():.4f} um")
    lines.append(f"Min gap width   (period - line width): {gap_width_um.min():.4f} um")
    lines.append(f"Max gap width   (period - line width): {gap_width_um.max():.4f} um")
    lines.append("")
    lines.append("-- NOTES / CAVEATS --")
    lines.append(f"- A generic sanity floor of MIN_LINE_WIDTH_UM = {MIN_LINE_WIDTH_UM} um is already")
    lines.append("  applied: any pixel whose line width would fall below that is skipped, so the")
    lines.append("  min line width above should never be absurdly thin (e.g. sub-nanometer). This")
    lines.append("  is a generic floor, not your chosen process's real resolution -- once a process")
    lines.append("  is picked, tighten it to match (e.g. ~1.5 um for the DMD/UV tool discussed).")
    lines.append("- Sign convention: d*(sin(alpha)+sin(beta)) = lambda, alpha and beta on")
    lines.append("  the same side of the substrate normal. Verify against the real")
    lines.append("  illumination/observation geometry before fabrication.")
    lines.append("- These ranges are the true (non-quantized) physical values from the")
    lines.append("  design. A downstream GDS-generation step may quantize duty cycle to a")
    lines.append("  handful of discrete levels to keep the file small, but the equipment")
    lines.append("  still needs to resolve the min/max feature widths listed above.")
    lines.append("- Curvature correction (R/G/B blur) and proximity-effect dose correction")
    lines.append("  are not applied at this stage.")

    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    df, num_macro_pixels_x, num_macro_pixels_y = build_layer_table()
    df.to_csv(OUTPUT_CSV_PATH, index=False)

    total_possible_rows = len(IMAGE_PATHS) * 3 * num_macro_pixels_x * num_macro_pixels_y
    skipped_rows = total_possible_rows - len(df)
    print(f"Wrote {len(df)} rows to {OUTPUT_CSV_PATH}")
    print(f"({len(IMAGE_PATHS)} images x 3 channels x "
          f"{num_macro_pixels_x}x{num_macro_pixels_y} macro-pixels grid; "
          f"{skipped_rows} rows skipped as black/unfabricably thin "
          f"(<{MIN_LINE_WIDTH_UM} um line))")

    # Quick sanity printout: the 15 fixed periods, one per (image, channel).
    summary = (
        df[["image_index", "alpha_deg", "channel", "wavelength_nm",
            "period_um", "natural_blur_deg"]]
        .drop_duplicates()
        .sort_values(["image_index", "channel"])
    )
    print("\nFixed periods per (image, channel):")
    print(summary.to_string(index=False))

    write_fabrication_summary(df, num_macro_pixels_x, num_macro_pixels_y, FAB_SUMMARY_TXT_PATH)
    print(f"\nWrote fabrication summary to {FAB_SUMMARY_TXT_PATH}")


if __name__ == "__main__":
    main()