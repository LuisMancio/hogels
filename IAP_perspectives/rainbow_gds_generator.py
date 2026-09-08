"""
rainbow_gds_generator.py

Purpose
-------
Stage 3 of the rainbow (dot-matrix) hologram pipeline.

Reads the flat parameter table produced by rainbow_grattings_generator.py
(stage 2 -- one row per non-black macro-pixel/image/channel combination, with
a fixed period and a per-pixel duty cycle) and draws the actual grating
geometry, exporting a single GDSII file for EBL.

Design decisions for this first version
----------------------------------------
- Duty cycle is quantized to N_DUTY_LEVELS discrete levels (default 16)
  before drawing. Combined with the fact that period is already fixed per
  (image_index, channel) -- 5 images x 3 channels = 15 possible periods --
  this means there are at most 15 x N_DUTY_LEVELS unique grating "types" in
  the whole hologram. Each unique type is drawn ONCE as a small gdsfactory
  Component and every macro-pixel that needs it gets a reference
  (ComponentReference) to that same component instead of its own copy of the
  polygons. This is what keeps the GDS file small despite tens of thousands
  of grating placements (the "GDoeSII-style polygon-grouping" approach
  mentioned in rainbow-ebeam.pdf / CLAUDE.md).
- Each mini-grating (elementary_grating_size_um x elementary_grating_size_um,
  5x5 um by default) is CENTERED inside its slot within the 60x60 um mother
  cell. The mother cell is divided into 3 columns (R, G, B) x 5 rows (one
  per source image / observation angle), so each slot is
  (macro_pixel_pitch_um/3) x (macro_pixel_pitch_um/5) -- 20x12 um by
  default -- and the 5x5 um grating sits centered in it.
- Grating lines run vertically (periodic along x, spanning the full slot
  height along y) -- i.e. this models a hologram whose single diffraction
  direction is horizontal (alpha/beta measured in the x-z plane). Flip the
  orientation in build_grating_cell() if your real illumination/observation
  geometry differs.
- Each color channel is drawn on its own GDS layer (R/G/B), so KLayout can
  show/hide/edit them independently and downstream fab steps can treat them
  separately if needed.

Explicitly OUT OF SCOPE for this first version (see CLAUDE.md):
    - Grating curvature to equalize R/G/B angular blur.
    - Proximity-effect dose correction.
    - Further GDS size optimization (e.g. GDoeSII-specific tricks beyond
      basic cell reuse) -- this script only does the "obvious" reuse
      (dedup by period+quantized duty cycle), nothing more.
"""

import gdsfactory as gf
import pandas as pd
from pathlib import Path


# ---------------------------------------------------------------------------
# CONFIG -- edit these for your actual setup
# ---------------------------------------------------------------------------

INPUT_CSV_PATH = "rainbow_hologram_layer_table.csv"
OUTPUT_GDS_PATH = "rainbow_hologram.gds"

# Mother cell layout (fixed by the experiment design): 3 columns (R,G,B) x
# 5 rows (one per source image / observation angle).
CELL_COLUMNS = 3
CELL_ROWS = 5

# Number of discrete duty-cycle levels used to quantize the (continuous)
# duty_cycle column before drawing, so grating geometry can be reused across
# macro-pixels instead of drawing one unique grating per row.
N_DUTY_LEVELS = 16

# Theoretical max duty cycle from the physics (see rainbow_grattings_generator.py):
# h/d in [0, 0.5].
MAX_DUTY_CYCLE = 0.5

# GDS layer (layer, datatype) per color channel.
LAYER_BY_CHANNEL = {
    "R": (1, 0),
    "G": (2, 0),
    "B": (3, 0),
}

TOP_CELL_NAME = "rainbow_hologram"


# ---------------------------------------------------------------------------
# GEOMETRY
# ---------------------------------------------------------------------------

def quantize_duty_cycle(duty_cycle: float, n_levels: int, max_duty: float) -> float:
    """
    Snaps a continuous duty cycle value to the nearest of n_levels evenly
    spaced levels in [0, max_duty]. This is what lets many macro-pixels
    share the same grating Component instead of each getting a unique one.
    """
    if n_levels <= 1:
        return 0.0
    step_index = round(duty_cycle / max_duty * (n_levels - 1))
    step_index = min(max(step_index, 0), n_levels - 1)
    return step_index / (n_levels - 1) * max_duty


def build_grating_cell(period_um: float, duty_cycle: float, size_um: float,
                        layer: tuple) -> gf.Component:
    """
    Draws one elementary grating: vertical lines of width
    (duty_cycle * period_um), spaced by period_um, filling a size_um x
    size_um square. Lines that would overflow the square are clipped to fit.
    """
    comp = gf.Component()
    line_width_um = duty_cycle * period_um
    if line_width_um <= 0.0 or period_um <= 0.0:
        return comp  # duty cycle 0 (or degenerate period) -> no visible lines

    num_lines = int(size_um // period_um)
    for i in range(num_lines):
        x0 = i * period_um
        x1 = min(x0 + line_width_um, size_um)
        if x1 <= x0:
            continue
        comp.add_polygon([(x0, 0.0), (x1, 0.0), (x1, size_um), (x0, size_um)], layer=layer)
    return comp


# ---------------------------------------------------------------------------
# MAIN PIPELINE
# ---------------------------------------------------------------------------

def build_hologram_gds() -> gf.Component:
    if not Path(INPUT_CSV_PATH).exists():
        raise FileNotFoundError(f"Input table not found: {INPUT_CSV_PATH}")

    gf.gpdk.PDK.activate()

    df = pd.read_csv(INPUT_CSV_PATH)

    # Single source of truth for geometry constants: read them from the
    # table itself (written by stage 2) instead of duplicating the numbers
    # here, so the two stages can't silently drift apart.
    macro_pixel_pitch_um = float(df["macro_pixel_pitch_um"].iloc[0])
    grating_size_um = float(df["elementary_grating_size_um"].iloc[0])

    slot_width_um = macro_pixel_pitch_um / CELL_COLUMNS
    slot_height_um = macro_pixel_pitch_um / CELL_ROWS
    margin_x_um = (slot_width_um - grating_size_um) / 2.0
    margin_y_um = (slot_height_um - grating_size_um) / 2.0
    if margin_x_um < 0 or margin_y_um < 0:
        raise ValueError(
            f"Grating size ({grating_size_um} um) does not fit in its slot "
            f"({slot_width_um} x {slot_height_um} um) -- check "
            "MACRO_PIXEL_PITCH_UM / GRATING_SIZE_UM in stage 2."
        )

    df["duty_cycle_q"] = df["duty_cycle"].apply(
        lambda d: quantize_duty_cycle(d, N_DUTY_LEVELS, MAX_DUTY_CYCLE)
    )

    # macro_pixel_row follows numpy/image convention (row 0 = top of source
    # image, row increases downward), but GDS y increases upward. Without
    # flipping, the hologram comes out upside-down. macro_pixel_col needs no
    # such flip (left-to-right already matches between the two conventions).
    num_macro_pixels_y = int(df["num_macro_pixels_y"].iloc[0])

    top = gf.Component(name=TOP_CELL_NAME)
    grating_cell_cache: dict[tuple, gf.Component] = {}
    num_placed = 0

    for row in df.itertuples(index=False):
        cache_key = (row.channel, round(row.period_um, 6), round(row.duty_cycle_q, 6))
        cell = grating_cell_cache.get(cache_key)
        if cell is None:
            cell = build_grating_cell(
                period_um=row.period_um,
                duty_cycle=row.duty_cycle_q,
                size_um=grating_size_um,
                layer=LAYER_BY_CHANNEL[row.channel],
            )
            grating_cell_cache[cache_key] = cell

        flipped_row = num_macro_pixels_y - 1 - row.macro_pixel_row
        macro_x0_um = row.macro_pixel_col * macro_pixel_pitch_um
        macro_y0_um = flipped_row * macro_pixel_pitch_um
        slot_x_um = macro_x0_um + row.cell_col_index * slot_width_um + margin_x_um
        slot_y_um = macro_y0_um + row.cell_row_index * slot_height_um + margin_y_um

        ref = top.add_ref(cell)
        ref.dmove((slot_x_um, slot_y_um))
        num_placed += 1

    print(f"Placed {num_placed} gratings using {len(grating_cell_cache)} unique grating cells "
          f"(vs. {num_placed} if each row had its own geometry).")

    return top


def main():
    top = build_hologram_gds()
    out_path = top.write_gds(OUTPUT_GDS_PATH)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
