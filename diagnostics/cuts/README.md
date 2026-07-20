# Subhalo cut diagnostics

Catalogue-wise diagnostics used to select pointlike and extended subhalo cuts for the NSIDE=2048 all-sky templates.

## Central-pixel scan

The script `scripts/scan_central_pixel_cuts.py` builds a combined HEALPix proxy map using:

- `Js` in the source pixel for pointlike halos;
- `Jtheta` in the central pixel for extended halos.

For each pair `(pointlike_f, extended_f)`, the cuts are defined by:

    J_cut = f * J_pixel_ref

Here, `J_pixel_ref` is the maximum individual pointlike or extended central-pixel proxy.

The main diagnostic is:

    max(discarded combined map) / max(final combined map)

This follows the pixel-wise metric previously used with CLUMPY-rendered discarded maps.

The output also records the numbers of pointlike and extended halos kept after each cut.

## Scope

This first-stage scan does not render extended-halo profiles or include their adjacent pixels.

After selecting a small number of candidate cut pairs, a separate conservative test will evaluate possible contributions from adjacent pixels.

Large HDF5 catalogues, FITS maps, temporary outputs, and full scan results should not be committed.
