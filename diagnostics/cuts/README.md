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

## Conservative extended-halo envelopes

Optional conservative envelopes can be enabled with:

    --extended-envelope-modes aperture,theta-s

The available modes are:

- `aperture`: repeats each discarded extended-halo `Jtheta` contribution
  in every HEALPix pixel intersected by the aperture used to calculate
  `Jtheta`;
- `theta-s`: repeats the same contribution in every pixel intersected
  by a disc of radius `theta_s`. This is an intentionally extreme
  stress test; `theta_s` is a characteristic angular scale, not a
  physical outer boundary.

The central-pixel result, aperture envelope, and `theta-s` envelope are
stored as separate pixel-wise metrics.

These envelopes are not physical rendered maps. They deliberately
repeat the same `Jtheta` contribution across multiple pixels and must
not be used for integrated or cumulative J-factor quantities. All
integrated quantities in the CSV continue to use the central-pixel
proxy only.

## Scope

This diagnostic does not render the radial surface-brightness profiles
of extended halos. The envelope modes provide conservative pixel-wise
bounds for choosing cut candidates before a smaller number of full
CLUMPY tests.

Large HDF5 catalogues, FITS maps, temporary outputs, and full scan
results should not be committed.
