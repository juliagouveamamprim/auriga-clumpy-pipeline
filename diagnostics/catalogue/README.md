# Catalogue diagnostics

## Minimum-DGC-distance distribution

`scripts/plot_dmin_distribution.py` reads `scenario`, `repop_id`, and
`min_dgc_kpc` from the 500-repopulation catalogue, validates both scenarios,
and writes a two-panel absolute-count histogram: narrow linear bins for
fragile and logarithmic bins for resilient.

    python3 diagnostics/catalogue/scripts/plot_dmin_distribution.py \
        --input-csv outputs/diagnostics/full_catalogue_dgc_0500.csv \
        --output-dir diagnostics/catalogue/plots

PNG and PDF are written by default and must not be committed.

## Vmax of the D_min subhalo

`scripts/plot_dmin_vmax_distribution.py` scans the 500 HDF5 catalogues of
one scenario (default: `resilient`) in chunks.  It reconstructs `D_GC` from
the stored Earth-centred coordinates using the observer position saved in each
HDF5 input group, selects the unique D_min subhalo, and reads only that
subhalo's `Vmax`.  The reconstructed D_min is checked against
`min_dgc_kpc` in the input CSV before a one-panel absolute-count histogram is
written.  No per-halo output or CSV modification is made.

    python3 diagnostics/catalogue/scripts/plot_dmin_vmax_distribution.py \
        --input-root /path/to/Auriga_outputs_hdf5_v2 \
        --input-csv outputs/diagnostics/full_catalogue_dgc_0500.csv \
        --output-dir diagnostics/catalogue/plots/dmin_vmax_distribution \
        --n-bins 30

The command writes `dmin_vmax_distribution.png` and
`dmin_vmax_distribution.pdf`; these generated files must not be committed.
