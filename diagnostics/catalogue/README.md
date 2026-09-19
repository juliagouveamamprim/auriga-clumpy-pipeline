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

`scripts/plot_dmin_vmax_distribution.py` has separate resumable scan and plot
stages.  The scan processes the 500 HDF5 catalogues of one scenario (default:
`resilient`) in chunks.  It reconstructs `D_GC` from the stored Earth-centred
coordinates using the observer position saved in each HDF5 input group,
selects the unique D_min subhalo, and reads only that subhalo's `Vmax`.  Each
reconstructed D_min is checked against `min_dgc_kpc` in the input CSV before a
row is atomically checkpointed.  The checkpoint columns are `scenario`,
`repop_id`, `source_h5`, `dmin_kpc`, and `vmax_kms`; rerunning the scan skips
only already validated rows.

Run the scan on Lattes without creating a figure (and therefore without a
LaTeX `type1cm.sty` dependency):

    python3 diagnostics/catalogue/scripts/plot_dmin_vmax_distribution.py \
        --input-root /path/to/Auriga_outputs_hdf5_v2 \
        --input-csv outputs/diagnostics/full_catalogue_dgc_0500.csv \
        --output-csv outputs/diagnostics/dmin_vmax_distribution.csv \
        --skip-plot

After copying that checkpoint CSV to Dell, create the same histogram without
opening any HDF5 catalogue or requiring `full_catalogue_dgc_0500.csv`:

    python3 diagnostics/catalogue/scripts/plot_dmin_vmax_distribution.py \
        --output-csv /path/to/dmin_vmax_distribution.csv \
        --output-dir diagnostics/catalogue/plots/dmin_vmax_distribution \
        --n-bins 30 \
        --plot-only

The plot command writes `dmin_vmax_distribution.png` and
`dmin_vmax_distribution.pdf`; these generated files must not be committed.
