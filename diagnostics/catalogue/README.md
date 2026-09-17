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
