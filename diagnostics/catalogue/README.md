# Catalogue diagnostics

## Brightest retained production subhalo

`scripts/scan_retained_subhalo_jsmax.py` identifies the retained pointlike and
extended populations for each completed CLUMPY case and records the retained
subhalo with the largest catalogue `Js`.  Pointlike membership reuses
`build_valid_mask()` and the effective `THETACUT`/`JCPL` stored in the
pointlike FITS.  Extended membership comes from the corrected external-halo
list and must match the corrected `halo_rendered.log` exactly in count, name,
type, and order.

The catalogue is read once in bounded chunks.  The pointlike map columns are
not read, and the diagnostic deliberately does not compare summed catalogue
`Js` with the pointlike FITS map.  Every retained object is streamed during
that same pass to:

    retained_subhalos_<scenario>_repop_XXXX_nside<NSIDE>.csv.gz

The compressed catalogue contains `halo_name`, the zero-based `hdf5_row`,
component, and the exact HDF5 values of `Js`, `theta_s`, `D_Earth`, `Vmax`,
`r_s`, `rho_s`, `Cv`, `Xearth`, `Yearth`, and `Zearth`.  The one-row summary is
written as:

    retained_jsmax_<scenario>_repop_XXXX_nside<NSIDE>.csv

It records the brightest retained object, counts, cuts, provenance, source
paths, and the retained-catalogue SHA-256.  Both files must exist and validate
before a case is skipped on restart.  Each file is published atomically, with
the summary written last as the completion checkpoint.

Example for the audited initial Lattes case:

    python3 -u diagnostics/catalogue/scripts/scan_retained_subhalo_jsmax.py \
        fragile \
        --start-id 0 \
        --end-id 0 \
        --nside 2048 \
        --catalog-root /path/to/Auriga_outputs_hdf5_v2 \
        --clumpy-root /path/to/auriga-clumpy-pipeline/outputs/clumpy \
        --output-dir /path/to/diagnostic_outputs/retained_jsmax \
        --production-git-commit 07f693821e70f61f5f9bf86405896e7a613d8130

Use `scenario=both` for both disruption scenarios.  Inclusive repopulation
ranges are selected with `--start-id` and `--end-id`.  A rerun validates and
skips an existing case CSV unless `--overwrite` is supplied.  An optional
`--merge-output PATH` deterministically merges only the cases selected by the
current command.

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
