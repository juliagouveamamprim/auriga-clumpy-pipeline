# HEALPix map diagnostics

## Annular subhalo-to-smooth J-factor ratio

`scripts/scan_annular_subhalo_to_smooth.py` compares the summed subhalo and
smooth emission in annuli about the Galactic centre. For each pixel,

    psi_GC = arccos(n_pixel dot n_GC),

where `n_GC` points to Galactic coordinates `(l, b) = (0 deg, 0 deg)`. Here
`J_sub = Jpointlike + Jextended` is the **rendered subhalo component**: the
subhalos actually retained and rendered in the maps after any configured
catalogue cuts. In an annulus `A`, the saved quantity is

    R_ann(A) = sum_A (J_pointlike + J_extended) / sum_A J_smooth.

This is the ratio of the annular sums, equivalently a pixel-wise ratio weighted
by `J_smooth`. It is not the unweighted mean of pixel-wise ratios. The total
map is not used to construct the numerator.

The pipeline does not add the contribution of subhalos rejected by the
catalogue cuts back into either component. The preparation step applies the
extended and pointlike masks before writing the CLUMPY list and pointlike map;
the final combinator only copies the rendered CLUMPY `Jlist` to `Jextended`
and adds the already-filtered `Jpointlike`. No unresolved-subhalo or discarded-
population boost is added by the combinator. The CLUMPY templates used by this
pipeline also disable statistical Milky Way substructure
(`gMW_SUBS_N_INM1M2=0`) and substructure inside listed external halos
(`gDSPH/GALAXY/CLUSTER_SUBS_MASSFRACTION=0`). Therefore this diagnostic must
not be interpreted as the contribution of the complete pre-cut subhalo
population or as including an unresolved-subhalo boost. The smooth-host
renormalization is computed independently of the catalogue cuts; it does not
restore the discarded subhalos' J-factor emission.

The reference levels in the plot mean:

- `R_ann = 0.01`: subhalos contribute 1% of the smooth annular J-factor;
- `R_ann = 0.1`: subhalos contribute 10% of the smooth annular J-factor;
- `R_ann = 1`: subhalo and smooth annular J-factors are equal.

### Pipeline map format

The final product written by `scripts/combine_clumpy_pointlike.py` is

    outputs/clumpy/<scenario>/outputs/total/repop_XXXX_nside<NSIDE>/
        auriga_total_nside<NSIDE>.fits

It contains a primary HDU and two binary-table HDUs:

- `JFACTOR`: `PIXEL`, `Jtot`, `Jsmooth`, `Jextended`, `Jpointlike`, in
  `GeV^2 cm^-5`;
- `JFACTOR_PER_SR`: `PIXEL`, `Jtot_per_sr`, `Jsmooth_per_sr`,
  `Jextended_per_sr`, `Jpointlike_per_sr`, in `GeV^2 cm^-5 sr^-1`.

The diagnostic reads the three separate component columns from `JFACTOR`.
The tables use explicit pixel indices (`INDXSCHM=EXPLICIT`), Galactic
coordinates (`COORDSYS=G`), and `ORDERING=NESTED`. NSIDE is configurable in
the pipeline and is encoded in both the header and path; the current pipeline
default is `NSIDE=2048`, corresponding to `12 * NSIDE^2 = 50,331,648` rows.

The corrected CLUMPY intermediate contains `Jsmooth` and the extended
component under its upstream name `Jlist`; the standalone pointlike FITS
contains `Jpointlike`. The combine step renames `Jlist` to `Jextended` and
writes all separate components into the final file. Consequently no
`Jtotal - Jsmooth` fallback is needed for this diagnostic.

### Scan on Lattes

Run from the repository root. This exact example scans repopulation 0 at the
pipeline default NSIDE:

    python3 -u diagnostics/maps/scripts/scan_annular_subhalo_to_smooth.py \
        --repop-id 0 \
        --fragile-map outputs/clumpy/fragile/outputs/total/repop_0000_nside2048/auriga_total_nside2048.fits \
        --resilient-map outputs/clumpy/resilient/outputs/total/repop_0000_nside2048/auriga_total_nside2048.fits \
        --output-dir outputs/diagnostics/maps/annular_subhalo_to_smooth/repop_0000 \
        --bin-width-deg 1

The reader opens each scenario FITS once, memory-maps it, and processes rows in
chunks. It checks NSIDE, row count, ordering, pixel indices, coordinate system,
and unit compatibility before combining components. Use `--chunk-size` to tune
the default chunk of one million pixels.

### Plot from the CSV

The plotting script never reopens FITS files. It creates PNG and PDF by
default:

    python3 diagnostics/maps/scripts/plot_annular_subhalo_to_smooth.py \
        --input-csv outputs/diagnostics/maps/annular_subhalo_to_smooth/repop_0000/annular_subhalo_to_smooth_repop_0000.csv \
        --output-dir outputs/diagnostics/maps/annular_subhalo_to_smooth/repop_0000

Diagnostic CSVs and figures are generated products and should not be
committed.
