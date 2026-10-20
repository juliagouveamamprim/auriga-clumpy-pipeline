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

## Coarse all-sky subhalo-to-smooth ratio

`scripts/map_coarse_subhalo_to_smooth.py` preserves the two-dimensional sky
dependence by aggregating the final integrated-J maps into coarser NESTED
HEALPix pixels. For every diagnostic pixel `b`, it computes

    J_smooth,b = sum_b Jsmooth
    J_sub,b = sum_b (Jpointlike + Jextended_occ)
    R_b = J_sub,b / J_smooth,b.

`Jextended_occ` differs from `Jextended` only where the latter is exactly the
float32 CLUMPY empty-Jlist sentinel `np.float32(1e-40)`; those values become
zero. No threshold is applied to other positive values. The ratio is formed
only after the two components have been summed, and `hp.ud_grade` is not used.

The reader validates the `JFACTOR` table and streams it once in chunks. For a
NESTED input map, fine pixel `p` belongs to coarse pixel

    p // (NSIDE_input / NSIDE_diag)^2.

Both components are accumulated independently in float64, and their global
fine/coarse sums must agree. A non-positive coarse `Jsmooth` is an error. At
diagnostic NSIDE=16, a zero-substructure pixel means that the aggregated
substructure contribution in the generated map is zero, not that its FITS
data are missing or invalid. For positive `J_smooth`, such a pixel has `R=0`
and therefore no finite `log10(R)`; the plotting routine masks it instead of
assigning an artificial finite value.

For illustrative fragile/resilient comparisons, matched repopulation IDs
were selected only when both scenarios had zero empty coarse pixels. Of the
31 matched pairs examined, 17 met this occupancy criterion. This selection is
intended for visualization and is not an unbiased or statistically
representative subset of the full repopulation ensemble.

Run the initial matched example from the repository root with:

    python3 -u diagnostics/maps/scripts/map_coarse_subhalo_to_smooth.py \
        --repop-id 0 \
        --input-nside 2048 \
        --diag-nside 16

By default this reads the fragile and resilient final maps under
`outputs/clumpy`, then writes a shared-scale two-panel Galactic Mollweide
figure (PNG and PDF) and a numerical summary CSV under

    outputs/diagnostics/maps/coarse_subhalo_to_smooth/repop_0000/

Use `--fragile-map` and `--resilient-map` to override either input, and
`--output-dir`, `--formats`, or `--basename` to configure generated products.
