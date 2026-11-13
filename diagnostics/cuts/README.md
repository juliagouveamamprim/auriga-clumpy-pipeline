# Subhalo cut diagnostics

Catalogue-wise diagnostics used to select pointlike and extended subhalo cuts for the NSIDE=2048 all-sky templates.

## Conservative pixel-level scan

The script `scripts/scan_central_pixel_cuts.py` builds combined HEALPix
proxy maps using:

- `Js` in the source pixel for pointlike halos;
- the projected, CLUMPY-like central-pixel proxy for extended halos.

For extended halos, the proxy uses the circular integration aperture

    alpha_int = hp.max_pixrad(NSIDE)

and the exact projected NFW-squared fraction for a profile truncated at
`r_s`, followed by the CLUMPY aperture-to-pixel area rescaling. See
`../central_pixel_validation/README.md` for the complete derivation and
validation.

For each pair `(pointlike_f, extended_f)`, the cuts are defined by:

    J_cut = f * J_pixel_ref

Here, `J_pixel_ref` is the maximum individual pointlike or extended
central-pixel proxy.

The discarded pointlike contribution is assigned to the source pixel.
For each discarded extended halo, its central-pixel proxy is repeated
in every HEALPix pixel intersected by a disc of radius `theta_s`.
Contributions from overlapping halos are summed pixel by pixel.

The primary pixel-level validation metric is

    max(conservative discarded proxy map) / J_pixel_ref

It is stored as
`ratio_max_discarded_theta_s_envelope_to_j_pixel_ref`. The denominator
is computed from the complete catalogue before applying any cuts, as
the maximum between the brightest pointlike `Js` and the brightest
extended central-pixel proxy.

When reading legacy CSVs without this ratio column, the aggregation and
plotting scripts derive it in memory from `j_pixel_ref` and
`max_discarded_combined_theta_s_envelope_pixel`. The input CSVs are not
modified, and non-finite or non-positive reference values are rejected.

The output also stores the separately normalized peak discarded pointlike
map and extended `theta_s` envelope map. These component peaks are useful
diagnostics, but their maxima can occur in different pixels and therefore
must not be added to reconstruct the combined peak.

The `theta-s` envelope is always applied. It is an intentionally extreme
stress test: `theta_s` is a characteristic angular scale, not a physical
outer boundary. The resulting map is not a physical rendered map and
must not be used for integrated or cumulative J-factor quantities.

Separately from the proxy maps, the output records the discarded
catalogue-integrated fraction

    sum(Js_discarded) / sum(Js_full)

using the original `Js` values for both pointlike and extended halos.
Neither the extended central-pixel proxy nor its `theta-s` envelope
enters this integrated quantity. The output also records the numbers of
pointlike and extended halos kept after each cut.

## Scope

This diagnostic does not render the radial surface-brightness profiles
of extended halos. The `theta-s` envelope provides a conservative pixel-wise bound for
choosing cut candidates before a smaller number of full CLUMPY tests.

Large HDF5 catalogues, FITS maps, temporary outputs, and full scan
results should not be committed.

## Relative-J cumulative distribution

The script `scripts/scan_relative_js_cumulative.py` characterizes how
faint the subhalo population is relative to the brightest object in
each catalogue. For every catalogue, it computes the cumulative
distribution of

    Js / Js,max

where `Js,max` is evaluated independently for each repopulation. The
catalogues are read in chunks and are never concatenated in memory.
The saved NPZ contains one CDF per catalogue together with the
repopulation IDs, catalogue sizes, maximum `log10(Js)`, and invalid or
out-of-range counts. The `max_vmax_km_s` field records the applied selection;
it is `NaN` for the unfiltered full-population scan.

This is a descriptive population diagnostic. It motivates the need for
catalogue reduction but does not select or validate a particular value
of `f`.

Example for scanning 500 repopulations in both disruption scenarios:

    python -u diagnostics/cuts/scripts/scan_relative_js_cumulative.py \
        --input-root /path/to/Auriga_outputs_hdf5_v2 \
        --repop-start 0 \
        --n-repops 500 \
        --scenarios fragile,resilient \
        --output-npz outputs/diagnostics/cuts/relative_js_cumulative/relative_js_cumulative_0500repops.npz

The script `scripts/plot_relative_js_cumulative.py` reads the saved
NPZ without reopening the HDF5 catalogues. It plots the median CDF and
the 16th–84th percentile interval across repopulations for each
scenario. The dotted vertical lines mark the relative `Js` at which
the median CDF reaches 99%. The 50%, 90%,
and 99% thresholds are also printed to the terminal.

Example:

    python diagnostics/cuts/scripts/plot_relative_js_cumulative.py \
        --input-npz outputs/diagnostics/cuts/relative_js_cumulative/relative_js_cumulative_0500repops.npz \
        --output-dir diagnostics/cuts/plots \
        --formats png,pdf

For the dark-only Figure 7 population, select the inclusive
`Vmax <= 8 km/s` sample while scanning. The catalogue-wise `Js,max` and CDF
normalization are then evaluated after this selection. These commands replace
the original Figure 7 cache and figures in place:

    python -u diagnostics/cuts/scripts/scan_relative_js_cumulative.py \
        --input-root /dados5/julia/Auriga_outputs_hdf5_v2 \
        --repop-start 0 \
        --n-repops 500 \
        --scenarios fragile,resilient \
        --max-vmax-km-s 8 \
        --output-npz outputs/diagnostics/cuts/relative_js_cumulative/relative_js_cumulative_0500repops.npz

    python diagnostics/cuts/scripts/plot_relative_js_cumulative.py \
        --input-npz outputs/diagnostics/cuts/relative_js_cumulative/relative_js_cumulative_0500repops.npz \
        --output-dir diagnostics/cuts/plots \
        --formats png,pdf

The generated NPZ files and figures should not be committed.

## Discarded-Js distribution

The script `scripts/scan_discarded_js_distribution.py` reapplies the
pointlike and extended catalogue cuts and saves one compact histogram of
`Js / Js,max` per catalogue and population (`all`, `pointlike`, and
`extended`). `Js,max` is computed in a preliminary chunked pass from the
production-valid population before numerical brightness cuts. With
`--max-vmax-km-s`, the same inclusive Vmax selection is applied before
`Js,max`, `J_pixel_ref`, morphology, and the brightness cuts. Catalogues are
never combined before their relative `Js` values are calculated. The default
range and resolution match the cumulative diagnostic (`-16 <= log10(j_rel)
<= 0`, 320 bins), and underflow, overflow, invalid values, totals, maxima,
and source provenance are stored explicitly. The NPZ is checkpointed
atomically after every catalogue and is resumed automatically.

    python -u diagnostics/cuts/scripts/scan_discarded_js_distribution.py \
        --input-root /path/to/Auriga_outputs_hdf5_v2 \
        --output-npz outputs/diagnostics/cuts/discarded_js_distribution.npz

For the dark-only Figure 8 population and the same 50 repopulations used by
Figure 9:

    python -u diagnostics/cuts/scripts/scan_discarded_js_distribution.py \
        --input-root /dados5/julia/Auriga_outputs_hdf5_v2 \
        --output-npz outputs/diagnostics/cuts/discarded_js_distribution_dark_vmax_le_8kms_0050repops.npz \
        --repop-start 0 \
        --n-repops 50 \
        --scenarios fragile,resilient \
        --nside 2048 \
        --f 1e-3 \
        --max-vmax-km-s 8 \
        --chunk-size 500000

The companion `scripts/plot_discarded_js_distribution.py` first sums pairs of
adjacent 0.05-dex bins by default, producing 0.10-dex steps; this can be
configured with `--rebin-factor`. It then normalizes every catalogue before
calculating the percentage of discarded subhalos in each rebinned interval
(`100 * n_bin / n_discarded`). The arithmetic mean and pointwise 16th--84th
percentiles are calculated across repopulations after this normalization.
The shaded step areas show those percentile intervals, including
zero-percentage catalogues in every bin. Individual catalogue histograms sum
to 100%; the pointwise percentile curves are not expected to. By default,
`--population combined` produces one figure with pointlike subhalos in the
upper panel and extended subhalos in the lower panel. The compatibility and
diagnostic modes `--population all`, `pointlike`, and `extended` remain
available; PNG and PDF are produced by default.
The displayed x range is limited to
`1e-12 <= Js / Js,max^cat <= 1e-2`; this changes only the view and does not
truncate the stored histograms or their normalization.

Suggested figure caption: The upper panel shows the discarded pointlike
subhalos and the lower panel shows the discarded extended subhalos. Each
histogram is normalized separately by the total number of discarded subhalos
in its respective population and catalogue. Solid lines show the arithmetic
means across the 50 repopulations of each scenario. The shaded bands are
bin-wise 16--84 percentile intervals calculated between the repopulations;
they are pointwise summaries and do not represent individual catalogue
histograms.

    python diagnostics/cuts/scripts/plot_discarded_js_distribution.py \
        --input-npz outputs/diagnostics/cuts/discarded_js_distribution.npz \
        --output-dir diagnostics/cuts/plots \
        --population combined

Both the scanner and plotter print the maximum discarded `Js / Js,max` for
each scenario and for the pointlike and extended populations, including the
repopulation ID. The exact per-catalogue maxima remain stored in the NPZ.

## Multiple-repopulation scan

The script `scripts/scan_multiple_repops.py` runs the individual cut
diagnostic over a configurable range of repopulations and aggregates the
results.

The repopulation range is selected with:

    --repop-start 0
    --n-repops 10

This example evaluates `repop_0000` through `repop_0009`.

The scenarios can be selected independently or together:

    --scenarios fragile,resilient

Each repopulation and scenario still produces an individual CSV and log.
Existing compatible individual CSV files are reused by default. Use
`--overwrite` to recompute them.

After all scans, the driver produces:

- a combined CSV containing every individual result;
- a summary CSV grouped by scenario, NSIDE, CLUMPY integration aperture, envelope
  mode, and pair of cut fractions.

For each numerical diagnostic, the summary records:

- mean;
- sample standard deviation;
- median;
- minimum;
- maximum.

For the pixel-impact ratios, the summary also records the repopulation
that produced the maximum value. The mean describes the typical
behaviour, while the maximum across repopulations should be checked when
deciding whether a cut satisfies the adopted tolerance.

Example for scanning ten repopulations with
`pointlike_f, extended_f = 1e-2, 1e-3, 1e-4, 1e-5`:

    python -u diagnostics/cuts/scripts/scan_multiple_repops.py \
        --repop-start 0 \
        --n-repops 10 \
        --scenarios fragile,resilient \
        --input-root /path/to/Auriga_outputs_hdf5_v2 \
        --nside 2048 \
        --pointlike-f-values 1e-2,1e-3,1e-4,1e-5 \
        --extended-f-values 1e-2,1e-3,1e-4,1e-5

Use `--aggregate-only` to regenerate the combined and summary CSV files
without rerunning the individual scans.

## Dark-only cut revalidation

Both scan scripts accept an optional inclusive physical selection:

    --max-vmax-km-s 8

When enabled, each HDF5 chunk first receives the unchanged production
validity mask, and otherwise-valid objects with `Vmax > 8 km/s` are removed.
The remaining dark population is then classified as pointlike or extended.
The first pass computes `J_pixel_ref` from that population only. The second
pass constructs the uncut reference maps from every valid dark object and the
discarded maps from the subset that fails the brightness cuts. Consequently,
dark objects below a brightness threshold remain in the reference population
but not in the cut population. Objects above the Vmax ceiling enter neither.

The map methodology is unchanged. Pointlike objects contribute `Js` to their
NESTED source pixel. Extended objects contribute the projected central-pixel
proxy to their source pixel in the reference and cut representations. For the
conservative impact metric, the central-pixel proxy of every discarded
extended object is repeated in all NESTED pixels intersecting its `theta_s`
disc. Overlapping pointlike and extended discarded contributions are summed,
and the maximum accumulated pixel is divided by the same dark-only
`J_pixel_ref`.

Dark outputs carry `population_selection=dark_vmax_le_8kms` and
`max_vmax_km_s=8`. Default result and log directories gain a
`dark_vmax_le_8kms/` subdirectory, and individual filenames contain the same
tag. Existing full-population defaults and legacy CSV reuse remain unchanged.
The summary CSV records mean, sample standard deviation, median, minimum,
maximum, 16th, 84th, and 95th percentiles for pixel-impact ratios, together
with the count strictly above 1% and the repopulation producing the maximum.

The repository retains the 50-repopulation plot but not its source combined
CSV or exact launch command. The multi-repopulation driver selects consecutive
IDs, and the documented validation convention starts at zero; the directly
comparable sample is therefore `repop_0000` through `repop_0049`. This should
be confirmed against any external archived job record before the 50-catalogue
run.

### Lattes pilot: one repopulation per scenario

Run this first and review both individual CSVs, the combined CSV, and summary:

```bash
python -u diagnostics/cuts/scripts/scan_multiple_repops.py \
    --repop-start 0 \
    --n-repops 1 \
    --scenarios fragile,resilient \
    --input-root /dados5/julia/Auriga_outputs_hdf5_v2 \
    --nside 2048 \
    --pointlike-f-values 1e-3 \
    --extended-f-values 1e-3 \
    --max-vmax-km-s 8 \
    --chunk-size 500000 \
    --results-dir /dados5/julia/Auriga_dark_cut_validation/pilot_repop_0000/results \
    --logs-dir /dados5/julia/Auriga_dark_cut_validation/pilot_repop_0000/logs
```

### Lattes 50-repopulation run after pilot approval

```bash
python -u diagnostics/cuts/scripts/scan_multiple_repops.py \
    --repop-start 0 \
    --n-repops 50 \
    --scenarios fragile,resilient \
    --input-root /dados5/julia/Auriga_outputs_hdf5_v2 \
    --nside 2048 \
    --pointlike-f-values 1e-3 \
    --extended-f-values 1e-3 \
    --max-vmax-km-s 8 \
    --chunk-size 500000 \
    --results-dir /dados5/julia/Auriga_dark_cut_validation/all_50_repops/results \
    --logs-dir /dados5/julia/Auriga_dark_cut_validation/all_50_repops/logs
```

These commands do not invoke CLUMPY. The driver refuses to reuse an
individual CSV whose population tag or Vmax ceiling differs. Do not add
`--overwrite` when preserving an existing diagnostic run is required.

### Methodological limitations retained intentionally

- At NSIDE 2048, every float64 map contains 50,331,648 pixels. With one
  pointlike and one extended fraction, the unchanged implementation estimates
  roughly 2.25 GiB of persistent map arrays at peak, in addition to temporary
  arrays and catalogue chunks. Lattes jobs need appropriate memory headroom.
- The extended `theta_s` envelope is deliberately conservative: `theta_s` is
  a characteristic scale, not a truncation boundary, and a constant
  central-pixel proxy is repeated over the footprint. It is not a physical
  rendered extended-halo map.
- The metric is a maximum accumulated discarded-proxy pixel divided by
  `J_pixel_ref`. Catalogue-integrated `Js` fractions remain secondary
  bookkeeping and are not substitutes for this map-level test.
- Separate pointlike and extended peaks need not be co-located. The combined
  accumulated map, rather than the sum of separate maxima, is authoritative.

## Multiple-repopulation plots

The script `scripts/plot_multi_repop_cut_diagnostics.py` reads the
combined CSV produced by the multiple-repopulation scan and generates
one figure for each disruption scenario.

Only symmetric cuts along the diagonal

    pointlike_f = extended_f = f

are shown.

The upper panel presents the conservative combined-map impact from the
`theta-s` envelope:

    max(conservative discarded proxy map) / J_pixel_ref

Individual repopulations are shown together with the mean and the
16th–84th percentile interval. The dashed 1% reference line is retained.

The lower panel shows the mean retained catalogue fraction separately
for pointlike and extended subhalos. The annotations report the mean
numbers of kept (`K`) and discarded (`D`) objects.

Example:

    python diagnostics/cuts/scripts/plot_multi_repop_cut_diagnostics.py \
        --combined-csv multi_repop_nside2048_combined.csv \
        --output-dir plots \
        --formats png,pdf

The generated figures and the large combined scan tables should not be
committed to the repository.

## Adopted NSIDE=2048 cuts

The production configuration uses symmetric pointlike and extended cuts:

    pointlike_f = extended_f = 1e-3

Both fractions are applied to the same `J_pixel_ref` defined above.
Pointlike halos are compared using `Js`, while extended halos are compared
using the projected central-pixel proxy described above.
