# Dwarf-exclusion catalogue diagnostic

`scripts/analyze_dwarf_exclusion.py` compares the existing catalogue cuts with
a selection in which `Vmax > 8 km/s` objects are removed before the common
brightness reference is calculated.  It is read-only and independent of
CLUMPY and FITS products.

The diagnostic imports these definitions directly from
`scripts/prepare_subhalo_components.py`:

- the production validity mask;
- the NSIDE-derived angular threshold and its conservative rounding;
- the corrected-CLUMPY central-pixel proxy for extended objects.

Its scientific constants are fixed to the requested production comparison:
NSIDE 2048, point-like and extended cut fractions `1e-3`, and dark-subhalo
boundary `Vmax <= 8 km/s`.

Each catalogue is read twice in configurable chunks.  The first pass finds the
original and dark-only references and their defining objects.  The second
applies both sets of thresholds and measures retained-set differences.  No
full catalogue or HEALPix map is held in memory.

## Lattes pilot

From the repository root, run one repopulation of each scenario into new,
diagnostic-only directories:

```bash
python3 diagnostics/dwarf_exclusion/scripts/analyze_dwarf_exclusion.py \
  --input-root /dados5/julia/Auriga_outputs_hdf5_v2 \
  --output-dir outputs/diagnostics/dwarf_exclusion/pilot_repop_0000_fragile \
  --repop-id 0 \
  --scenarios fragile

python3 diagnostics/dwarf_exclusion/scripts/analyze_dwarf_exclusion.py \
  --input-root /dados5/julia/Auriga_outputs_hdf5_v2 \
  --output-dir outputs/diagnostics/dwarf_exclusion/pilot_repop_0000_resilient \
  --repop-id 0 \
  --scenarios resilient
```

The script refuses a non-empty output directory unless `--resume` is supplied.
Resume is checkpointed through the validated `per_catalogue.csv` file.

After pilot review, the corresponding all-catalogue invocation would be:

```bash
python3 diagnostics/dwarf_exclusion/scripts/analyze_dwarf_exclusion.py \
  --input-root /dados5/julia/Auriga_outputs_hdf5_v2 \
  --output-dir outputs/diagnostics/dwarf_exclusion/all_500 \
  --repop-start 0 \
  --repop-end 499 \
  --scenarios fragile resilient
```

Do not launch the all-500 command until the pilot results have been reviewed.

## Outputs

- `per_catalogue.csv`: references and defining objects, dwarf counts, retained
  counts for both selections, newly admitted and lost-set diagnostics, and
  catalogue-`Js` sums.
- `summary_by_scenario.csv`: ratio percentiles, change fractions, newly admitted
  count statistics, and total/average newly admitted catalogue `Js`.
- `jref_ratio_distribution.png`.
- `newly_admitted_extended_distribution.png`.
- `jref_ratio_vs_new_extended.png`.

The reported catalogue `Js` sums are bookkeeping diagnostics and are not
map-level impact estimates.
