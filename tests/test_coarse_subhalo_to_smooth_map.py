import importlib.util
import sys
from pathlib import Path

import healpy as hp
import matplotlib.pyplot as plt
import numpy as np
import pytest
from astropy.io import fits


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = (
    REPOSITORY_ROOT
    / "diagnostics"
    / "maps"
    / "scripts"
    / "map_coarse_subhalo_to_smooth.py"
)


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


diagnostic = load_module("map_coarse_subhalo_to_smooth", SCRIPT_PATH)


def write_final_map(
    path,
    smooth,
    pointlike,
    extended,
    nside,
    *,
    pixels=None,
    ordering="NESTED",
    coordsys="G",
    unit="GeV^2 cm^-5",
):
    npix = hp.nside2npix(nside)
    if pixels is None:
        pixels = np.arange(npix, dtype=np.int32)
    hdu = fits.BinTableHDU.from_columns(
        [
            fits.Column(
                name="PIXEL",
                format="1J",
                array=np.asarray(pixels, dtype=np.int32),
            ),
            fits.Column(
                name="Jsmooth",
                format="1E",
                unit=unit,
                array=np.asarray(smooth, dtype=np.float32),
            ),
            fits.Column(
                name="Jpointlike",
                format="1E",
                unit=unit,
                array=np.asarray(pointlike, dtype=np.float32),
            ),
            fits.Column(
                name="Jextended",
                format="1E",
                unit=unit,
                array=np.asarray(extended, dtype=np.float32),
            ),
        ],
        name="JFACTOR",
    )
    hdu.header["PIXTYPE"] = "HEALPIX"
    hdu.header["ORDERING"] = ordering
    hdu.header["NSIDE"] = nside
    hdu.header["INDXSCHM"] = "EXPLICIT"
    hdu.header["COORDSYS"] = coordsys
    hdu.header["FIRSTPIX"] = 0
    hdu.header["LASTPIX"] = npix - 1
    fits.HDUList([fits.PrimaryHDU(), hdu]).writeto(path)
    return path


def aggregate(path, *, fine_nside=4, coarse_nside=2, chunk_size=7):
    return diagnostic.aggregate_map(
        path=path,
        scenario="fragile",
        repop_id=0,
        input_nside=fine_nside,
        diagnostic_nside=coarse_nside,
        chunk_size=chunk_size,
    )


def test_nested_parent_mapping_agrees_with_healpy_angles():
    fine_nside = 8
    coarse_nside = 2
    pixels = np.arange(hp.nside2npix(fine_nside), dtype=np.int64)

    parents = diagnostic.nested_parent_pixels(pixels, fine_nside, coarse_nside)
    lon, lat = hp.pix2ang(fine_nside, pixels, nest=True, lonlat=True)
    angular_parents = hp.ang2pix(coarse_nside, lon, lat, nest=True, lonlat=True)

    np.testing.assert_array_equal(parents, angular_parents)
    np.testing.assert_array_equal(
        parents,
        pixels // (fine_nside // coarse_nside) ** 2,
    )


def test_aggregation_conserves_integrated_j_and_matches_occupancy_reference(
    tmp_path,
):
    fine_nside = 4
    coarse_nside = 2
    npix = hp.nside2npix(fine_nside)
    smooth = ((np.arange(npix) % 11) + 1).astype(np.float32)
    pointlike = ((np.arange(npix) % 5) * 0.25).astype(np.float32)
    extended = ((np.arange(npix) % 7) * 0.125).astype(np.float32)
    extended[::13] = diagnostic.CLUMPY_EMPTY_JLIST
    path = write_final_map(
        tmp_path / "components.fits",
        smooth,
        pointlike,
        extended,
        fine_nside,
    )

    result = aggregate(path, chunk_size=17)
    children = (fine_nside // coarse_nside) ** 2
    extended_occ = extended.astype(np.float64)
    extended_occ[extended == diagnostic.CLUMPY_EMPTY_JLIST] = 0.0
    fine_sub = pointlike.astype(np.float64) + extended_occ
    reference_smooth = smooth.astype(np.float64).reshape(-1, children).sum(1)
    # This is the independent reshape equivalent of the exploratory
    # occupancy script's NESTED parent bincount.
    reference_sub = fine_sub.reshape(-1, children).sum(1)

    np.testing.assert_allclose(result.j_smooth, reference_smooth, rtol=0)
    np.testing.assert_allclose(result.j_sub, reference_sub, rtol=0)
    assert result.j_smooth.sum() == pytest.approx(
        smooth.astype(np.float64).sum(), rel=1e-15
    )
    assert result.j_sub.sum() == pytest.approx(fine_sub.sum(), rel=1e-15)
    assert result.n_extended_sentinel_pixels == np.count_nonzero(
        extended == diagnostic.CLUMPY_EMPTY_JLIST
    )


def test_ratio_is_ratio_of_sums_not_mean_of_pixel_ratios(tmp_path):
    fine_nside = 2
    npix = hp.nside2npix(fine_nside)
    smooth = np.ones(npix, dtype=np.float32)
    pointlike = np.ones(npix, dtype=np.float32)
    smooth[:4] = [1.0, 100.0, 1.0, 1.0]
    pointlike[:4] = [100.0, 1.0, 1.0, 1.0]
    extended = np.full(npix, diagnostic.CLUMPY_EMPTY_JLIST)
    path = write_final_map(
        tmp_path / "ratio.fits",
        smooth,
        pointlike,
        extended,
        fine_nside,
    )

    result = aggregate(path, fine_nside=fine_nside, coarse_nside=1, chunk_size=5)
    expected_ratio = pointlike[:4].sum(dtype=np.float64) / smooth[:4].sum(
        dtype=np.float64
    )
    mean_pixel_ratio = np.mean(pointlike[:4] / smooth[:4])

    assert result.ratio[0] == pytest.approx(expected_ratio)
    assert result.ratio[0] != pytest.approx(mean_pixel_ratio)


def test_only_exact_float32_clumpy_sentinel_is_zeroed(tmp_path):
    nside = 1
    npix = hp.nside2npix(nside)
    sentinel = diagnostic.CLUMPY_EMPTY_JLIST
    adjacent = np.nextafter(sentinel, np.float32(np.inf), dtype=np.float32)
    extended = np.full(npix, sentinel, dtype=np.float32)
    extended[1] = adjacent
    path = write_final_map(
        tmp_path / "sentinel.fits",
        np.ones(npix),
        np.zeros(npix),
        extended,
        nside,
    )

    result = aggregate(path, fine_nside=nside, coarse_nside=nside, chunk_size=3)

    assert result.n_extended_sentinel_pixels == npix - 1
    assert result.j_sub[0] == 0.0
    assert result.j_sub[1] == float(adjacent)
    assert result.j_sub.sum() == float(adjacent)


def test_non_positive_coarse_smooth_denominator_is_rejected(tmp_path):
    fine_nside = 2
    npix = hp.nside2npix(fine_nside)
    smooth = np.ones(npix, dtype=np.float32)
    smooth[:4] = 0.0
    path = write_final_map(
        tmp_path / "zero_denominator.fits",
        smooth,
        np.ones(npix),
        np.zeros(npix),
        fine_nside,
    )

    with pytest.raises(ValueError, match="non-positive.*Jsmooth"):
        aggregate(
            path,
            fine_nside=fine_nside,
            coarse_nside=1,
            chunk_size=11,
        )


@pytest.mark.parametrize(
    ("change", "message"),
    [("pixels", "complete ordered NESTED"), ("coordsys", "Galactic")],
)
def test_full_sky_galactic_nested_layout_is_enforced(tmp_path, change, message):
    nside = 2
    npix = hp.nside2npix(nside)
    pixels = np.arange(npix, dtype=np.int32)
    coordsys = "G"
    if change == "pixels":
        pixels[[1, 2]] = pixels[[2, 1]]
    else:
        coordsys = "C"
    path = write_final_map(
        tmp_path / f"bad_{change}.fits",
        np.ones(npix),
        np.ones(npix),
        np.ones(npix),
        nside,
        pixels=pixels,
        coordsys=coordsys,
    )

    with pytest.raises(ValueError, match=message):
        aggregate(path, fine_nside=nside, coarse_nside=1, chunk_size=6)


@pytest.mark.parametrize(
    ("column", "invalid"),
    [
        ("smooth", np.nan),
        ("pointlike", -1.0),
        ("extended", np.inf),
    ],
)
def test_invalid_component_values_are_rejected(tmp_path, column, invalid):
    nside = 1
    arrays = {
        "smooth": np.ones(12, dtype=np.float32),
        "pointlike": np.ones(12, dtype=np.float32),
        "extended": np.ones(12, dtype=np.float32),
    }
    arrays[column][3] = invalid
    path = write_final_map(
        tmp_path / f"bad_{column}.fits",
        arrays["smooth"],
        arrays["pointlike"],
        arrays["extended"],
        nside,
    )

    with pytest.raises(ValueError, match="non-finite|negative"):
        aggregate(path, fine_nside=nside, coarse_nside=nside, chunk_size=4)


def test_each_source_fits_is_opened_only_once(tmp_path, monkeypatch):
    values = np.ones(12, dtype=np.float32)
    path = write_final_map(
        tmp_path / "single_pass.fits", values, values, values, nside=1
    )
    opened = []
    original_open = diagnostic.fits.open

    def tracked_open(candidate, *args, **kwargs):
        opened.append(Path(candidate))
        return original_open(candidate, *args, **kwargs)

    monkeypatch.setattr(diagnostic.fits, "open", tracked_open)
    aggregate(path, fine_nside=1, coarse_nside=1, chunk_size=3)

    assert opened == [path.resolve()]


def test_summary_reports_empty_bins_and_substructure_dominance():
    coarse_map = diagnostic.CoarseRatioMap(
        scenario="resilient",
        repop_id=9,
        input_nside=2,
        diagnostic_nside=1,
        j_smooth=np.ones(12),
        j_sub=np.array([0.0, 0.5, 2.0] + [1.0] * 9),
        ratio=np.array([0.0, 0.5, 2.0] + [1.0] * 9),
        n_extended_sentinel_pixels=0,
    )

    summary = diagnostic.summarize_map(coarse_map)

    assert summary.n_bins == 12
    assert summary.n_zero_substructure_bins == 1
    assert summary.min_finite_R == 0.0
    assert summary.max_finite_R == 2.0
    assert summary.n_bins_R_gt_1 == 1
    assert summary.fraction_bins_R_gt_1 == pytest.approx(1 / 12)


def test_two_panel_figure_uses_shared_fixed_two_slope_norm(tmp_path):
    base = dict(
        repop_id=0,
        input_nside=1,
        diagnostic_nside=1,
        j_smooth=np.ones(12),
        n_extended_sentinel_pixels=0,
    )
    maps = {
        "fragile": diagnostic.CoarseRatioMap(
            scenario="fragile",
            j_sub=np.full(12, 1.0e-4),
            ratio=np.full(12, 1.0e-4),
            **base,
        ),
        "resilient": diagnostic.CoarseRatioMap(
            scenario="resilient",
            j_sub=np.full(12, 10.0),
            ratio=np.full(12, 10.0),
            **base,
        ),
    }
    original_ratios = {
        scenario: coarse_map.ratio.copy() for scenario, coarse_map in maps.items()
    }

    figure = diagnostic.make_figure(maps, repop_id=0, diagnostic_nside=1)
    output = tmp_path / "two_panel.png"
    figure.savefig(output)
    plt.close(figure)

    assert output.is_file()
    map_axes = [axis for axis in figure.axes if axis.images]
    assert len(map_axes) == 2
    norms = [axis.images[0].norm for axis in map_axes]
    assert all(isinstance(norm, diagnostic.TwoSlopeNorm) for norm in norms)
    assert norms[0] is norms[1]
    assert [axis.images[0].get_clim() for axis in map_axes] == [
        (-3.0, 1.5),
        (-3.0, 1.5),
    ]
    assert norms[0](0.0) == pytest.approx(0.5)

    colorbar_axes = [axis for axis in figure.axes if hasattr(axis, "_colorbar")]
    assert len(colorbar_axes) == 1
    colorbar = colorbar_axes[0]._colorbar
    assert colorbar.mappable.norm is norms[0]
    assert colorbar.extend == "min"
    np.testing.assert_allclose(colorbar.get_ticks(), diagnostic.COLORBAR_TICKS)

    cmap = map_axes[0].images[0].cmap
    np.testing.assert_allclose(cmap(norms[0](0.0)), cmap(0.5))
    plotted_fragile = map_axes[0].images[0].get_array()
    assert np.min(plotted_fragile.compressed()) == pytest.approx(-3.0)
    for scenario, ratio in original_ratios.items():
        np.testing.assert_array_equal(maps[scenario].ratio, ratio)


def test_cli_paths_and_nested_nside_validation():
    args = diagnostic.parse_args(["--repop-id", "7"])

    assert diagnostic.resolve_map_paths(args)["fragile"] == Path(
        "outputs/clumpy/fragile/outputs/total/repop_0007_nside2048/"
        "auriga_total_nside2048.fits"
    )
    assert diagnostic.default_output_dir(7) == Path(
        "outputs/diagnostics/maps/coarse_subhalo_to_smooth/repop_0007"
    )

    args.input_nside = 3
    with pytest.raises(ValueError, match="Invalid input NSIDE"):
        diagnostic.validate_cli_args(args)
