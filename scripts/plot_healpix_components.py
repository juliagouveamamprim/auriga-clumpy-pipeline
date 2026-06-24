#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Plot smooth, extended, pointlike, and total Auriga J-factor maps."""

import argparse
from pathlib import Path

import healpy as hp
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
BASE_RUN_DIR = REPOSITORY_ROOT / "outputs" / "clumpy"
DEFAULT_NSIDE = 1024

COMPONENTS = {
    "smooth": ("Jsmooth_per_sr", "Smooth Milky Way"),
    "extended": ("Jextended_per_sr", "Extended subhalos"),
    "pointlike": ("Jpointlike_per_sr", "Pointlike subhalos"),
    "total": ("Jtot_per_sr", "Total"),
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot the components of one total Auriga HEALPix map."
    )
    parser.add_argument("repop_id", type=int)
    parser.add_argument(
        "scenario",
        choices=["resilient", "fragile"],
    )
    parser.add_argument(
        "--nside",
        type=int,
        default=DEFAULT_NSIDE,
    )
    parser.add_argument(
        "--lower-percentile",
        type=float,
        default=1.0,
        help="Lower percentile used for component-specific display scaling.",
    )
    parser.add_argument(
        "--upper-percentile",
        type=float,
        default=99.9,
        help="Upper percentile used for component-specific display scaling.",
    )
    parser.add_argument(
        "--floor",
        type=float,
        default=1e-35,
        help="Values at or below this floor are masked before log10.",
    )
    return parser.parse_args()


def get_paths(repop_id, scenario, nside):
    repop_tag = f"repop_{repop_id:04d}"

    input_fits = (
        BASE_RUN_DIR
        / scenario
        / "outputs"
        / "total"
        / repop_tag
        / f"auriga_total_nside{nside}.fits"
    )

    output_dir = (
        BASE_RUN_DIR
        / scenario
        / "plots"
        / repop_tag
    )

    return input_fits, output_dir


def load_components(path, nside):
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Total FITS not found: {path}")

    with fits.open(path, memmap=True) as hdul:
        hdu = hdul["JFACTOR_PER_SR"]

        expected = {
            "PIXTYPE": "HEALPIX",
            "ORDERING": "NESTED",
            "NSIDE": nside,
            "COORDSYS": "G",
            "INDXSCHM": "EXPLICIT",
        }

        for key, value in expected.items():
            actual = hdu.header.get(key)

            if actual != value:
                raise ValueError(
                    f"Expected {key}={value!r}, got {actual!r}."
                )

        pixels = np.asarray(hdu.data["PIXEL"], dtype=np.int64)
        expected_pixels = np.arange(hp.nside2npix(nside))

        if not np.array_equal(pixels, expected_pixels):
            raise ValueError(
                "The FITS PIXEL column is not the complete ordered "
                "NESTED pixel sequence."
            )

        maps = {
            key: np.asarray(hdu.data[column], dtype=np.float64)
            for key, (column, _) in COMPONENTS.items()
        }

    return maps


def prepare_log_map(values, floor):
    values = np.asarray(values, dtype=np.float64)

    valid = (
        np.isfinite(values)
        & (values > floor)
    )

    result = np.full(values.shape, hp.UNSEEN, dtype=np.float64)
    result[valid] = np.log10(values[valid])

    return result, valid


def display_limits(log_map, valid, lower_percentile, upper_percentile):
    values = log_map[valid]

    if values.size == 0:
        raise ValueError("No positive finite pixels available for plotting.")

    vmin, vmax = np.percentile(
        values,
        [lower_percentile, upper_percentile],
    )

    if not np.isfinite(vmin) or not np.isfinite(vmax):
        raise ValueError("Could not determine finite plotting limits.")

    if vmax <= vmin:
        padding = 0.5
        vmin -= padding
        vmax += padding

    return float(vmin), float(vmax)


def plot_one(
    log_map,
    title,
    output_path,
    vmin,
    vmax,
):
    plt.figure(figsize=(11, 6))

    hp.mollview(
        log_map,
        nest=True,
        coord="G",
        title=title,
        unit=r"$\log_{10}(J\,[\mathrm{GeV}^2\,\mathrm{cm}^{-5}\,\mathrm{sr}^{-1}])$",
        min=vmin,
        max=vmax,
        cmap="magma",
        badcolor="white",
        bgcolor="white",
        cbar=True,
        hold=True,
    )

    hp.graticule(
        dpar=30,
        dmer=30,
        alpha=0.25,
    )

    plt.savefig(
        output_path,
        dpi=220,
        bbox_inches="tight",
    )
    plt.close()


def plot_panel(prepared, limits, output_path):
    figure = plt.figure(figsize=(16, 10))

    for index, component in enumerate(COMPONENTS, start=1):
        _, title = COMPONENTS[component]
        log_map = prepared[component]
        vmin, vmax = limits[component]

        hp.mollview(
            log_map,
            fig=figure.number,
            sub=(2, 2, index),
            nest=True,
            coord="G",
            title=title,
            unit=r"$\log_{10}(J/\mathrm{sr})$",
            min=vmin,
            max=vmax,
            cmap="magma",
            badcolor="white",
            bgcolor="white",
            cbar=True,
            hold=True,
        )

        hp.graticule(
            dpar=30,
            dmer=30,
            alpha=0.25,
        )

    plt.savefig(
        output_path,
        dpi=220,
        bbox_inches="tight",
    )
    plt.close()


def main():
    args = parse_args()

    if args.repop_id < 0:
        raise ValueError("repop_id must be a non-negative integer.")

    if not 0.0 <= args.lower_percentile < args.upper_percentile <= 100.0:
        raise ValueError(
            "Percentiles must satisfy "
            "0 <= lower < upper <= 100."
        )

    input_fits, output_dir = get_paths(
        args.repop_id,
        args.scenario,
        args.nside,
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    maps = load_components(
        input_fits,
        args.nside,
    )

    prepared = {}
    limits = {}

    for component, values in maps.items():
        log_map, valid = prepare_log_map(
            values,
            args.floor,
        )

        vmin, vmax = display_limits(
            log_map,
            valid,
            args.lower_percentile,
            args.upper_percentile,
        )

        prepared[component] = log_map
        limits[component] = (vmin, vmax)

        _, title = COMPONENTS[component]

        output_path = (
            output_dir
            / f"{component}_mollview.png"
        )

        plot_one(
            log_map=log_map,
            title=title,
            output_path=output_path,
            vmin=vmin,
            vmax=vmax,
        )

        print(
            f"{component}: "
            f"vmin={vmin:.3f}, vmax={vmax:.3f}, "
            f"saved={output_path}"
        )

    panel_path = output_dir / "components_mollview.png"

    plot_panel(
        prepared=prepared,
        limits=limits,
        output_path=panel_path,
    )

    print(f"Panel saved: {panel_path}")


if __name__ == "__main__":
    main()
