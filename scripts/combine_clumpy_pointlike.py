#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Combine the corrected CLUMPY map with the pointlike subhalo map."""

import argparse
from pathlib import Path

import numpy as np
from astropy.io import fits


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
BASE_RUN_DIR = REPOSITORY_ROOT / "outputs" / "clumpy"

CLUMPY_BASENAME = "annihil_gal2D_LOS0_0_FOV360x180_nside1024"
DEFAULT_NSIDE = 1024


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Combine corrected CLUMPY smooth+extended emission with "
            "the Auriga pointlike subhalo HEALPix map."
        )
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
    return parser.parse_args()


def get_paths(repop_id, scenario, nside):
    repop_tag = f"repop_{repop_id:04d}"
    scenario_dir = BASE_RUN_DIR / scenario

    corrected_fits = (
        scenario_dir
        / "outputs"
        / "corrected_clumpy"
        / repop_tag
        / f"{CLUMPY_BASENAME}.fits"
    )

    pointlike_fits = (
        scenario_dir
        / "pointlike"
        / f"{repop_tag}_pointlike_nside{nside}.fits"
    )

    output_fits = (
        scenario_dir
        / "outputs"
        / "total"
        / repop_tag
        / f"auriga_total_nside{nside}.fits"
    )

    return corrected_fits, pointlike_fits, output_fits


def validate_healpix_hdu(hdu, nside, label):
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
                f"{label}: expected {key}={value!r}, got {actual!r}."
            )


def make_hdu(
    name,
    pixels,
    total,
    smooth,
    extended,
    pointlike,
    nside,
    per_sr,
):
    if per_sr:
        suffix = "_per_sr"
        unit = "GeV^2 cm^-5 sr^-1"
    else:
        suffix = ""
        unit = "GeV^2 cm^-5"

    columns = [
        fits.Column(
            name="PIXEL",
            format="1J",
            array=pixels,
        ),
        fits.Column(
            name=f"Jtot{suffix}",
            format="1E",
            unit=unit,
            array=total.astype(np.float32),
        ),
        fits.Column(
            name=f"Jsmooth{suffix}",
            format="1E",
            unit=unit,
            array=smooth.astype(np.float32),
        ),
        fits.Column(
            name=f"Jextended{suffix}",
            format="1E",
            unit=unit,
            array=extended.astype(np.float32),
        ),
        fits.Column(
            name=f"Jpointlike{suffix}",
            format="1E",
            unit=unit,
            array=pointlike.astype(np.float32),
        ),
    ]

    hdu = fits.BinTableHDU.from_columns(columns, name=name)

    npix = pixels.size
    pixel_area_sr = 4.0 * np.pi / npix

    hdu.header["PIXTYPE"] = "HEALPIX"
    hdu.header["ORDERING"] = "NESTED"
    hdu.header["NSIDE"] = nside
    hdu.header["FIRSTPIX"] = 0
    hdu.header["LASTPIX"] = npix - 1
    hdu.header["INDXSCHM"] = "EXPLICIT"
    hdu.header["COORDSYS"] = "G"
    hdu.header["OBJECT"] = "FULLSKY"
    hdu.header["PIXAREA"] = (
        pixel_area_sr,
        "HEALPix pixel solid angle [sr]",
    )

    return hdu


def combine_maps(corrected_fits, pointlike_fits, output_fits, nside):
    corrected_fits = Path(corrected_fits)
    pointlike_fits = Path(pointlike_fits)
    output_fits = Path(output_fits)

    if not corrected_fits.exists():
        raise FileNotFoundError(
            f"Corrected CLUMPY FITS not found: {corrected_fits}"
        )

    if not pointlike_fits.exists():
        raise FileNotFoundError(
            f"Pointlike FITS not found: {pointlike_fits}"
        )

    output_fits.parent.mkdir(parents=True, exist_ok=True)

    with (
        fits.open(corrected_fits, memmap=True) as clumpy,
        fits.open(pointlike_fits, memmap=True) as pointlike,
    ):
        for index, name in ((1, "JFACTOR"), (2, "JFACTOR_PER_SR")):
            if clumpy[index].name != name:
                raise ValueError(
                    f"Corrected CLUMPY HDU {index} is "
                    f"{clumpy[index].name!r}, expected {name!r}."
                )

            if pointlike[index].name != name:
                raise ValueError(
                    f"Pointlike HDU {index} is "
                    f"{pointlike[index].name!r}, expected {name!r}."
                )

            validate_healpix_hdu(
                clumpy[index],
                nside,
                f"CLUMPY {name}",
            )
            validate_healpix_hdu(
                pointlike[index],
                nside,
                f"Pointlike {name}",
            )

        pixels_clumpy = np.asarray(
            clumpy[1].data["PIXEL"],
            dtype=np.int32,
        )
        pixels_pointlike = np.asarray(
            pointlike[1].data["PIXEL"],
            dtype=np.int32,
        )

        if not np.array_equal(pixels_clumpy, pixels_pointlike):
            raise ValueError(
                "Corrected CLUMPY and pointlike PIXEL columns differ."
            )

        pixels_clumpy_sr = np.asarray(
            clumpy[2].data["PIXEL"],
            dtype=np.int32,
        )
        pixels_pointlike_sr = np.asarray(
            pointlike[2].data["PIXEL"],
            dtype=np.int32,
        )

        if not np.array_equal(pixels_clumpy, pixels_clumpy_sr):
            raise ValueError(
                "CLUMPY PIXEL columns differ between the two HDUs."
            )

        if not np.array_equal(pixels_clumpy, pixels_pointlike_sr):
            raise ValueError(
                "Pointlike PIXEL columns differ between the two HDUs."
            )

        clumpy_total = np.asarray(
            clumpy[1].data["Jtot"],
            dtype=np.float64,
        )
        smooth = np.asarray(
            clumpy[1].data["Jsmooth"],
            dtype=np.float64,
        )
        extended = np.asarray(
            clumpy[1].data["Jlist"],
            dtype=np.float64,
        )
        pointlike_j = np.asarray(
            pointlike[1].data["Jpointlike"],
            dtype=np.float64,
        )

        clumpy_total_sr = np.asarray(
            clumpy[2].data["Jtot_per_sr"],
            dtype=np.float64,
        )
        smooth_sr = np.asarray(
            clumpy[2].data["Jsmooth_per_sr"],
            dtype=np.float64,
        )
        extended_sr = np.asarray(
            clumpy[2].data["Jlist_per_sr"],
            dtype=np.float64,
        )
        pointlike_j_sr = np.asarray(
            pointlike[2].data["Jpointlike_per_sr"],
            dtype=np.float64,
        )

        arrays = [
            clumpy_total,
            smooth,
            extended,
            pointlike_j,
            clumpy_total_sr,
            smooth_sr,
            extended_sr,
            pointlike_j_sr,
        ]

        if any(not np.all(np.isfinite(array)) for array in arrays):
            raise ValueError("Input FITS contains non-finite J-factor values.")

        total = clumpy_total + pointlike_j
        total_sr = clumpy_total_sr + pointlike_j_sr

        pixel_area_sr = 4.0 * np.pi / pixels_clumpy.size

        if not np.allclose(
            total_sr * pixel_area_sr,
            total,
            rtol=2e-6,
            atol=0.0,
        ):
            raise RuntimeError(
                "Integrated and per-sr total maps are inconsistent."
            )

        hdu_j = make_hdu(
            name="JFACTOR",
            pixels=pixels_clumpy,
            total=total,
            smooth=smooth,
            extended=extended,
            pointlike=pointlike_j,
            nside=nside,
            per_sr=False,
        )

        hdu_sr = make_hdu(
            name="JFACTOR_PER_SR",
            pixels=pixels_clumpy,
            total=total_sr,
            smooth=smooth_sr,
            extended=extended_sr,
            pointlike=pointlike_j_sr,
            nside=nside,
            per_sr=True,
        )

        primary = fits.PrimaryHDU()
        primary.header["CONTENT"] = (
            "Auriga smooth, extended and pointlike J-factor map"
        )

        fits.HDUList(
            [
                primary,
                hdu_j,
                hdu_sr,
            ]
        ).writeto(output_fits, overwrite=True)

    print()
    print("=" * 80)
    print("Combined corrected CLUMPY and pointlike maps")
    print("=" * 80)
    print(f"Corrected CLUMPY: {corrected_fits}")
    print(f"Pointlike:        {pointlike_fits}")
    print(f"Total output:     {output_fits}")
    print(f"sum corrected J:  {clumpy_total.sum(dtype=np.float64):.16e}")
    print(f"sum pointlike J:  {pointlike_j.sum(dtype=np.float64):.16e}")
    print(f"sum total J:      {total.sum(dtype=np.float64):.16e}")
    print("=" * 80)


def main():
    args = parse_args()

    if args.repop_id < 0:
        raise ValueError("repop_id must be a non-negative integer.")

    corrected_fits, pointlike_fits, output_fits = get_paths(
        args.repop_id,
        args.scenario,
        args.nside,
    )

    combine_maps(
        corrected_fits=corrected_fits,
        pointlike_fits=pointlike_fits,
        output_fits=output_fits,
        nside=args.nside,
    )


if __name__ == "__main__":
    main()
