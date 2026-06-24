#!/usr/bin/env python3

import sys
import tempfile
from pathlib import Path

import healpy as hp
import numpy as np
from astropy.io import fits

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "scripts"))

from combine_clumpy_pointlike import combine_maps
from prepare_subhalo_components import write_pointlike_fits


def make_clumpy_hdu(name, pixels, total, smooth, extended, nside, per_sr):
    suffix = "_per_sr" if per_sr else ""
    unit = "GeV^2 cm^-5 sr^-1" if per_sr else "GeV^2 cm^-5"

    hdu = fits.BinTableHDU.from_columns(
        [
            fits.Column(name="PIXEL", format="1J", array=pixels),
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
                name=f"Jlist{suffix}",
                format="1E",
                unit=unit,
                array=extended.astype(np.float32),
            ),
        ],
        name=name,
    )

    hdu.header["PIXTYPE"] = "HEALPIX"
    hdu.header["ORDERING"] = "NESTED"
    hdu.header["NSIDE"] = nside
    hdu.header["FIRSTPIX"] = 0
    hdu.header["LASTPIX"] = pixels.size - 1
    hdu.header["INDXSCHM"] = "EXPLICIT"
    hdu.header["COORDSYS"] = "G"

    return hdu


def main():
    nside = 2
    npix = hp.nside2npix(nside)
    area = hp.nside2pixarea(nside)
    pixels = np.arange(npix, dtype=np.int32)

    smooth = np.full(npix, 2.0e16, dtype=np.float64)
    extended = np.zeros(npix, dtype=np.float64)
    extended[3] = 4.0e17
    extended[10] = 6.0e17

    clumpy_total = smooth + extended

    pointlike = np.zeros(npix, dtype=np.float64)
    pointlike[3] = 1.0e18
    pointlike[20] = 2.0e18

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)

        clumpy_path = tmp / "corrected_clumpy.fits"
        pointlike_path = tmp / "pointlike.fits"
        total_path = tmp / "total.fits"

        fits.HDUList(
            [
                fits.PrimaryHDU(),
                make_clumpy_hdu(
                    "JFACTOR",
                    pixels,
                    clumpy_total,
                    smooth,
                    extended,
                    nside,
                    per_sr=False,
                ),
                make_clumpy_hdu(
                    "JFACTOR_PER_SR",
                    pixels,
                    clumpy_total / area,
                    smooth / area,
                    extended / area,
                    nside,
                    per_sr=True,
                ),
            ]
        ).writeto(clumpy_path)

        write_pointlike_fits(
            output_path=pointlike_path,
            pointlike_map=pointlike,
            nside=nside,
            scenario="resilient",
            repop_id=7,
            theta_cut_deg=7.33,
            n_pointlike=2,
        )

        combine_maps(
            corrected_fits=clumpy_path,
            pointlike_fits=pointlike_path,
            output_fits=total_path,
            nside=nside,
        )

        with fits.open(total_path) as hdul:
            assert hdul[1].name == "JFACTOR"
            assert hdul[2].name == "JFACTOR_PER_SR"
            assert hdul[1].header["ORDERING"] == "NESTED"
            assert hdul[1].header["NSIDE"] == nside

            total = np.asarray(hdul[1].data["Jtot"], dtype=np.float64)
            out_smooth = np.asarray(
                hdul[1].data["Jsmooth"],
                dtype=np.float64,
            )
            out_extended = np.asarray(
                hdul[1].data["Jextended"],
                dtype=np.float64,
            )
            out_pointlike = np.asarray(
                hdul[1].data["Jpointlike"],
                dtype=np.float64,
            )
            total_per_sr = np.asarray(
                hdul[2].data["Jtot_per_sr"],
                dtype=np.float64,
            )

            expected_total = clumpy_total + pointlike

            assert np.allclose(total, expected_total, rtol=2e-6)
            assert np.allclose(out_smooth, smooth, rtol=2e-6)
            assert np.allclose(out_extended, extended, rtol=2e-6)
            assert np.allclose(out_pointlike, pointlike, rtol=2e-6)
            assert np.allclose(
                total_per_sr * area,
                total,
                rtol=2e-6,
            )

    print(
        "PASS: corrected CLUMPY and pointlike maps are combined "
        "without double counting, with all components preserved."
    )


if __name__ == "__main__":
    main()
