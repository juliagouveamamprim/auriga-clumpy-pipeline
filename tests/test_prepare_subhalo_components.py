#!/usr/bin/env python3

import sys
import tempfile
from pathlib import Path

import h5py
import healpy as hp
import numpy as np
from astropy.io import fits

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "scripts"))

from prepare_subhalo_components import prepare_subhalo_components


def main():
    column_names = [
        "Js",
        "D_Earth",
        "Vmax",
        "theta_s",
        "Cv",
        "r_s",
        "rho_s",
        "Xearth",
        "Yearth",
        "Zearth",
    ]

    data = np.array(
        [
            [1e18, 10.0, 20.0, 8.0, 1.0, 1.0, 1e7, 10.0, 0.0, 0.0],
            [2e18, 10.0, 20.0, 1.0, 1.0, 1.0, 1e7, 0.0, 10.0, 0.0],
            [3e18, 10.0, 20.0, 2.0, 1.0, 1.0, 1e7, 0.0, 10.0, 0.0],
            [4e18, 10.0, 20.0, 9.0, 1.0, 1.0, 1e7, -10.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )

    names = np.array(
        [b"extended_a", b"pointlike_a", b"pointlike_b", b"extended_b"]
    )

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        input_h5 = tmp / "input.h5"
        output_list = tmp / "extended.txt"
        output_fits = tmp / "pointlike.fits"

        with h5py.File(input_h5, "w") as h5:
            group = h5.create_group("iteration_0")
            dataset = group.create_dataset("data", data=data)
            dataset.attrs["column_names"] = column_names
            group.create_dataset("halo_name", data=names)

        prepare_subhalo_components(
            input_h5=input_h5,
            output_list=output_list,
            output_pointlike_fits=output_fits,
            scenario="resilient",
            repop_id=7,
            iteration=0,
            top_n=1,
            halo_type="DSPH",
            nside=8,
            round_up_decimals=2,
            chunk_size=2,
        )

        rows = [
            line.split()[0]
            for line in output_list.read_text().splitlines()
            if line.strip() and not line.startswith("#")
        ]

        assert rows == ["extended_a"]

        with fits.open(output_fits) as hdul:
            assert hdul[1].name == "JFACTOR"
            assert hdul[2].name == "JFACTOR_PER_SR"
            assert hdul[1].header["ORDERING"] == "NESTED"
            assert hdul[1].header["NSIDE"] == 8
            assert hdul[1].header["NPOINT"] == 2

            jmap = np.asarray(hdul[1].data["Jpointlike"], dtype=np.float64)
            jsr = np.asarray(
                hdul[2].data["Jpointlike_per_sr"],
                dtype=np.float64,
            )

            pixel = hp.ang2pix(
                8,
                90.0,
                0.0,
                lonlat=True,
                nest=True,
            )
            pixel_area = hp.nside2pixarea(8)

            assert np.isclose(jmap.sum(), 5e18, rtol=1e-12)
            assert np.isclose(jmap[pixel], 5e18, rtol=1e-12)
            assert np.allclose(jsr * pixel_area, jmap, rtol=1e-12)

    print(
        "PASS: extended TOP_N does not truncate the pointlike map; "
        "NESTED pixel placement and J conservation are correct."
    )


if __name__ == "__main__":
    main()
