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


def test_prepare_subhalo_components_with_cuts():
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
            # kept extended: high Jtheta proxy
            [1e19, 10.0, 20.0, 8.0, 1.0, 1.0, 1e7, 10.0, 0.0, 0.0],
            # discarded extended: low Jtheta proxy
            [1e16, 10.0, 20.0, 8.0, 1.0, 1.0, 1e7, -10.0, 0.0, 0.0],
            # kept pointlike: Js >= 1e-4 * Jref
            [1e20, 10.0, 20.0, 1.0, 1.0, 1.0, 1e7, 0.0, 10.0, 0.0],
            # discarded pointlike: Js < 1e-4 * Jref
            [1e14, 10.0, 20.0, 1.0, 1.0, 1.0, 1e7, 0.0, -10.0, 0.0],
        ],
        dtype=np.float64,
    )

    names = np.array(
        [
            b"extended_keep",
            b"extended_drop",
            b"pointlike_keep",
            b"pointlike_drop",
        ]
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
            top_n=None,
            halo_type="DSPH",
            nside=8,
            round_up_decimals=2,
            chunk_size=2,
            extended_cut_f=1e-3,
            pointlike_cut_f=1e-4,
            theta_aperture_deg=1.0,
        )

        rows = [
            line.split()[0]
            for line in output_list.read_text().splitlines()
            if line.strip() and not line.startswith("#")
        ]

        assert rows == ["extended_keep"]

        with fits.open(output_fits) as hdul:
            assert hdul[1].header["CUTS"] is True
            assert hdul[1].header["NPOINT"] == 1
            assert hdul[1].header["NPTOTAL"] == 2
            assert hdul[1].header["NSIDE"] == 8

            jmap = np.asarray(hdul[1].data["Jpointlike"], dtype=np.float64)

            pixel = hp.ang2pix(
                8,
                90.0,
                0.0,
                lonlat=True,
                nest=True,
            )

            assert np.isclose(jmap.sum(), 1e20, rtol=1e-12)
            assert np.isclose(jmap[pixel], 1e20, rtol=1e-12)
