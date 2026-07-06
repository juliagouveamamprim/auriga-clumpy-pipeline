import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import combine_clumpy_pointlike as comb
import correct_rhos_from_clumpy_raw as corr
import generate_clumpy_params as gen


def test_default_nside_keeps_historical_names():
    paths = gen.get_paths(230, "resilient", "raw", 1024)

    assert paths["input_list"].name == "repop_0230_raw_nopointlike.txt"
    assert paths["output_dir"].name == "repop_0230"
    assert paths["output_param"].name == "repop_0230_raw_params.txt"


def test_nondefault_nside_uses_suffixed_names():
    raw = gen.get_paths(230, "resilient", "raw", 2048)
    corrected = gen.get_paths(230, "resilient", "corrected", 2048)

    assert raw["input_list"].name == "repop_0230_raw_nopointlike_nside2048.txt"
    assert raw["output_dir"].name == "repop_0230_nside2048"
    assert raw["output_param"].name == "repop_0230_raw_params_nside2048.txt"

    assert corrected["input_list"].name == "repop_0230_rhocorr_nside2048.txt"
    assert corrected["output_dir"].name == "repop_0230_nside2048"
    assert corrected["output_param"].name == "repop_0230_corrected_params_nside2048.txt"


def test_correction_and_combine_paths_use_nside():
    paths = corr.get_paths(230, "resilient", 2048)

    assert paths["input_list"].name == "repop_0230_raw_nopointlike_nside2048.txt"
    assert paths["raw_clumpy_output_dir"].name == "repop_0230_nside2048"
    assert (
        paths["rendered_log"].name
        == "annihil_gal2D_LOS0_0_FOV360x180_nside2048.halo_rendered.log"
    )
    assert paths["output_list"].name == "repop_0230_rhocorr_nside2048.txt"

    corrected_fits, pointlike_fits, total_fits = comb.get_paths(
        230,
        "resilient",
        2048,
    )

    assert corrected_fits.parent.name == "repop_0230_nside2048"
    assert corrected_fits.name == "annihil_gal2D_LOS0_0_FOV360x180_nside2048.fits"
    assert pointlike_fits.name == "repop_0230_pointlike_nside2048.fits"
    assert total_fits.name == "auriga_total_nside2048.fits"
