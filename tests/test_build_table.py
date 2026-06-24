from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import h5py
import numpy as np
from astropy import constants as c
from astropy import units as u


module_path = Path("src/repop_algorithm.py")
spec = spec_from_file_location("repop_algorithm", module_path)
module = module_from_spec(spec)
spec.loader.exec_module(module)

algorithm = module.RepopAlgorithm.__new__(module.RepopAlgorithm)
algorithm.input_dict = {
    "cosmo_constants": {
        "G": c.G,
    },
    "host": {
        "position_Earth": np.array([8.0, 0.0, 0.0]) * u.kpc,
    },
}

catalog = {
    "Js": np.array([1.0, 4.0, 3.0, 2.0]) * 1e18 * u.GeV**2 / u.cm**5,
    "D_GC": np.array([5.0, 6.0, 7.0, 8.0]) * u.kpc,
    "D_Earth": np.array([10.0, 20.0, 30.0, 40.0]) * u.kpc,
    "Vmax": np.array([10.0, 20.0, 30.0, 40.0]) * u.km / u.s,
    "theta_s": np.array([0.10, 0.20, 0.30, 0.40]) * u.deg,
    "Cv": np.array([1e4, 2e4, 3e4, 4e4]),
    "R_s": np.array([0.10, 0.20, 0.30, 0.40]) * u.kpc,
    "galactocentric_X": np.array([1.0, 2.0, 3.0, 4.0]) * u.kpc,
    "galactocentric_Y": np.array([0.0, 1.0, 2.0, 3.0]) * u.kpc,
    "galactocentric_Z": np.array([0.0, 0.0, 1.0, 2.0]) * u.kpc,
    "engulfs_Earth": np.array([False, False, True, False]),
    "survives_Roche": np.array([True, False, True, True]),
}

output = Path("/tmp/auriga_hdf5_test_table.h5")

with h5py.File(output, "w") as h5:
    algorithm._write_table_iteration(h5, 0, catalog)

with h5py.File(output, "r") as h5:
    group = h5["iteration_0"]
    data = group["data"][:]
    names = [
        value.decode("utf-8") if isinstance(value, bytes) else str(value)
        for value in group["halo_name"][:]
    ]
    columns = [
        value.decode("utf-8") if isinstance(value, bytes) else str(value)
        for value in group["data"].attrs["column_names"]
    ]
    units = [
        value.decode("utf-8") if isinstance(value, bytes) else str(value)
        for value in group["data"].attrs["units"]
    ]
    stats = {
        "n_generated": int(group.attrs["n_generated"]),
        "n_removed_engulfing": int(group.attrs["n_removed_engulfing"]),
        "n_removed_roche": int(group.attrs["n_removed_roche"]),
        "n_saved": int(group.attrs["n_saved"]),
    }

expected_columns = [
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

assert data.shape == (2, 10), data.shape
assert columns == expected_columns, columns
assert len(units) == 10, units

# Halos 1 and 2 are removed; the two survivors are sorted by decreasing Js.
np.testing.assert_allclose(data[:, 0], [2e18, 1e18])
np.testing.assert_allclose(data[:, 1], [40.0, 10.0])

# Xearth = -(Xgc - Xearth_position).
np.testing.assert_allclose(data[:, 7], [4.0, 7.0])
np.testing.assert_allclose(data[:, 8], [3.0, 0.0])
np.testing.assert_allclose(data[:, 9], [2.0, 0.0])

assert names == ["hydro_res_00001", "hydro_res_00002"], names
assert stats["n_generated"] == 4
assert stats["n_removed_engulfing"] == 1
assert stats["n_removed_roche"] == 1
assert stats["n_saved"] == 2

print("PASS: v2 table has 10 columns and preserves filtering, sorting, coordinates, and metadata.")
print("Columns:", columns)
print("Shape:", data.shape)
print("Saved Js:", data[:, 0])
