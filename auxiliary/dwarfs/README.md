# Milky Way dwarf-spheroidal J-factor templates

This auxiliary component generates all-sky annihilation J-factor maps for the eight dwarf spheroidal galaxies with density-profile parameters published in Tables 2, A2, and A3 of *Prospects for dark matter observations in dwarf spheroidal galaxies with the Cherenkov Telescope Array Observatory* (arXiv:2508.19120v3). It is independent of the Auriga repopulation pipeline.

The catalogues use the central values from the article. Table 2 J2000 coordinates were converted to Galactic longitude and latitude with Astropy. Table A2/A3 density values were multiplied by `1e7` to obtain `Msun/kpc^3`. The profile-specific published tidal radius is supplied to CLUMPY as `Rdelta`, and internal dwarf substructure is disabled.

Both configurations retain the fragile-template smooth Milky Way model, including its already-renormalized `gMW_RHOSOL = 0.3949787768 GeV/cm^3`. This is an operational configuration choice and does not associate the observed dwarfs with the fragile disruption scenario.

Run from the repository root on Lattes with the physical-boundary-patched executable:

```bash
mkdir -p outputs/dwarfs/einasto outputs/dwarfs/burkert

/home/julia/src/CLUMPY_external_nobkg/build/cli/clumpy -g6 \
  -i auxiliary/dwarfs/configs/clumpy_params_g6_dwarfs_einasto.txt

/home/julia/src/CLUMPY_external_nobkg/build/cli/clumpy -g6 \
  -i auxiliary/dwarfs/configs/clumpy_params_g6_dwarfs_burkert.txt

mv outputs/dwarfs/einasto/annihil_gal2D_LOS0_0_FOV360x180_nside2048.fits \
  outputs/dwarfs/dwarfs_einasto_nside2048.fits
mv outputs/dwarfs/burkert/annihil_gal2D_LOS0_0_FOV360x180_nside2048.fits \
  outputs/dwarfs/dwarfs_burkert_nside2048.fits
```

The native `JFACTOR` extension is retained in full. Its `Jlist` column is the pixel-integrated dwarf-only contribution in `GeV^2 cm^-5`; `Jsmooth` is the smooth Milky Way contribution and `Jtot` is their sum.
