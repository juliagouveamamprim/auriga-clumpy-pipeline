# Validation of the extended-subhalo central-pixel proxy

## Purpose

This document records the problem identified in the catalogue-level proxy
used to estimate the central-pixel contribution of extended Auriga
subhalos, the validation tests performed, and the corrected proxy adopted
in the pipeline.

The proxy is used to define catalogue cuts before CLUMPY rendering. It
must approximate the value assigned by corrected CLUMPY to the central
HEALPix pixel of an isolated extended halo, while remaining inexpensive
enough to evaluate for catalogues containing millions of objects.

The corrected implementation uses the original `Js`, `D_Earth`, and
`r_s` columns stored in the repopulation HDF5 table. It does not perform
a numerical integral for every subhalo.

---

## Pointlike and extended classification

The pointlike/extended classification was not changed.

```math
\theta_s < \theta_{\min}
\quad\Longrightarrow\quad
\mathrm{pointlike}
```

```math
\theta_s \geq \theta_{\min}
\quad\Longrightarrow\quad
\mathrm{extended}
```

The pipeline calculates the characteristic HEALPix angular scale from
the square root of the pixel solid angle and rounds it upward to two
decimal places.

For `NSIDE=2048`:

```math
\sqrt{\Omega_{\mathrm{pix}}}
\simeq
0.0286^\circ
\quad\Longrightarrow\quad
\theta_{\min}
=
0.03^\circ
```

The CLUMPY integration aperture discussed below is used only after a halo
has already been classified as extended.

---

## Previous extended-halo proxy

The previous implementation estimated the central-pixel contribution as

```math
J_{\mathrm{old}}
=
J_s
\frac{
1-(1+x)^{-3}
}{
7/8
}
```

with

```math
x
=
\frac{
D_{\mathrm{Earth}}
\tan\theta_{\mathrm{aperture}}
}{
r_s
}
```

The HDF5 quantity `Js` is the annihilation J-factor integrated over the
halo up to `r_s`.

For an NFW profile truncated at `r_s`, the previous normalization used

```math
J_s
=
\frac{7}{8}
J_{\mathrm{general}}
```

When no explicit aperture was supplied, the pipeline used half of the
pointlike/extended angular threshold:

```math
\theta_{\mathrm{aperture}}
=
\frac{
\theta_{\min}
}{
2
}
```

For `NSIDE=2048`, this gave

```math
\theta_{\mathrm{aperture}}
=
0.015^\circ
```

Operationally, the old method read `Js` from the repopulation catalogue
and multiplied it by an analytic scaling factor. The corrected method
preserves this inexpensive catalogue-level approach, but replaces the
scaling factor.

---

## Why the previous proxy was incorrect

Three independent problems were identified.

### 1. Radial fraction instead of projected fraction

The factor

```math
1-(1+x)^{-3}
```

is the cumulative annihilation luminosity inside a three-dimensional
sphere.

A circular aperture on the sky does not select a sphere. It selects a
projected cylinder: every line-of-sight contribution whose impact
parameter lies inside the aperture contributes.

The required quantity is therefore a projected NFW-squared fraction, not
a three-dimensional radial cumulative fraction.

### 2. Incorrect CLUMPY aperture radius

Inspection of the CLUMPY source showed that, when the integration angle
is derived from `NSIDE`, CLUMPY uses

```cpp
gSIM_ALPHAINT = hp_nside2resol(gSIM_HEALPIX_NSIDE);
```

The function `hp_nside2resol` returns

```cpp
T_Healpix_Base::max_pixrad()
```

The equivalent Healpy expression is

```python
alpha_int_rad = hp.max_pixrad(nside)
```

The resulting aperture radii are:

| NSIDE | CLUMPY aperture radius |
|---:|---:|
| 1024 | 0.059800825955 deg |
| 2048 | 0.029903187205 deg |

Therefore, at `NSIDE=2048`, CLUMPY uses

```math
\alpha_{\mathrm{int}}
=
0.029903187205^\circ
```

rather than `0.015 deg`.

The value `0.029903187205 deg` is already a radius: the maximum angular
distance between a HEALPix pixel centre and one of its corners. It must
not be divided by two.

### 3. Missing aperture-to-pixel area rescaling

CLUMPY first evaluates a J-factor integrated in a circular aperture of
radius `alpha_int`.

It then rescales this value from the circular-aperture solid angle to the
HEALPix pixel solid angle:

```math
\frac{
\Omega_{\mathrm{pix}}
}{
\Omega_{\mathrm{aperture}}
}
```

where

```math
\Omega_{\mathrm{aperture}}
=
2\pi
\left(
1-\cos\alpha_{\mathrm{int}}
\right)
```

This area factor was absent from the previous proxy.

---

## Diagnostic checks

The discrepancy was investigated using ten bright, isolated extended
halos from the corrected fragile `repop_0000`, `NSIDE=2048` CLUMPY run.

The comparison used the corrected CLUMPY `Jlist` component in the central
pixel. The smooth-host contribution was not included.

The following checks were performed:

- the selected halos were isolated from every other rendered halo;
- the exact HDF5 position and the rounded CLUMPY-list position selected
  the same `NSIDE=2048` NESTED pixel;
- multiplying `Jlist_per_sr` by the pixel solid angle reproduced `Jlist`;
- the selected central pixel was the local maximum;
- the total value in `halo_rendered.log` reproduced the original HDF5
  `Js` after the rho correction.

The total normalization check gave

```math
\mathrm{median}
\left(
\frac{
J_{\mathrm{rendered}}
}{
J_s
}
\right)
=
0.999649
```

These tests excluded:

- use of the wrong FITS component;
- a unit-conversion error;
- a HEALPix pixel-index error;
- contamination by neighbouring halos;
- failure of the total rho-normalization correction.

---

## Comparison with the old proxy

For the ten selected halos, the ratio between the corrected CLUMPY
central pixel and the old proxy had median

```math
\mathrm{median}
\left(
\frac{
J_{\mathrm{CLUMPY}}
}{
J_{\mathrm{old}}
}
\right)
=
0.653053
```

with range

```math
0.527833
\leq
\frac{
J_{\mathrm{CLUMPY}}
}{
J_{\mathrm{old}}
}
\leq
0.753297
```

The old expression therefore systematically overestimated the central
pixel.

---

## Reconstructing the CLUMPY prescription

Inspection of the CLUMPY renderer showed that it does not integrate the
exact HEALPix pixel boundary for every halo.

For each pixel centre, CLUMPY evaluates an angular profile corresponding
to a circular aperture of radius `gSIM_ALPHAINT`. The result is then
rescaled by

```math
\frac{
\Omega_{\mathrm{pix}}
}{
\Omega_{\mathrm{aperture}}
}
```

A CLUMPY-like analytic calculation was constructed using:

1. an NFW-squared profile truncated at `r_s`;
2. a circular aperture of radius `alpha_int`;
3. `alpha_int = hp.max_pixrad(NSIDE)`;
4. normalization to the original HDF5 `Js`;
5. multiplication by the aperture-to-pixel solid-angle ratio.

For a proxy centred on the halo position, the comparison gave

```math
\mathrm{median}
\left(
\frac{
J_{\mathrm{CLUMPY}}
}{
J_{\mathrm{proxy}}
}
\right)
=
1.026056
```

with range

```math
0.962042
\leq
\frac{
J_{\mathrm{CLUMPY}}
}{
J_{\mathrm{proxy}}
}
\leq
1.120343
```

Nine of the ten halos agreed within approximately 6%.

The largest difference occurred for a poorly resolved halo that touched
only 17 pixels and was particularly sensitive to CLUMPY's internal
angular discretization.

Ignoring the exact subpixel offset is therefore adequate for the
catalogue-cut proxy. The proxy can depend only on `Js`, `D_Earth`, `r_s`,
and `NSIDE`.

---

## Closed-form projected NFW fraction

Define the dimensionless projected aperture

```math
y
=
\frac{
D_{\mathrm{Earth}}
\sin\alpha_{\mathrm{int}}
}{
r_s
}
```

For an NFW profile truncated at `r_s`, the fraction of `Js` contained
inside a centred projected circular aperture is

```math
F_{\mathrm{proj}}(y)
=
1-\frac{24}{7}G(y),
\qquad
0<y<1
```

where

```math
G(y)
=
-y\arccos y
+
\frac{
7
-36y^2
+45y^4
-16y^6
-12y^2
\left(
2y^4-5y^2+4
\right)
\ln y
}{
24
\left(
1-y^2
\right)^{5/2}
}
```

The boundary values are

```math
F_{\mathrm{proj}}(0)
=
0
```

and

```math
F_{\mathrm{proj}}(y)
=
1,
\qquad
y\geq1
```

Series expansions are used close to `y=0` and `y=1` to avoid numerical
cancellation.

---

## Validation of the closed form

The closed expression was compared with direct numerical integration of
the projected NFW-squared profile for the ten validation halos.

The ratio had median

```math
\mathrm{median}
\left(
\frac{
J_{\mathrm{closed}}
}{
J_{\mathrm{numerical}}
}
\right)
=
1.00001861
```

with range

```math
1.00001629
\leq
\frac{
J_{\mathrm{closed}}
}{
J_{\mathrm{numerical}}
}
\leq
1.00003093
```

The closed form therefore reproduces the numerical projected integral
at substantially better precision than required for the catalogue cuts.

Merely replacing `0.015 deg` by `0.0299 deg` in the old radial expression
and adding the area factor was not sufficient.

That simple approximation gave

```math
\mathrm{median}
\left(
\frac{
J_{\mathrm{CLUMPY}}
}{
J_{\mathrm{simple}}
}
\right)
=
1.260249
```

The projected fraction itself is therefore essential.

---

## Adopted proxy

For extended halos, the catalogue-level central-pixel proxy is now

```math
J_{\mathrm{pixel,proxy}}^{\mathrm{ext}}
=
J_s
F_{\mathrm{proj}}(y)
\frac{
\Omega_{\mathrm{pix}}
}{
\Omega_{\mathrm{aperture}}
}
```

with

```math
y
=
\frac{
D_{\mathrm{Earth}}
\sin\alpha_{\mathrm{int}}
}{
r_s
}
```

The aperture is obtained in the implementation with

```python
alpha_int_rad = hp.max_pixrad(nside)
```

For pointlike halos, the proxy remains

```math
J_{\mathrm{pixel,proxy}}^{\mathrm{point}}
=
J_s
```

The common reference is

```math
J_{\mathrm{pixel,ref}}
=
\max
\left[
\max
\left(
J_s^{\mathrm{point}}
\right),
\max
\left(
J_{\mathrm{pixel,proxy}}^{\mathrm{ext}}
\right)
\right]
```

The cut thresholds remain

```math
J_{\mathrm{cut}}^{\mathrm{point}}
=
f_{\mathrm{point}}
J_{\mathrm{pixel,ref}}
```

and

```math
J_{\mathrm{cut}}^{\mathrm{ext}}
=
f_{\mathrm{ext}}
J_{\mathrm{pixel,ref}}
```

Only the definition of the extended central-pixel proxy has changed.

---

## Computational cost

No numerical integral is performed for each subhalo.

The calculation is fully vectorized:

```python
alpha_int_rad = hp.max_pixrad(nside)

omega_pixel = hp.nside2pixarea(nside)

omega_aperture = (
    2.0
    * np.pi
    * (1.0 - np.cos(alpha_int_rad))
)

y = (
    d_earth_kpc
    * np.sin(alpha_int_rad)
    / rs_kpc
)

fraction = projected_nfw_fraction(y)

j_pixel_proxy = (
    js
    * fraction
    * omega_pixel
    / omega_aperture
)
```

Operationally, the pipeline still:

1. reads `Js` from the repopulation table;
2. calculates an analytic dimensionless fraction;
3. multiplies `Js` by that fraction and by the aperture-to-pixel area
   factor.

No lookup table and no per-halo numerical quadrature are required.

---

## Consequences for the cut scan

All previous cut scans used the legacy radial proxy with

```math
\theta_{\mathrm{aperture}}
=
0.015^\circ
```

Those scans must be rerun with the projected CLUMPY-like proxy before the
final values of `pointlike_f` and `extended_f` are frozen.

The previously quoted result

```text
pointlike_f = 1e-3
extended_f  = 1e-3
```

must be treated as a result of the legacy diagnostic until the corrected
ten-repopulation scan is completed.

Even if the preferred numerical value of `f` remains unchanged, the
selected catalogues must be regenerated because both the extended proxy
and the common reference `J_pixel_ref` may have changed.
