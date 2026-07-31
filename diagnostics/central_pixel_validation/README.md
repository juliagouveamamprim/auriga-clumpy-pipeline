# Validation of the extended-subhalo central-pixel proxy

## Purpose

This document records the problem found in the catalogue-level proxy used
to estimate the central-pixel contribution of extended Auriga subhalos,
the tests performed to isolate the origin of the discrepancy, and the
replacement adopted in the pipeline.

The proxy is used before CLUMPY rendering to define catalogue cuts. It
must therefore approximate the quantity assigned by corrected CLUMPY to
the central HEALPix pixel of an isolated extended halo, while remaining
cheap enough to evaluate for catalogues containing millions of objects.

The final implementation uses the original `Js`, `D_Earth`, and `r_s`
columns stored in the repopulation HDF5 table. It does not perform a
numerical integral for every subhalo.

## Classification of pointlike and extended halos

The pointlike/extended classification was not changed by this work.

The masks remain

\[
\theta_s < \theta_{\min}
\quad\Longrightarrow\quad
\text{pointlike},
\]

and

\[
\theta_s \geq \theta_{\min}
\quad\Longrightarrow\quad
\text{extended},
\]

where

\[
\theta_{\min}
=
\operatorname{round\_up}
\left(
\sqrt{\Omega_{\rm pix}}
\right).
\]

For `NSIDE=2048`,

\[
\sqrt{\Omega_{\rm pix}}\simeq 0.0286^\circ,
\qquad
\theta_{\min}=0.03^\circ.
\]

The new aperture definition described below is used only after a halo
has already been classified as extended.

## Previous extended-halo proxy

The previous implementation estimated the central-pixel contribution as

\[
J_{\rm old}
=
J_s
\frac{1-(1+x)^{-3}}{7/8},
\]

with

\[
x=
\frac{D_{\rm Earth}\tan\theta_{\rm aperture}}{r_s}.
\]

The HDF5 quantity `Js` is the annihilation J-factor integrated up to
\(r_s\). For an NFW profile truncated at \(r_s\), the old expression
used

\[
J_s = \frac{7}{8}J_{\rm general}.
\]

When no aperture was supplied, the pipeline set

\[
\theta_{\rm aperture}
=
\frac{\theta_{\min}}{2}.
\]

At `NSIDE=2048`, this gave

\[
\theta_{\rm aperture}=0.015^\circ.
\]

## Why the previous expression was incorrect

Three independent issues were identified.

### 1. It used a three-dimensional radial fraction

The factor

\[
1-(1+x)^{-3}
\]

is the cumulative annihilation luminosity inside a three-dimensional
sphere.

A sky aperture does not select a sphere. It selects a projected
cylinder: every line-of-sight contribution with impact parameter inside
the angular aperture contributes.

The required quantity is therefore a projected NFW-squared fraction,
not a three-dimensional radial fraction.

### 2. The CLUMPY aperture radius was misidentified

Inspection of the CLUMPY source showed that, when the integration angle
is obtained from `NSIDE`, CLUMPY uses

```cpp
gSIM_ALPHAINT = hp_nside2resol(gSIM_HEALPIX_NSIDE);
```

and `hp_nside2resol` returns

```cpp
T_Healpix_Base::max_pixrad()
```

The Python equivalent is

```python
alpha_int_rad = hp.max_pixrad(nside)
```

This gives

| NSIDE | `hp.max_pixrad` |
|---:|---:|
| 1024 | \(0.059800825955^\circ\) |
| 2048 | \(0.029903187205^\circ\) |

Thus, for `NSIDE=2048`, the integration aperture radius used by CLUMPY
is approximately \(0.0299^\circ\), not \(0.015^\circ\).

The value \(0.0299^\circ\) is already a radius: the maximum angular
distance between a HEALPix pixel centre and its corners. It must not be
divided by two.

### 3. CLUMPY rescales aperture area to pixel area

CLUMPY evaluates a J-factor integrated in a circular aperture of radius
\(\alpha_{\rm int}\), then rescales it from the circular-aperture solid
angle to the HEALPix pixel solid angle:

\[
\frac{\Omega_{\rm pix}}{\Omega_{\rm aperture}},
\]

where

\[
\Omega_{\rm aperture}
=
2\pi\left(1-\cos\alpha_{\rm int}\right).
\]

This area factor was absent from the previous proxy.

## Diagnostic sequence

The discrepancy was investigated using isolated extended halos from the
corrected fragile `repop_0000`, `NSIDE=2048` CLUMPY run.

The initial comparison used the corrected CLUMPY `Jlist` value in the
halo central pixel. The smooth-host contribution was not included.

Checks confirmed that:

- the selected halos were isolated from every other rendered halo;
- the exact HDF5 position and the rounded CLUMPY-list position selected
  the same `NSIDE=2048` NESTED pixel;
- `Jlist_per_sr * Omega_pix` reproduced `Jlist`;
- the selected central pixel was the local maximum;
- the corrected total rendered value from `halo_rendered.log` reproduced
  the original HDF5 `Js`, with median
  \(J_{\rm rendered}/J_s \simeq 0.999649\).

These checks excluded a wrong HDU, a unit conversion, a pixel-index
error, contamination by nearby halos, and failure of the total
rho-normalization correction.

For the ten selected halos, the corrected CLUMPY central pixel divided
by the old proxy had

\[
\mathrm{median}=0.653053,
\]

with range

\[
0.527833 \leq
\frac{J_{\rm CLUMPY}}{J_{\rm old}}
\leq 0.753297.
\]

The old expression therefore systematically overestimated the central
pixel.

## Reconstructing the CLUMPY prescription

Inspection of the CLUMPY renderer showed that it does not numerically
integrate the exact HEALPix pixel boundary for every halo.

Instead, for each pixel centre, it evaluates the halo radial angular
profile, where each tabulated value corresponds to a circular aperture
of radius `gSIM_ALPHAINT`. The map is subsequently rescaled by

\[
\Omega_{\rm pix}/\Omega_{\rm aperture}.
\]

A CLUMPY-like analytic calculation was therefore constructed with:

1. an NFW-squared profile truncated at \(r_s\);
2. a circular aperture of radius
   \(\alpha_{\rm int}=\texttt{hp.max\_pixrad(NSIDE)}\);
3. the projected aperture centred on the halo;
4. normalization to the original HDF5 `Js`;
5. multiplication by
   \(\Omega_{\rm pix}/\Omega_{\rm aperture}\).

For the centred proxy, the corrected CLUMPY central pixel divided by the
analytic CLUMPY-like value had

\[
\mathrm{median}=1.026056,
\]

with range

\[
0.962042 \leq
\frac{J_{\rm CLUMPY}}{J_{\rm proxy}}
\leq 1.120343.
\]

Nine of the ten test halos agreed within approximately 6%. The largest
difference occurred for a poorly resolved halo that touched only 17
pixels and was therefore particularly sensitive to CLUMPY's internal
angular discretization.

Ignoring the exact subpixel offset is consequently adequate for the
catalogue-cut proxy. The proxy can depend only on `Js`, `D_Earth`,
`r_s`, and `NSIDE`.

## Closed-form projected NFW fraction

Define

\[
y=
\frac{D_{\rm Earth}\sin\alpha_{\rm int}}{r_s}.
\]

For an NFW profile truncated at \(r_s\), the fraction of `Js` contained
inside a centred projected circular aperture is

\[
F_{\rm proj}(y)
=
1-\frac{24}{7}G(y),
\qquad 0<y<1,
\]

where

\[
G(y)
=
-y\arccos y
+
\frac{
7-36y^2+45y^4-16y^6
-12y^2(2y^4-5y^2+4)\ln y
}{
24(1-y^2)^{5/2}
}.
\]

The boundary values are

\[
F_{\rm proj}(0)=0,
\qquad
F_{\rm proj}(y\geq1)=1.
\]

Series expansions are used close to \(y=0\) and \(y=1\) to prevent
catastrophic cancellation.

The closed form was checked against the direct numerical projected NFW
integration for the ten validation halos. The ratio was

\[
\mathrm{median}
\left(
\frac{J_{\rm closed}}{J_{\rm numerical}}
\right)
=1.00001861,
\]

with range

\[
1.00001629
\leq
\frac{J_{\rm closed}}{J_{\rm numerical}}
\leq
1.00003093.
\]

This confirms that the closed form reproduces the numerical projected
integral at substantially better precision than required for the
catalogue cuts.

Merely replacing \(0.015^\circ\) by \(0.0299^\circ\) in the old radial
formula and adding the area rescaling was not sufficient. That simple
variant underestimated CLUMPY, with median

\[
\frac{J_{\rm CLUMPY}}{J_{\rm simple}}\simeq1.260.
\]

The projected fraction itself is therefore essential.

## Adopted proxy

For extended halos, the catalogue-level central-pixel proxy is now

\[
J_{\rm pixel,proxy}^{\rm ext}
=
J_s
F_{\rm proj}(y)
\frac{\Omega_{\rm pix}}{\Omega_{\rm aperture}},
\]

with

\[
y=
\frac{
D_{\rm Earth}\sin[\texttt{hp.max\_pixrad(NSIDE)}]
}{
r_s
}.
\]

For pointlike halos, the proxy remains

\[
J_{\rm pixel,proxy}^{\rm point}=J_s.
\]

The common reference is

\[
J_{\rm pixel,ref}
=
\max\left[
\max(J_s^{\rm point}),
\max(J_{\rm pixel,proxy}^{\rm ext})
\right].
\]

The cut thresholds remain

\[
J_{\rm cut}^{\rm point}
=
f_{\rm point}J_{\rm pixel,ref},
\]

and

\[
J_{\rm cut}^{\rm ext}
=
f_{\rm ext}J_{\rm pixel,ref}.
\]

## Computational cost

No numerical integral is performed for each subhalo.

The calculation is fully vectorized:

```python
alpha_int_rad = hp.max_pixrad(nside)

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

Operationally, the pipeline still reads `Js` from the repopulation table
and multiplies it by an analytic scaling factor.

## Consequences for the cut scan

All previous cut scans used the legacy radial proxy with
\(\theta_{\rm aperture}=0.015^\circ\).

Those scans must be rerun with the projected CLUMPY-like proxy before
the final values of `pointlike_f` and `extended_f` are frozen.

The previously quoted `f=1e-3` result should be treated as a result of
the legacy diagnostic, not as the final result of the corrected proxy.
