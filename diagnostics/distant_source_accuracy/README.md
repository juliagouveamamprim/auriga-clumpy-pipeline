# Distant-source accuracy diagnostic

This diagnostic checks the distant-source approximation used for catalogue
subhalo J-factors.

For each input catalogue, it computes eta = r_s / D_Earth for every subhalo.
The script writes a catalogue-level summary and stores individual rows only for
halos above the 1% correction threshold.

Generated CSV outputs and run logs should be written under outputs/ and are not
tracked by git.
