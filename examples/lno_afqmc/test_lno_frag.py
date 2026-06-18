"""Sanity check: fragment 2-body energy over ALL occupied orbitals == standard total.

If pinned == full occupied space, sum_I (ecoul_lno - exx_lno) must equal the standard
UHF 2-body energy (ecoul - exx). Anchors both the LNO kernels and the slicing in
run_lno_afqmc._frag_2body.
"""
import numpy as np

from ipie.estimators.local_energy_sd import (
    ecoul_kernel_batch_real_rchol_uhf,
    exx_kernel_batch_real_rchol,
    ecorrcoul_lno_real_rchol_uhf,
    ecorrxx_lno_real_rchol,
)

rng = np.random.default_rng(0)
naux, nocc, nbasis, nw = 5, 3, 4, 2
# Closed-shell LNO reference: rchola == rcholb (the coul kernel assumes this).
rchola = rng.standard_normal((naux, nocc, nbasis))
rcholb = rchola
Ga = (rng.standard_normal((nw, nocc, nbasis)) + 1j * rng.standard_normal((nw, nocc, nbasis)))
Gb = (rng.standard_normal((nw, nocc, nbasis)) + 1j * rng.standard_normal((nw, nocc, nbasis)))

# Standard total 2-body (e2 = ecoul - exx).
e_std = ecoul_kernel_batch_real_rchol_uhf(
    rchola.reshape(naux, -1), rcholb.reshape(naux, -1),
    Ga.reshape(nw, -1), Gb.reshape(nw, -1),
)
e_std -= exx_kernel_batch_real_rchol(rchola.reshape(naux, -1), Ga)
e_std -= exx_kernel_batch_real_rchol(rcholb.reshape(naux, -1), Gb)

# Fragment = ALL occupied -> must reproduce the total.
ecoul = ecorrcoul_lno_real_rchol_uhf(rchola, rchola, Ga, Ga, Gb, Gb)
exx = ecorrxx_lno_real_rchol(rchola, rchola, Ga, Ga)
exx += ecorrxx_lno_real_rchol(rcholb, rcholb, Gb, Gb)
e_frag = (ecoul - exx).sum(axis=1)

assert np.allclose(e_frag, e_std, atol=1e-10), np.abs(e_frag - e_std).max()
print("PASS: fragment(all-occ) == standard total 2-body, max|diff| =",
      np.abs(e_frag - e_std).max())
