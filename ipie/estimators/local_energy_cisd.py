# Copyright 2022 The ipie Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either eprXess or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Authors: Jinghong Zhang <jinghongzhang@fas.harvard.edu>
#
from ipie.utils.backend import arraylib as xp
from ipie.hamiltonians.generic import GenericRealChol
from ipie.walkers.uhf_walkers import UHFWalkers
from ipie.trial_wavefunction.cisd import CISD


def local_energy_rcisd_batch(hamiltonian: GenericRealChol, walkers: UHFWalkers, trial: CISD):
    """Calculate local energy for RCISD trial wavefunction.

    Parameters
    ----------
    hamiltonian : object
        Hamiltonian object.
    walkers : object
        WalkerBatch object (this stores some intermediates for the particular trial wfn).
    trial : object
        Trial wavefunction object.

    Returns
    -------
    eloc : float / complex
        Local energy.
    """
    gova = walkers.ghalfa[:, :, trial.nalpha:]
    nvira = hamiltonian.nbasis - trial.nalpha
    eye_a = -xp.broadcast_to(xp.eye(nvira)[None, :, :], (gova.shape[0], nvira, nvira))
    calg_a = xp.concatenate([gova, eye_a], axis=1)

    h1g = xp.einsum("kq,wkq->w", trial.rh1, walkers.ghalfa, optimize=True)
    c1gov = xp.einsum("ia,wia->w", trial.c1a, gova, optimize=True)
    c2g_dir = xp.einsum("iajb,wia->wjb", trial.c2aa, gova, optimize=True)
    c2g_cross = xp.einsum("iajb,wib->wja", trial.c2aa, gova, optimize=True)
    c2g_dir_G = xp.einsum("wia,wiq->wqa", c2g_dir, walkers.ghalfa, optimize=True)
    calg_c2g_dir_G = xp.einsum("wpa,wqa->wpq", calg_a, c2g_dir_G, optimize=True)
    c2g_cross_G = xp.einsum("wja,wjq->wqa", c2g_cross, walkers.ghalfa, optimize=True)
    calg_c2g_cross_G = xp.einsum("wpa,wqa->wpq", calg_a, c2g_cross_G, optimize=True)
    chol = hamiltonian.chol.reshape((hamiltonian.nbasis, hamiltonian.nbasis, hamiltonian.nchol))
    chol_g = xp.einsum("kqX,wkq->wX", trial.rchol, walkers.ghalfa, optimize=True)
    cholg_mat = xp.einsum("krX,wks->wXsr", trial.rchol, walkers.ghalfa, optimize=True)
    c1calg = xp.einsum("ia,wpa->wpi", trial.c1a, calg_a, optimize=True)
    c1calgG = xp.einsum("wpi,wiq->wpq", c1calg, walkers.ghalfa, optimize=True)
    lc1calgG = xp.einsum("prX,wpr->wX", chol, c1calgG, optimize=True)
    c1g_mat = xp.einsum("ia,wir->wra", trial.c1a, walkers.ghalfa, optimize=True)
    c1g_matcalg = xp.einsum("wra,wqa->wrq", c1g_mat, calg_a, optimize=True)
    cholg_c1gcalg = xp.einsum("wXsr,wrq->wXsq", cholg_mat, c1g_matcalg, optimize=True)
    c2g_dir_calg = xp.einsum("wpa,wia->wip", calg_a, c2g_dir, optimize=True)
    c2g_cross_calg = xp.einsum("wpa,wja->wjp", calg_a, c2g_cross, optimize=True)
    c2gcalG = 2.0 * c2g_dir_calg - c2g_cross_calg
    c2gcalGG = xp.einsum("wip,wir->wpr", c2gcalG, walkers.ghalfa, optimize=True)
    lc2gcalGG_c = xp.einsum("prX,wpr->wX", chol, c2gcalGG, optimize=True)
    lc2gcalGG_x = xp.einsum("qsX,wqr->wXsr", chol, c2gcalGG, optimize=True)
    cholgT = xp.einsum("prX,wir->wXpi", chol, walkers.ghalfa, optimize=True)
    cholgTcalg = xp.einsum("wXpi,wpa->wXai", cholgT, calg_a, optimize=True)

    e1_0 = 2.0 * h1g
    e1_1_1 = 4.0 * xp.einsum("w,w->w", h1g, c1gov, optimize=True)
    e1_1_2 = -2.0 * xp.einsum(
        "pq,ia,wiq,wpa->w", hamiltonian.h1e[0], trial.c1a, walkers.ghalfa, calg_a, optimize=True
    )
    e1_2_1 = 4.0 * xp.einsum("w,wjb,wjb->w", h1g, c2g_dir, gova, optimize=True)
    e1_2_2 = -2.0 * xp.einsum("w,wja,wja->w", h1g, c2g_cross, gova, optimize=True)
    e1_2_3 = -4.0 * xp.einsum("pq,wpq->w", hamiltonian.h1e[0], calg_c2g_dir_G, optimize=True)
    e1_2_4 = 2.0 * xp.einsum("pq,wpq->w", hamiltonian.h1e[0], calg_c2g_cross_G, optimize=True)
    e1 = e1_0 + e1_1_1 + e1_1_2 + e1_2_1 + e1_2_2 + e1_2_3 + e1_2_4

    e2_0_1 = 2.0 * xp.einsum("wX,wX->w", chol_g, chol_g, optimize=True)
    e2_0_2 = -1.0 * xp.einsum("wXsr,wXrs->w", cholg_mat, cholg_mat, optimize=True)
    e2_1_1 = 2.0 * c1gov * (e2_0_1 + e2_0_2)
    e2_1_2 = -4.0 * xp.einsum("wX,wX->w", lc1calgG, chol_g, optimize=True)
    e2_1_3 = 2.0 * xp.einsum("wXsq,qsX->w", cholg_c1gcalg, chol, optimize=True)
    c2g_dir_g = xp.einsum("wjb,wjb->w", c2g_dir, gova, optimize=True)
    c2g_cross_g = xp.einsum("wja,wja->w", c2g_cross, gova, optimize=True)
    e2_2_1 = (e2_0_1 + e2_0_2) * (2.0 * c2g_dir_g - c2g_cross_g)
    e2_2_2 = -4.0 * xp.einsum("wX,wX->w", lc2gcalGG_c, chol_g, optimize=True)
    e2_2_3 = 2.0 * xp.einsum("wXsr,wXsr->w", cholg_mat, lc2gcalGG_x, optimize=True)
    e2_2_4 = 2.0 * xp.einsum(
        "iajb,wXai,wXbj->w", trial.c2aa, cholgTcalg, cholgTcalg, optimize=True
    )
    e2_2_5 = -1.0 * xp.einsum(
        "iajb,wXaj,wXbi->w", trial.c2aa, cholgTcalg, cholgTcalg, optimize=True
    )
    e2 = e2_0_1 + e2_0_2 + e2_1_1 + e2_1_2 + e2_1_3 + e2_2_1 + e2_2_2 + e2_2_3 + e2_2_4 + e2_2_5

    ovlp_ratio = trial.ovlp_ratio_cisd
    e1b = e1 / ovlp_ratio
    e2b = e2 / ovlp_ratio
    nwalkers = walkers.nwalkers
    energy = xp.zeros((nwalkers, 3), dtype=xp.complex128)
    energy[:, 0] = e1b + e2b + hamiltonian.ecore
    energy[:, 1] = e1b + hamiltonian.ecore
    energy[:, 2] = e2b
    return energy

def local_energy_ucisd_batch(hamiltonian: GenericRealChol, walkers: UHFWalkers, trial: CISD):
    """Calculate local energy for UCISD trial wavefunction.

    Parameters
    ----------
    hamiltonian : object
        Hamiltonian object.
    walkers : object
        WalkerBatch object (this stores some intermediates for the particular trial wfn).
    trial : object
        Trial wavefunction object.

    Returns
    -------
    eloc : float / complex
        Local energy.
    """
    if hamiltonian.h1eb is None or hamiltonian.cholb is None:
        hamiltonian.construct_beta_integrals(trial.mo_coeffb)
    h1eb = hamiltonian.h1eb
    chol = hamiltonian.chol.reshape((hamiltonian.nbasis, hamiltonian.nbasis, hamiltonian.nchol))
    cholb = hamiltonian.cholb

    gova = walkers.ghalfa[:, :, trial.nalpha:]
    govb = walkers.ghalfb[:, :, trial.nbeta:]
    nvira = hamiltonian.nbasis - trial.nalpha
    nvirb = hamiltonian.nbasis - trial.nbeta
    eye_a = -xp.broadcast_to(xp.eye(nvira)[None, :, :], (gova.shape[0], nvira, nvira))
    eye_b = -xp.broadcast_to(xp.eye(nvirb)[None, :, :], (govb.shape[0], nvirb, nvirb))
    calg_a = xp.concatenate([gova, eye_a], axis=1)
    calg_b = xp.concatenate([govb, eye_b], axis=1)

    h1g_a = xp.einsum("kq,wkq->w", trial.rh1a, walkers.ghalfa, optimize=True)
    h1g_b = xp.einsum("kq,wkq->w", trial.rh1b, walkers.ghalfb, optimize=True)
    hg = h1g_a + h1g_b
    c1g_a = xp.einsum("ia,wia->w", trial.c1a, gova, optimize=True)
    c1g_b = xp.einsum("ia,wia->w", trial.c1b, govb, optimize=True)
    c1g = c1g_a + c1g_b
    c1calg_a = xp.einsum("ia,wpa->wpi", trial.c1a, calg_a, optimize=True)
    c1calg_b = xp.einsum("ia,wpa->wpi", trial.c1b, calg_b, optimize=True)
    c1calgG_a = xp.einsum("wpi,wiq->wpq", c1calg_a, walkers.ghalfa, optimize=True)
    c1calgG_b = xp.einsum("wpi,wiq->wpq", c1calg_b, walkers.ghalfb, optimize=True)
    chol_g_a = xp.einsum("kqX,wkq->wX", trial.rchola, walkers.ghalfa, optimize=True)
    chol_g_b = xp.einsum("kqX,wkq->wX", trial.rcholb, walkers.ghalfb, optimize=True)
    chol_g = chol_g_a + chol_g_b
    cholg_mat_a = xp.einsum("krX,wks->wXsr", trial.rchola, walkers.ghalfa, optimize=True)
    cholg_mat_b = xp.einsum("krX,wks->wXsr", trial.rcholb, walkers.ghalfb, optimize=True)
    lc1calgG_a = xp.einsum("pqX,wpq->wX", chol, c1calgG_a, optimize=True)
    lc1calgG_b = xp.einsum("pqX,wpq->wX", cholb, c1calgG_b, optimize=True)
    c1g_mat_a = xp.einsum("ia,wir->wra", trial.c1a, walkers.ghalfa, optimize=True)
    c1g_mat_b = xp.einsum("ia,wir->wra", trial.c1b, walkers.ghalfb, optimize=True)
    c1g_matcalg_a = xp.einsum("wra,wqa->wrq", c1g_mat_a, calg_a, optimize=True)
    c1g_matcalg_b = xp.einsum("wra,wqa->wrq", c1g_mat_b, calg_b, optimize=True)
    cholg_c1gcalg_a = xp.einsum("wXsr,wrq->wXsq", cholg_mat_a, c1g_matcalg_a, optimize=True)
    cholg_c1gcalg_b = xp.einsum("wXsr,wrq->wXsq", cholg_mat_b, c1g_matcalg_b, optimize=True)
    gl_a = xp.einsum("wpr,rqX->wpqX", walkers.ghalfa, chol, optimize=True)
    gl_b = xp.einsum("wpr,rqX->wpqX", walkers.ghalfb, cholb, optimize=True)
    cholgT_a = xp.einsum("prX,wir->wXpi", chol, walkers.ghalfa, optimize=True)
    cholgT_b = xp.einsum("prX,wir->wXpi", cholb, walkers.ghalfb, optimize=True)
    cholgTcalg_a = xp.einsum("wXpi,wpa->wXai", cholgT_a, calg_a, optimize=True)
    cholgTcalg_b = xp.einsum("wXpi,wpa->wXai", cholgT_b, calg_b, optimize=True)

    if trial.c2_antisymm:
        c2g_aa = xp.einsum("iajb,wia->wjb", trial.c2aa, gova, optimize=True)
        c2g_bb = xp.einsum("iajb,wia->wjb", trial.c2bb, govb, optimize=True)
        c2g_ab_a = xp.einsum("iajb,wjb->wia", trial.c2ab, govb, optimize=True)
        c2g_ab_b = xp.einsum("iajb,wia->wjb", trial.c2ab, gova, optimize=True)
        gci2g_aa_c = 0.5 * xp.einsum("wia,wia->w", c2g_aa, gova, optimize=True)
        gci2g_bb_c = 0.5 * xp.einsum("wia,wia->w", c2g_bb, govb, optimize=True)
        gci2g_aa_x = xp.zeros_like(gci2g_aa_c)
        gci2g_bb_x = xp.zeros_like(gci2g_bb_c)
        gci2g_ab = xp.einsum("wia,wia->w", c2g_ab_a, gova, optimize=True)
        gci2g = gci2g_aa_c + gci2g_bb_c + gci2g_ab
        c2g_aa_G = xp.einsum("wia,wiq->wqa", c2g_aa, walkers.ghalfa, optimize=True)
        c2g_bb_G = xp.einsum("wia,wiq->wqa", c2g_bb, walkers.ghalfb, optimize=True)
        c2g_ab_a_G = xp.einsum("wia,wiq->wqa", c2g_ab_a, walkers.ghalfa, optimize=True)
        c2g_ab_b_G = xp.einsum("wia,wiq->wqa", c2g_ab_b, walkers.ghalfb, optimize=True)
        calg_c2g_aa_G = xp.einsum("wpa,wqa->wpq", calg_a, c2g_aa_G, optimize=True)
        calg_c2g_bb_G = xp.einsum("wpa,wqa->wpq", calg_b, c2g_bb_G, optimize=True)
        calg_c2g_ab_a_G = xp.einsum("wpa,wqa->wpq", calg_a, c2g_ab_a_G, optimize=True)
        calg_c2g_ab_b_G = xp.einsum("wpa,wqa->wpq", calg_b, c2g_ab_b_G, optimize=True)
        e2_2_4_aa = 0.5 * xp.einsum(
            "iajb,wXai,wXbj->w", trial.c2aa, cholgTcalg_a, cholgTcalg_a, optimize=True
        )
        e2_2_4_bb = 0.5 * xp.einsum(
            "iajb,wXai,wXbj->w", trial.c2bb, cholgTcalg_b, cholgTcalg_b, optimize=True
        )
        e2_2_5_aa = xp.zeros_like(e2_2_4_aa)
        e2_2_5_bb = xp.zeros_like(e2_2_4_bb)
        e1_2_1 = hg * gci2g
        e1_2_2 = xp.zeros_like(e1_2_1)
        e1_2_3_aa_c = -xp.einsum("pq,wpq->w", hamiltonian.h1e[0], calg_c2g_aa_G, optimize=True)
        e1_2_3_bb_c = -xp.einsum("pq,wpq->w", h1eb, calg_c2g_bb_G, optimize=True)
        e1_2_3_ab_a = -xp.einsum("pq,wpq->w", hamiltonian.h1e[0], calg_c2g_ab_a_G, optimize=True)
        e1_2_3_ab_b = -xp.einsum("pq,wpq->w", h1eb, calg_c2g_ab_b_G, optimize=True)
        e1_2_4_aa_x = xp.zeros_like(e1_2_1)
        e1_2_4_bb_x = xp.zeros_like(e1_2_1)
    else:
        c2g_aa_c = xp.einsum("iajb,wia->wjb", trial.c2aa, gova, optimize=True)
        c2g_bb_c = xp.einsum("iajb,wia->wjb", trial.c2bb, govb, optimize=True)
        c2g_aa_x = xp.einsum("iajb,wib->wja", trial.c2aa, gova, optimize=True)
        c2g_bb_x = xp.einsum("iajb,wib->wja", trial.c2bb, govb, optimize=True)
        c2g_ab_a = xp.einsum("iajb,wjb->wia", trial.c2ab, govb, optimize=True)
        c2g_ab_b = xp.einsum("iajb,wia->wjb", trial.c2ab, gova, optimize=True)
        gci2g_aa_c = 0.5 * xp.einsum("wia,wia->w", c2g_aa_c, gova, optimize=True)
        gci2g_aa_x = -0.5 * xp.einsum("wia,wia->w", c2g_aa_x, gova, optimize=True)
        gci2g_bb_c = 0.5 * xp.einsum("wia,wia->w", c2g_bb_c, govb, optimize=True)
        gci2g_bb_x = -0.5 * xp.einsum("wia,wia->w", c2g_bb_x, govb, optimize=True)
        gci2g_ab = xp.einsum("wia,wia->w", c2g_ab_a, gova, optimize=True)
        gci2g = gci2g_aa_c + gci2g_aa_x + gci2g_bb_c + gci2g_bb_x + gci2g_ab
        c2g_aa_G_c = 0.5 * xp.einsum("wia,wiq->wqa", c2g_aa_c, walkers.ghalfa, optimize=True)
        c2g_aa_G_x = -0.5 * xp.einsum("wja,wjq->wqa", c2g_aa_x, walkers.ghalfa, optimize=True)
        c2g_bb_G_c = 0.5 * xp.einsum("wia,wiq->wqa", c2g_bb_c, walkers.ghalfb, optimize=True)
        c2g_bb_G_x = -0.5 * xp.einsum("wja,wjq->wqa", c2g_bb_x, walkers.ghalfb, optimize=True)
        c2g_ab_a_G = xp.einsum("wia,wiq->wqa", c2g_ab_a, walkers.ghalfa, optimize=True)
        c2g_ab_b_G = xp.einsum("wia,wiq->wqa", c2g_ab_b, walkers.ghalfb, optimize=True)
        calg_c2g_aa_G_c = xp.einsum("wpa,wqa->wpq", calg_a, c2g_aa_G_c, optimize=True)
        calg_c2g_aa_G_x = xp.einsum("wpa,wqa->wpq", calg_a, c2g_aa_G_x, optimize=True)
        calg_c2g_bb_G_c = xp.einsum("wpa,wqa->wpq", calg_b, c2g_bb_G_c, optimize=True)
        calg_c2g_bb_G_x = xp.einsum("wpa,wqa->wpq", calg_b, c2g_bb_G_x, optimize=True)
        calg_c2g_ab_a_G = xp.einsum("wpa,wqa->wpq", calg_a, c2g_ab_a_G, optimize=True)
        calg_c2g_ab_b_G = xp.einsum("wpa,wqa->wpq", calg_b, c2g_ab_b_G, optimize=True)
        e2_2_4_aa = 0.5 * xp.einsum(
            "iajb,wXai,wXbj->w", trial.c2aa, cholgTcalg_a, cholgTcalg_a, optimize=True
        )
        e2_2_5_aa = -0.5 * xp.einsum(
            "iajb,wXaj,wXbi->w", trial.c2aa, cholgTcalg_a, cholgTcalg_a, optimize=True
        )
        e2_2_4_bb = 0.5 * xp.einsum(
            "iajb,wXai,wXbj->w", trial.c2bb, cholgTcalg_b, cholgTcalg_b, optimize=True
        )
        e2_2_5_bb = -0.5 * xp.einsum(
            "iajb,wXaj,wXbi->w", trial.c2bb, cholgTcalg_b, cholgTcalg_b, optimize=True
        )
        e1_2_1 = hg * (gci2g_aa_c + gci2g_bb_c + gci2g_ab)
        e1_2_2 = hg * (gci2g_aa_x + gci2g_bb_x)
        e1_2_3_aa_c = -xp.einsum(
            "pq,wpa,wia,wiq->w", hamiltonian.h1e[0], calg_a, c2g_aa_c, walkers.ghalfa, optimize=True
        )
        e1_2_3_bb_c = -xp.einsum(
            "pq,wpa,wia,wiq->w", h1eb, calg_b, c2g_bb_c, walkers.ghalfb, optimize=True
        )
        e1_2_3_ab_a = -xp.einsum("pq,wpq->w", hamiltonian.h1e[0], calg_c2g_ab_a_G, optimize=True)
        e1_2_3_ab_b = -xp.einsum("pq,wpq->w", h1eb, calg_c2g_ab_b_G, optimize=True)
        e1_2_4_aa_x = xp.einsum(
            "pq,wpa,wja,wjq->w", hamiltonian.h1e[0], calg_a, c2g_aa_x, walkers.ghalfa, optimize=True
        )
        e1_2_4_bb_x = xp.einsum(
            "pq,wpa,wja,wjq->w", h1eb, calg_b, c2g_bb_x, walkers.ghalfb, optimize=True
        )

    e1_0 = hg
    e1_1_1 = c1g * hg
    e1_1_2 = -xp.einsum("pq,wpq->w", hamiltonian.h1e[0], c1calgG_a, optimize=True)
    e1_1_2 -= xp.einsum("pq,wpq->w", h1eb, c1calgG_b, optimize=True)
    e1_2_3 = e1_2_3_aa_c + e1_2_3_bb_c + e1_2_3_ab_a + e1_2_3_ab_b
    e1_2_4 = e1_2_4_aa_x + e1_2_4_bb_x
    e2_2_4_ab = xp.einsum("iajb,wXai,wXbj->w", trial.c2ab, cholgTcalg_a, cholgTcalg_b, optimize=True)
    e2_0_1 = 0.5 * xp.einsum("wX,wX->w", chol_g, chol_g, optimize=True)
    e2_0_2 = -0.5 * (
        xp.einsum("wXsr,wXrs->w", cholg_mat_a, cholg_mat_a, optimize=True)
        + xp.einsum("wXsr,wXrs->w", cholg_mat_b, cholg_mat_b, optimize=True)
    )
    e2_1_1 = (e2_0_1 + e2_0_2) * c1g
    e2_1_2 = -xp.einsum("wX,wX->w", lc1calgG_a + lc1calgG_b, chol_g, optimize=True)
    e2_1_3 = xp.einsum("wXsq,qsX->w", cholg_c1gcalg_a, chol, optimize=True)
    e2_1_3 += xp.einsum("wXsq,qsX->w", cholg_c1gcalg_b, cholb, optimize=True)
    e2_2_1 = (e2_0_1 + e2_0_2) * gci2g

    if trial.c2_antisymm:
        e2_2_2 = -xp.einsum(
            "wX,wX->w",
            xp.einsum("pqX,wpq->wX", chol, calg_c2g_aa_G + calg_c2g_ab_a_G, optimize=True),
            chol_g,
            optimize=True,
        )
        e2_2_2 -= xp.einsum(
            "wX,wX->w",
            xp.einsum("pqX,wpq->wX", cholb, calg_c2g_bb_G + calg_c2g_ab_b_G, optimize=True),
            chol_g,
            optimize=True,
        )
        e2_2_3 = xp.einsum(
            "wpqX,prX,wqr->w", gl_a, trial.rchola, calg_c2g_aa_G + calg_c2g_ab_a_G, optimize=True
        )
        e2_2_3 += xp.einsum(
            "wpqX,prX,wqr->w", gl_b, trial.rcholb, calg_c2g_bb_G + calg_c2g_ab_b_G, optimize=True
        )
    else:
        e2_2_2 = -xp.einsum(
            "pqX,wpa,wia,wiq,wX->w", chol, calg_a, c2g_aa_c, walkers.ghalfa, chol_g, optimize=True
        )
        e2_2_2 -= xp.einsum(
            "pqX,wpa,wia,wiq,wX->w", cholb, calg_b, c2g_bb_c, walkers.ghalfb, chol_g, optimize=True
        )
        e2_2_2 += xp.einsum(
            "pqX,wpa,wja,wjq,wX->w", chol, calg_a, c2g_aa_x, walkers.ghalfa, chol_g, optimize=True
        )
        e2_2_2 += xp.einsum(
            "pqX,wpa,wja,wjq,wX->w", cholb, calg_b, c2g_bb_x, walkers.ghalfb, chol_g, optimize=True
        )
        e2_2_2 -= xp.einsum(
            "wX,wX->w", xp.einsum("pqX,wpq->wX", chol, calg_c2g_ab_a_G, optimize=True), chol_g, optimize=True
        )
        e2_2_2 -= xp.einsum(
            "wX,wX->w", xp.einsum("pqX,wpq->wX", cholb, calg_c2g_ab_b_G, optimize=True), chol_g, optimize=True
        )
        e2_2_3 = 2.0 * xp.einsum(
            "wpqX,prX,wqr->w", gl_a, trial.rchola, calg_c2g_aa_G_c, optimize=True
        )
        e2_2_3 += 2.0 * xp.einsum(
            "wpqX,prX,wqr->w", gl_b, trial.rcholb, calg_c2g_bb_G_c, optimize=True
        )
        e2_2_3 += 2.0 * xp.einsum(
            "wpqX,prX,wqr->w", gl_a, trial.rchola, calg_c2g_aa_G_x, optimize=True
        )
        e2_2_3 += 2.0 * xp.einsum(
            "wpqX,prX,wqr->w", gl_b, trial.rcholb, calg_c2g_bb_G_x, optimize=True
        )
        e2_2_3 += xp.einsum("wpqX,prX,wqr->w", gl_a, trial.rchola, calg_c2g_ab_a_G, optimize=True)
        e2_2_3 += xp.einsum("wpqX,prX,wqr->w", gl_b, trial.rcholb, calg_c2g_ab_b_G, optimize=True)

    e2_2_4 = e2_2_4_aa + e2_2_4_bb + e2_2_4_ab
    e2_2_5 = e2_2_5_aa + e2_2_5_bb
    e1 = e1_0 + e1_1_1 + e1_1_2 + e1_2_1 + e1_2_2 + e1_2_3 + e1_2_4
    e2 = e2_0_1 + e2_0_2 + e2_1_1 + e2_1_2 + e2_1_3 + e2_2_1 + e2_2_2 + e2_2_3 + e2_2_4 + e2_2_5
    ovlp_ratio = trial.ovlp_ratio_cisd
    e1b = e1 / ovlp_ratio
    e2b = e2 / ovlp_ratio
    nwalkers = walkers.nwalkers
    energy = xp.zeros((nwalkers, 3), dtype=xp.complex128)
    energy[:, 0] = e1b + e2b + hamiltonian.ecore
    energy[:, 1] = e1b + hamiltonian.ecore
    energy[:, 2] = e2b
    return energy

def local_energy_cisd_batch(system, hamiltonian: GenericRealChol, walkers: UHFWalkers, trial: CISD):
    """Calculate local energy for CISD trial wavefunction.

    Parameters
    ----------
    hamiltonian : object
        Hamiltonian object.
    walkers : object
        WalkerBatch object (this stores some intermediates for the particular trial wfn).
    trial : object
        Trial wavefunction object.

    Returns
    -------
    eloc : float / complex
        Local energy.
    """
    if walkers.rhf:
        return local_energy_rcisd_batch(hamiltonian, walkers, trial)
    else:
        return local_energy_ucisd_batch(hamiltonian, walkers, trial)