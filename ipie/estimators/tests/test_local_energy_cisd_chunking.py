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
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

from types import SimpleNamespace

import numpy
import pytest

import ipie.estimators.local_energy_cisd as local_energy_cisd


def _complex_random(rng, shape):
    return rng.standard_normal(shape) + 1j * rng.standard_normal(shape)


def _build_rcisd_case(seed=7):
    rng = numpy.random.default_rng(seed)

    nwalkers = 5
    nalpha = 2
    nbasis = 4
    nchol = 3
    nvira = nbasis - nalpha

    h1 = _complex_random(rng, (nbasis, nbasis))
    chol_flat = _complex_random(rng, (nbasis * nbasis, nchol))

    hamiltonian = SimpleNamespace(
        nbasis=nbasis,
        nchol=nchol,
        h1e=numpy.array([h1, h1]),
        chol=chol_flat,
        ecore=0.123,
    )

    walkers = SimpleNamespace(
        nwalkers=nwalkers,
        ghalfa=_complex_random(rng, (nwalkers, nalpha, nbasis)),
    )

    trial = SimpleNamespace(
        nalpha=nalpha,
        rh1=_complex_random(rng, (nalpha, nbasis)),
        c1a=_complex_random(rng, (nalpha, nvira)),
        c2aa=_complex_random(rng, (nalpha, nvira, nalpha, nvira)),
        rchol=_complex_random(rng, (nalpha, nbasis, nchol)),
        ovlp_ratio_cisd=1.0 + 0.1 * _complex_random(rng, (nwalkers,)),
    )

    return hamiltonian, walkers, trial


def _build_ucisd_case(seed=11):
    rng = numpy.random.default_rng(seed)

    nwalkers = 5
    nalpha = 2
    nbeta = 1
    nbasis = 4
    nchol = 3
    nvira = nbasis - nalpha
    nvirb = nbasis - nbeta

    h1a = _complex_random(rng, (nbasis, nbasis))
    h1b = _complex_random(rng, (nbasis, nbasis))

    hamiltonian = SimpleNamespace(
        nbasis=nbasis,
        nchol=nchol,
        h1e=numpy.array([h1a, h1a]),
        h1eb=h1b,
        chol=_complex_random(rng, (nbasis * nbasis, nchol)),
        cholb=_complex_random(rng, (nbasis, nbasis, nchol)),
        ecore=-0.456,
    )

    walkers = SimpleNamespace(
        nwalkers=nwalkers,
        ghalfa=_complex_random(rng, (nwalkers, nalpha, nbasis)),
        ghalfb=_complex_random(rng, (nwalkers, nbeta, nbasis)),
    )

    trial = SimpleNamespace(
        nalpha=nalpha,
        nbeta=nbeta,
        rh1a=_complex_random(rng, (nalpha, nbasis)),
        rh1b=_complex_random(rng, (nbeta, nbasis)),
        c1a=_complex_random(rng, (nalpha, nvira)),
        c1b=_complex_random(rng, (nbeta, nvirb)),
        c2aa=_complex_random(rng, (nalpha, nvira, nalpha, nvira)),
        c2bb=_complex_random(rng, (nbeta, nvirb, nbeta, nvirb)),
        c2ab=_complex_random(rng, (nalpha, nvira, nbeta, nvirb)),
        rchola=_complex_random(rng, (nalpha, nbasis, nchol)),
        rcholb=_complex_random(rng, (nbeta, nbasis, nchol)),
        ovlp_ratio_cisd=1.0 + 0.1 * _complex_random(rng, (nwalkers,)),
        c2_antisymm=False,
    )

    return hamiltonian, walkers, trial


@pytest.mark.unit
def test_local_energy_rcisd_chunking_invariant(monkeypatch):
    hamiltonian, walkers, trial = _build_rcisd_case()

    monkeypatch.setattr(local_energy_cisd, "_get_walker_chunks", lambda n, *_: [slice(0, n)])
    e_single = local_energy_cisd.local_energy_rcisd_batch(hamiltonian, walkers, trial)

    monkeypatch.setattr(
        local_energy_cisd,
        "_get_walker_chunks",
        lambda n, *_: [slice(0, 2), slice(2, 4), slice(4, n)],
    )
    e_chunked = local_energy_cisd.local_energy_rcisd_batch(hamiltonian, walkers, trial)

    assert numpy.allclose(e_single, e_chunked)


@pytest.mark.unit
@pytest.mark.parametrize("antisymm", [False, True])
def test_local_energy_ucisd_chunking_invariant(monkeypatch, antisymm):
    hamiltonian, walkers, trial = _build_ucisd_case()
    trial.c2_antisymm = antisymm

    monkeypatch.setattr(local_energy_cisd, "_get_walker_chunks", lambda n, *_: [slice(0, n)])
    e_single = local_energy_cisd.local_energy_ucisd_batch(hamiltonian, walkers, trial)

    monkeypatch.setattr(
        local_energy_cisd,
        "_get_walker_chunks",
        lambda n, *_: [slice(0, 2), slice(2, 4), slice(4, n)],
    )
    e_chunked = local_energy_cisd.local_energy_ucisd_batch(hamiltonian, walkers, trial)

    assert numpy.allclose(e_single, e_chunked)
