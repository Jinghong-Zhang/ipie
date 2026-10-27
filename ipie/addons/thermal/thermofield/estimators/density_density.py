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
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Authors: Jinghong Zhang <jinghongzhang@fas.harvard.edu>
#
import numpy

from ipie.addons.thermal.thermofield.estimators.hubbard import TraceRatioObservables
from ipie.addons.thermal.thermofield.hamiltonians.density_density import pair_table
from ipie.utils.backend import to_host


class DensityDensityObservables(TraceRatioObservables):
    r"""Equal-time observables of a density-density model.

    Energy, density, :math:`M_z = N_\uparrow - N_\downarrow` and its second and
    fourth moments, the atom average of :math:`\langle m_I^2 \rangle` with
    :math:`m_I` the moment of atom I, and the occupations per orbital. The
    moments come from the counting statistics of each walker's Gaussian,

    .. math::

        \log \langle e^{i\phi M_z} \rangle = \sum_s \log\det\left[I + (e^{\pm i\phi} - 1) G_s\right],

    whose cumulants are traces of powers of G (no Hermitization: the
    contraction need not be Hermitian). The Binder cumulant
    :math:`1 - \langle M_z^4 \rangle / (3 \langle M_z^2 \rangle^2)` is formed after
    the ensemble ratio.

    Parameters
    ----------
    h_energy : :class:`numpy.ndarray`
        One-body part counted in the energy (hopping and crystal field, no
        chemical potential), shape (nbasis, nbasis) or (2, nbasis, nbasis).
    pairs :
        Pair terms, see :func:`pair_table`.
    atoms : :class:`numpy.ndarray`
        Orbital indices of every atom, shape (natoms, norb).
    """

    def __init__(self, h_energy, pairs, atoms):
        self.atoms = numpy.asarray(atoms)
        if self.atoms.ndim != 2:
            raise ValueError("atoms must have shape (natoms, norb).")
        self.natoms, self.norb = self.atoms.shape
        self.hop = numpy.asarray(to_host(h_energy))
        self.nbasis = self.hop.shape[-1]
        if self.hop.shape == (self.nbasis, self.nbasis):
            self.hop = numpy.broadcast_to(self.hop, (2, self.nbasis, self.nbasis))
        if self.hop.shape != (2, self.nbasis, self.nbasis):
            raise ValueError("h_energy must have shape (M,M) or (2,M,M).")
        if numpy.unique(self.atoms).size != self.atoms.size or self.atoms.max() >= self.nbasis:
            raise ValueError("Atom orbitals must be distinct basis indices.")
        self.pair_spin, self.pair_orb, self.pair_V = pair_table(pairs, self.nbasis)
        self.names = (
            "energy",
            "density",
            "mz",
            "mz2",
            "mz4",
            "local_mz2",
            *(f"occupation_{k}" for k in range(self.norb)),
        )
        self.size = len(self.names)
        self.metadata = {
            "natoms": self.natoms,
            "norb": self.norb,
            "atoms": self.atoms.tolist(),
            "energy_convention": "<h_energy + sum_p V_p n_a n_b> (total)",
            "mz_convention": "M_z = N_up - N_down (total); local_mz2 = mean_I <m_I^2>",
            "occupation_convention": "per spin, averaged over atoms",
            "raw_names": list(self.names),
        }

    @staticmethod
    def _moments(G):
        """First four cumulants of N_up - N_down for the blocks G, shape (2, m, m)."""
        g2 = G @ G
        p1 = numpy.trace(G, axis1=1, axis2=2)
        p2 = numpy.trace(g2, axis1=1, axis2=2)
        p3 = numpy.sum(g2 * G.swapaxes(1, 2), axis=(1, 2))
        p4 = numpy.sum(g2 * g2.swapaxes(1, 2), axis=(1, 2))
        sign = numpy.array([1.0, -1.0])
        return (
            sign @ p1,
            numpy.sum(p1 - p2),
            sign @ (p1 - 3.0 * p2 + 2.0 * p3),
            numpy.sum(p1 - 7.0 * p2 + 12.0 * p3 - 6.0 * p4),
        )

    def measure(self, G):
        r"""Raw observable vector of one Green's function.

        Parameters
        ----------
        G : :class:`numpy.ndarray`
            Particle contraction :math:`G_{s, ji} = \langle c_i^\dagger c_j \rangle`,
            shape (2, nbasis, nbasis).

        Returns
        -------
        values : :class:`numpy.ndarray`
            Complex raw vector, shape (size,).
        """
        G = numpy.asarray(to_host(G))
        if G.shape != (2, self.nbasis, self.nbasis):
            raise ValueError("G must be the particle contraction with shape (2,M,M).")
        occ = numpy.diagonal(G, axis1=-2, axis2=-1)
        (sa, sb), (i, j) = self.pair_spin.T, self.pair_orb.T
        pair_nn = occ[sa, i] * occ[sb, j] - numpy.where(sa == sb, G[sa, i, j] * G[sa, j, i], 0.0)
        k1, k2, k3, k4 = self._moments(G)
        local = 0.0
        for orb in self.atoms:
            l1, l2, _, _ = self._moments(G[:, orb[:, None], orb[None, :]])
            local += l2 + l1 * l1
        out = numpy.empty(self.size, dtype=numpy.complex128)
        out[0] = numpy.einsum("sij,sji->", self.hop, G) + self.pair_V @ pair_nn
        out[1] = occ.sum()
        out[2] = k1
        out[3] = k2 + k1 * k1
        out[4] = k4 + 4.0 * k3 * k1 + 3.0 * k2 * k2 + 6.0 * k2 * k1 * k1 + k1**4
        out[5] = local / self.natoms
        out[6:] = occ[:, self.atoms].mean(axis=(0, 1))
        return out

    def _derived(self, mean_vector):
        values = numpy.asarray(mean_vector, dtype=numpy.complex128)
        if values.shape != (self.size,):
            raise ValueError("Mean vector does not match the raw observable schema.")
        mz2, mz4 = values[3], values[4]
        binder = 1.0 - mz4 / (3.0 * mz2 * mz2)
        m_rms = numpy.sqrt(mz2) / self.natoms
        return numpy.concatenate((values, [binder, m_rms]))

    def _as_dict(self, values):
        result = dict(zip(self.names[:6], (float(v) for v in values[:6])))
        result["occupation"] = [float(v) for v in values[6 : self.size]]
        result["binder"] = float(values[self.size])
        result["m_rms_per_atom"] = float(values[self.size + 1])
        return result
