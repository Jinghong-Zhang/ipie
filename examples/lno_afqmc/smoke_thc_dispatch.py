"""Smoke test: the GenericRealTHC plum-dispatch overloads import and resolve."""
import numpy
from ipie.hamiltonians.thc import GenericRealTHC
from ipie.propagation.phaseless_generic import PhaselessGeneric
from ipie.propagation.phaseless_base import (
    construct_mean_field_shift,
    construct_one_body_propagator,
)
from ipie.propagation.force_bias import _force_bias_thc  # import => registration ok

numpy.random.seed(1)
nb, Nmu = 6, 24
X = numpy.random.randn(nb, Nmu)
A = numpy.random.randn(Nmu, Nmu)
M = A @ A.T + numpy.eye(Nmu)                       # PSD
h1 = numpy.random.randn(nb, nb); h1 = 0.5 * (h1 + h1.T)
ham = GenericRealTHC(numpy.array([h1, h1]), X, M, verbose=True)
print("imports + GenericRealTHC build OK; nfields =", ham.nfields)

# construct_VHS dispatches on (self, GenericRealTHC, xshifted) -- no trial needed
prop = PhaselessGeneric(0.01)
prop.sqrt_dt = 0.1
prop.isqrt_dt = 1j * 0.1
xshift = numpy.random.randn(ham.nfields, 3) + 0j
VHS = prop.construct_VHS(ham, xshift)
assert VHS.shape == (3, nb, nb), VHS.shape
print("construct_VHS dispatch OK, shape", VHS.shape)

# mean-field shift + one-body need a TrialWavefunctionBase; emulate by registering
# a stub subclass so plum matches the (GenericRealTHC, TrialWavefunctionBase) overload.
from ipie.trial_wavefunction.wavefunction_base import TrialWavefunctionBase
class _StubTrial(TrialWavefunctionBase):
    def __init__(self, psi):
        self.psi0a = psi; self.psi0b = psi
        self.G = numpy.array([psi @ psi.T, psi @ psi.T])
    def calc_force_bias(self, *a, **k): pass
    def build(self, *a, **k): pass
    def half_rotate(self, *a, **k): pass
    def calc_greens_function(self, *a, **k): pass
    def calc_overlap(self, *a, **k): pass
psi = numpy.linalg.qr(numpy.random.randn(nb, nb))[0][:, :2]
trial = _StubTrial(psi)
mf = construct_mean_field_shift(ham, trial)
print("construct_mean_field_shift dispatch OK, shape", numpy.asarray(mf).shape)
expH1 = construct_one_body_propagator(ham, numpy.asarray(mf), 0.01)
print("construct_one_body_propagator dispatch OK, shape", numpy.asarray(expH1).shape)
print("SMOKE OK")
