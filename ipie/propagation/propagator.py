from ipie.hamiltonians.generic import GenericRealChol, GenericComplexChol
from ipie.hamiltonians.kpt_hamiltonian import KptComplexChol, KptComplexCholSymm
from ipie.hamiltonians.generic_chunked import GenericRealCholChunked
from ipie.hamiltonians.kpt_chunked import KptComplexCholChunked
from ipie.hamiltonians.thc import GenericRealTHC, GenericRealTHCUhf, GenericComplexTHC
from ipie.propagation.phaseless_generic import PhaselessGeneric, PhaselessGenericChunked
from ipie.propagation.phaseless_kpt import PhaselessKptChol, PhaselessKptCholChunked


# Propagator = {GenericRealChol: PhaselessGeneric, GenericComplexChol: PhaselessGeneric}
Propagator = {
    GenericRealChol: PhaselessGeneric,
    GenericComplexChol: PhaselessGeneric,
    GenericRealCholChunked: PhaselessGenericChunked,
    GenericRealTHC: PhaselessGeneric,   # factored THC uses the generic phaseless propagator
    GenericRealTHCUhf: PhaselessGeneric,  # per-spin-basis THC (UHF); apply_VHS overload routes it
    GenericComplexTHC: PhaselessGeneric,  # complex (no-TRS metal) THC; construct_VHS overload routes it
    KptComplexChol: PhaselessKptChol,
    KptComplexCholSymm: PhaselessKptChol,
    KptComplexCholChunked: PhaselessKptCholChunked
}
