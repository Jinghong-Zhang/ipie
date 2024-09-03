from ipie.hamiltonians.generic import GenericRealChol, GenericComplexChol
from ipie.hamiltonians.kpt_hamiltonian import KptComplexChol, KptComplexCholSymm
from ipie.hamiltonians.generic_chunked import GenericRealCholChunked
from ipie.propagation.phaseless_generic import PhaselessGeneric, PhaselessGenericChunked
from ipie.propagation.phaseless_kpt import PhaselessKptChol

# Propagator = {GenericRealChol: PhaselessGeneric, GenericComplexChol: PhaselessGeneric}
Propagator = {
    GenericRealChol: PhaselessGeneric,
    GenericComplexChol: PhaselessGeneric,
    GenericRealCholChunked: PhaselessGenericChunked,
    KptComplexChol: PhaselessKptChol,
    KptComplexCholSymm: PhaselessKptChol,
}
