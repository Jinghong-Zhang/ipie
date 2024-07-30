from ipie.hamiltonians.generic import GenericRealChol, GenericComplexChol
from ipie.hamiltonians.kpt_hamiltonian import KptComplexChol
from ipie.hamiltonians.generic_chunked import GenericRealCholChunked
from ipie.propagation.phaseless_generic import PhaselessGeneric, PhaselessGenericChunked
from ipie.propagation.phaseless_kpt import PhaselessKpt

# Propagator = {GenericRealChol: PhaselessGeneric, GenericComplexChol: PhaselessGeneric}
Propagator = {
    GenericRealChol: PhaselessGeneric,
    GenericComplexChol: PhaselessGeneric,
    GenericRealCholChunked: PhaselessGenericChunked,
    KptComplexChol: PhaselessKpt,
}
