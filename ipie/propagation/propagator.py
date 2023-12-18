from ipie.hamiltonians.generic import GenericRealChol, GenericComplexChol
from ipie.hamiltonians.sparse import SparseRealChol, SparseComplexChol,  SparseNonHermitian
from ipie.propagation.phaseless_generic import PhaselessGeneric

Propagator = {GenericRealChol: PhaselessGeneric, GenericComplexChol: PhaselessGeneric, SparseNonHermitian:PhaselessGeneric}
