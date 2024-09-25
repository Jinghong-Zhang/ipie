import numpy as np
from numba import jit


def apply_exponential(phi, VHS, exp_nmax):
    """Apply exponential propagator of the HS transformation
    Parameters
    ----------
    system :
        system class
    phi : numpy array
        a state
    VHS : numpy array
        HS transformation potential
    Returns
    -------
    phi : numpy array
        Exp(VHS) * phi
    """
    # Temporary array for matrix exponentiation.
    Temp = xp.zeros(phi.shape, dtype=phi.dtype)

    xp.copyto(Temp, phi)
    for n in range(1, exp_nmax + 1):
        Temp = VHS.dot(Temp) / n
        phi += Temp

    synchronize()
    return phi
    
def gemm(nwalkers, phia, VHS, exp_nmax)
    for iw in range(nwalkers):
        # 2.b Apply two-body
        phia[iw] = apply_exponential(phia[iw], VHS[iw], exp_nmax)