from numba import jit
import numpy as np

def cart2frac(reciprocal_vectors, kpts):
    """
    Convert k-points from Cartesian to fractional coordinates.
    kpts: (nkpts, 3) array of k-points in Cartesian coordinates
    cell: (3, 3) array of lattice vectors in Cartesian coordinates
    """
    b_inv = np.linalg.inv(reciprocal_vectors)
    kpts_frac = np.dot(kpts, b_inv)
    return kpts_frac

@jit(nopython=True, fastmath=True)
def BZ_to_1BZ(kpts):
    """
    Map k-points to the first Brillouin zone.
    kpts: (nkpts, 3) array of k-points in fractional coordinates
    """
    kpts = np.floor(0.5 - kpts) + kpts
    return kpts

@jit(nopython=True, fastmath=True)
def find_translated_index(kpt, q_vec, kpts_list, tol=1e-6):
    """
    Find the index of the k-point that is translated by trs_vector for the whole k point lists
    kpts: (nkpts, 3) array of k-points in fractional coordinates
    trs_vector: (3,) array of the translation vector in fractional coordinates
    """
    # assert np.max(np.abs(kpts)) < 0.5
    # here we do not do sanity check, make sure the kpts are in the first BZ
    kpt_translated = kpt + q_vec
    fbz_kpt_trs = BZ_to_1BZ(kpt_translated)
    # print(f"fbz_kpts_trs = {fbz_kpt_trs}")

    for j in range(len(kpts_list)):
        if np.allclose(fbz_kpt_trs, kpts_list[j], atol=tol):
            return j

@jit(nopython=True, fastmath=True)
def find_translated_index_batched(kpts, q_vec, tol=1e-6):
    """
    Find the index of the k-point that is translated by trs_vector for the whole k point lists
    kpts: (nkpts, 3) array of k-points in fractional coordinates
    trs_vector: (3,) array of the translation vector in fractional coordinates
    """
    # assert np.max(np.abs(kpts)) < 0.5
    # here we do not do sanity check, make sure the kpts are in the first BZ
    idxlis = []
    kpts_translated = kpts + q_vec
    fbz_kpts_trs = BZ_to_1BZ(kpts_translated)
    # print(fbz_kpts_trs)

    for i in range(fbz_kpts_trs.shape[0]):
        kpt = fbz_kpts_trs[i]
        for j in range(len(kpts)):
            if np.allclose(kpt, kpts[j], atol=tol):
                idxlis.append(j)
    idxlis = np.array(idxlis, dtype=np.int64)
    return idxlis

@jit(nopython=True, fastmath=True)
def find_inverted_index(kpt, kpts_list, tol=1e-6):
    """
    Find the index of the k-point that is transformed to -k for
    kpts: (nkpts, 3) array of k-points in fractional coordinates
    trs_vector: (3,) array of the translation vector in fractional coordinates
    """
    # assert np.max(np.abs(kpts)) < 0.5
    # here we do not do sanity check, make sure the kpts are in the first BZ
    mkpt = -kpt
    fbz_mkpt = BZ_to_1BZ(mkpt)
    for j in range(len(kpts_list)):
        if np.allclose(fbz_mkpt, kpts_list[j], atol=tol):
            return j

@jit(nopython=True, fastmath=True)
def find_inverted_index_batched(kpts, tol=1e-6):
    """
    Find the index of the k-point that is transformed to -k for
    kpts: (nkpts, 3) array of k-points in fractional coordinates
    trs_vector: (3,) array of the translation vector in fractional coordinates
    """
    # assert np.max(np.abs(kpts)) < 0.5
    # here we do not do sanity check, make sure the kpts are in the first BZ
    idxlis = []
    mkpts = -kpts
    fbz_mkpts = BZ_to_1BZ(mkpts)
    for i in range(fbz_mkpts.shape[0]):
        fbz_mkpt = fbz_mkpts[i]
        for j in range(len(kpts)):
            if np.allclose(fbz_mkpt, kpts[j], atol=tol):
                idxlis.append(j)
    idxlis = np.array(idxlis, dtype=np.int64)
    return idxlis


def find_gamma_pt(kpt):
    """
    Find the gamma point index
    kpt: (nk, 3) array of k-points in fractional coordinates
    """
    for i in range(kpt.shape[0]):
        if np.allclose(kpt[i], 0.0):
            return i

@jit(nopython=True, fastmath=True)
def get_walker_from_trial(trial_wfn):
    """
    Get initial walker from trial wavefunction
    trial_wfn: numpy.ndarray with shape (nk, nbasis, nocc)
    
    Returns:
    walkers : numpy.ndarray with shape (nk, nbasis, nk, nocc)
    """
    assert len(trial_wfn.shape) == 3
    nk, nbasis, nocc = trial_wfn.shape
    walkers = np.zeros((nk, nbasis, nk, nocc), dtype=np.complex128)
    for i in range(nk):
        walkers[i, :, i, :] = trial_wfn[i]
    return walkers