# Real THC (tensor hypercontraction / ISDF) Hamiltonian for LNO-AFQMC.
#
# Factorizes the cluster ERI in SEPARABLE form (kept factored throughout):
#
#     (pq|rs) = sum_{mu,nu} X[p,mu] X[q,mu] M[mu,nu] X[r,nu] X[s,nu]
#
# with collocation X (nbasis x Nmu) and a real PSD central matrix M (Nmu x Nmu),
# as exported by qcpbc lno_isdf (IsdfOrbitalSet::Full).  The auxiliary-field
# decomposition uses M = zeta zeta^T (eigendecomposition; M is PSD by
# construction -- it is a Coulomb metric), giving nfields = Ngamma <= Nmu and
# implicit Cholesky-like vectors
#
#     L_pq^gamma = sum_mu X[p,mu] X[q,mu] zeta[mu,gamma]
#
# that are NEVER materialized.  All AFQMC operations contract through X and zeta.
#
# Companion to GenericRealChol; same role in the propagator/energy but a
# different (separable) representation of the same two-body operator.

import os

import numpy

from ipie.hamiltonians.generic_base import GenericBase
from ipie.utils.backend import arraylib as xp


def _env_flag(name):
    """True if the environment variable is set to a truthy value."""
    return os.environ.get(name, "") not in ("", "0", "false", "False", "no", "off")


def _eigh_host_or_gpu(A, verbose=False):
    """Symmetric eigendecomposition of a host array; runs on the GPU via cupy
    (cusolver syevd) when one is available -- the host dsyevd on a 25k x 25k
    THC metric takes tens of minutes, the GPU ~1 min -- and falls back to
    numpy.linalg.eigh otherwise.  Returns host (w, U).

    Note: GPU and LAPACK eigenvectors of degenerate/sign-ambiguous subspaces
    differ (both valid); zeta then differs by an orthogonal recombination of
    the auxiliary fields, which is statistically equivalent but NOT
    trajectory-identical to a host-eigh run at fixed RNG seed."""
    try:
        import cupy
        if cupy.cuda.runtime.getDeviceCount() > 0:
            w_d, U_d = cupy.linalg.eigh(cupy.asarray(A))
            w, U = cupy.asnumpy(w_d), cupy.asnumpy(U_d)
            del w_d, U_d
            cupy.get_default_memory_pool().free_all_blocks()
            if verbose:
                print("# GenericRealTHC: zeta eigendecomposition on GPU (cusolver)")
            return w, U
    except Exception:
        pass
    return numpy.linalg.eigh(A)


class GenericRealTHC(GenericBase):
    r"""Real THC Hamiltonian: (pq|rs) = sum_{mu nu} X_{p mu} X_{q mu} M_{mu nu} X_{r nu} X_{s nu}.

    Parameters
    ----------
    h1e : ndarray (2, nbasis, nbasis)
        One-body integrals (per spin).
    X : ndarray (nbasis, Nmu)
        Real collocation (orbital values at the Nmu interpolation points).
    M : ndarray (Nmu, Nmu)
        Real symmetric PSD central matrix.
    ecore : float
        Constant (nuclear + frozen-core) energy.
    """

    def __init__(self, h1e, X, M, ecore=0.0, psd_tol=1e-10, verbose=False, M_prop=None):
        assert h1e.shape[0] == 2
        super().__init__(h1e, ecore, verbose)

        X = numpy.asarray(X, dtype=numpy.float64)
        M = numpy.asarray(M, dtype=numpy.float64)
        assert X.shape[0] == self.nbasis, f"X rows {X.shape[0]} != nbasis {self.nbasis}"
        assert M.shape[0] == M.shape[1] == X.shape[1]

        self.X = X                       # (nbasis x Nmu)
        self.M = 0.5 * (M + M.T)         # (Nmu x Nmu), symmetric
        self.nmu = X.shape[1]

        # Auxiliary-field factor: M = zeta zeta^T via eigendecomposition.  M is PSD
        # by construction (Coulomb kernel); drop (near-)null eigenvalues.
        # Canonical periodic-AFQMC split (KPTISDF-validated): the PROPAGATION
        # factor zeta comes from the G=0-FREE metric (M_prop) when provided;
        # the local energy / v0 one-body mod keep the G=0-INCLUDED self.M.
        M_for_prop = self.M if M_prop is None else numpy.ascontiguousarray(M_prop)
        w, U = _eigh_host_or_gpu(M_for_prop, verbose=verbose)
        wmax = max(1.0, float(w.max()))
        if w.min() < -psd_tol * wmax:
            print(f"# WARNING: THC central matrix M not PSD: min eig = {w.min():.4e}")
        keep = w > psd_tol * wmax
        self.zeta = (U[:, keep] * numpy.sqrt(numpy.clip(w[keep], 0.0, None))[None, :]).copy()  # from M_prop
        self.nchol = self.zeta.shape[1]  # Ngamma (number of THC fields)
        self.nfields = self.nchol
        self.chunked = False

        # One-body re-ordering correction (Eqn 17 of Motta17), THC form:
        #   v0_ij = 0.5 sum_k (ik|jk) = 0.5 [ X (M . (X^T X)) X^T ]_ij
        # (the k-sum gives the collocation overlap S = X^T X, Hadamard with M).
        S = self.X.T @ self.X                       # (Nmu x Nmu)
        W = self.M * S                               # Hadamard
        v0 = 0.5 * (self.X @ (W @ self.X.T))         # (nbasis x nbasis)
        h1e_mod = numpy.zeros(self.H1.shape, dtype=self.H1.dtype)
        h1e_mod[0] = self.H1[0] - v0
        h1e_mod[1] = self.H1[1] - v0
        self.h1e_mod = xp.array(h1e_mod)

        # Optional mixed-precision PROPAGATION factors (IPIE_THC_MIXED=1):
        # float32 copies of X and zeta used by construct_VHS_thc /
        # construct_force_bias_thc / propagate_one_body only.  The local-energy
        # and fragment-estimator path always contracts the fp64 self.X/self.M.
        # On GPU, additionally set CUPY_TF32=1 (before cupy is imported) so
        # float32/complex64 GEMMs use TF32 tensor cores.
        self.X32 = None
        self.zeta32 = None
        if _env_flag("IPIE_THC_MIXED"):
            self.X32 = self.X.astype(numpy.float32)
            self.zeta32 = self.zeta.astype(numpy.float32)
            if not os.environ.get("CUPY_TF32"):
                print(
                    "# GenericRealTHC: IPIE_THC_MIXED=1 (float32 propagation factors). "
                    "For TF32 tensor cores on GPU also export CUPY_TF32=1."
                )

        if verbose:
            memX = self.X.nbytes / 1024.0**3
            memM = self.M.nbytes / 1024.0**3
            print("# GenericRealTHC:")
            print(f"#   nbasis={self.nbasis}  Nmu={self.nmu}  nfields(Ngamma)={self.nfields}")
            print(f"#   X {self.X.shape} ({memX:.3f} GB), M {self.M.shape} ({memM:.3f} GB)")
            print(f"#   M eig range [{w.min():.3e}, {w.max():.3e}] (PSD)")

    # ---- factored two-body building blocks (no L_pq^gamma materialized) ----

    def rho_moment(self, Gmat):
        r"""rho_mu = sum_{pq} X_{p mu} G_{pq} X_{q mu} = colsum(X .* (G X)).

        Gmat : (nbasis, nbasis) (real or complex).  Returns (Nmu,)."""
        GX = Gmat @ self.X                      # (nbasis x Nmu)
        return numpy.sum(self.X * GX, axis=0)   # (Nmu,)

    def rho_moment_halfrot(self, Xocc, Ghalf):
        r"""Same rho but from the half-rotated collocation and Ghalf.

        rho_mu = sum_i Xocc_{i mu} (Ghalf X)_{i mu}.
        Xocc  : (nocc, Nmu) = psi_occ^T X.
        Ghalf : (nocc, nbasis).  Returns (Nmu,)."""
        B = Ghalf @ self.X                      # (nocc x Nmu)
        return numpy.sum(Xocc * B, axis=0)      # (Nmu,)

    def vbias_from_rho(self, rho):
        r"""Force-bias / mean-field moment in field space: vbias_gamma = sum_mu zeta_{mu gamma} rho_mu."""
        return self.zeta.T @ rho                # (Ngamma,)

    def half_rotate(self, psi_occ):
        r"""Half-rotated collocation Xocc = psi_occ^T X  (nocc x Nmu).  psi_occ : (nbasis x nocc)."""
        return psi_occ.T @ self.X


class GenericComplexTHC(GenericBase):
    r"""COMPLEX THC Hamiltonian for the no-TRS (general k-mesh) metal path.

    Factorization (the complex analog of GenericRealTHC, matching the
    GenericComplexChol convention (pq|rs) = sum_gamma L_pq^gamma conj(L_rs^gamma)):

        L_pq^gamma = sum_mu X[p,mu] conj(X[q,mu]) Z[mu,gamma]
        (pq|rs)    = sum_{mu,nu} X[p,mu] conj(X[q,mu]) M[mu,nu] conj(X[r,nu]) X[s,nu]

    with COMPLEX collocation X (nbasis x Nmu) and HERMITIAN PSD M (Nmu x Nmu),
    M = Z Z^H by Hermitian eigendecomposition.  L is NEVER materialized.

    Auxiliary fields.  The one-body density operator

        rho_mu = sum_pq X[p,mu] conj(X[q,mu]) a_p^dag a_q

    is HERMITIAN (the outer product X_mu X_mu^H is), so O_gamma = sum_mu Z[mu,gamma] rho_mu
    splits into Hermitian halves that involve only Re(Z) and Im(Z):

        A_gamma = (O + O^dag)/2 = sum_mu Re(Z[mu,gamma]) rho_mu
        B_gamma = i(O - O^dag)/2 = -sum_mu Im(Z[mu,gamma]) rho_mu

    and the HS two-body needs sum_gamma (A^2 + B^2), which is INVARIANT under the
    sign of B.  We therefore fold both field channels into a single REAL matrix

        Zc = [ Re(Z) , Im(Z) ]      (Nmu x 2*Ngamma, real)

    used IDENTICALLY by the propagator and the force bias:

        VHS[w]   = isqrt_dt * X diag(Zc @ x[:,w]) X^H
        vbias[w] = Zc^T @ rho[:,w]

    This removes every sign/conjugation ambiguity between the two (they share one
    matrix), and keeps Zc REAL so the real-split GEMM trick of the real kernels
    still applies to all field-space contractions -- only the X-contractions are
    genuinely complex.  nfields = 2 * Ngamma, but the fields are never enumerated.

    Crucially NO nbasis^2 x nchol tensor is ever formed (contrast
    GenericComplexChol, which materializes L, A and B), so memory is
    O(nbasis*Nmu + Nmu^2) -- the whole point of this class.
    """

    def __init__(self, h1e, X, M, ecore=0.0, psd_tol=1e-10, verbose=False, M_prop=None):
        assert h1e.shape[0] == 2
        super().__init__(h1e, ecore, verbose)

        X = numpy.asarray(X, dtype=numpy.complex128)
        M = numpy.asarray(M, dtype=numpy.complex128)
        assert X.shape[0] == self.nbasis, f"X rows {X.shape[0]} != nbasis {self.nbasis}"
        assert M.shape[0] == M.shape[1] == X.shape[1]

        self.X = X                            # (nbasis x Nmu) complex
        self.M = 0.5 * (M + M.conj().T)       # (Nmu x Nmu) Hermitian
        self.nmu = X.shape[1]

        # M = Z Z^H via HERMITIAN eigendecomposition (real eigenvalues, complex
        # eigenvectors).  Same G=0-free / G=0-included split as the real class:
        # the PROPAGATION factor comes from M_prop when supplied; the local energy
        # and the v0 one-body mod always use self.M.
        M_for_prop = self.M if M_prop is None else numpy.ascontiguousarray(M_prop)
        M_for_prop = 0.5 * (M_for_prop + M_for_prop.conj().T)
        # If M is REAL (Hermitian + real => symmetric), diagonalize the REAL
        # matrix: the complex eigh returns eigenvectors with arbitrary per-column
        # phases, which would make zeta spuriously complex and force the B field
        # channel to be carried even though it is physically null.  Using the
        # real solver yields a real zeta and HALVES nfields (see below).  This is
        # the common metal case -- M is the AO-level ISDF metric, whose realness
        # is an AO-Gaussian identity independent of the (complex) orbitals.
        im_M = float(numpy.abs(M_for_prop.imag).max()) if M_for_prop.size else 0.0
        sc_M = max(1.0, float(numpy.abs(M_for_prop.real).max()) if M_for_prop.size else 1.0)
        if im_M <= 1e-12 * sc_M:
            w, U = _eigh_host_or_gpu(numpy.ascontiguousarray(M_for_prop.real),
                                     verbose=verbose)
            U = U.astype(numpy.complex128)
        else:
            w, U = _eigh_host_or_gpu(M_for_prop, verbose=verbose)
        wmax = max(1.0, float(w.max()))
        if w.min() < -psd_tol * wmax:
            print(f"# WARNING: complex THC central matrix M not PSD: min eig = {w.min():.4e}")
        keep = w > psd_tol * wmax
        self.zeta = (U[:, keep] * numpy.sqrt(numpy.clip(w[keep], 0.0, None))[None, :]).copy()
        self.nchol = self.zeta.shape[1]       # Ngamma
        self.chunked = False

        # Single REAL field matrix shared by VHS and vbias (see class docstring).
        #
        # OPTIMIZATION: the B channel couples to Im(zeta), so when the central
        # matrix M is REAL (Hermitian AND real => symmetric) the eigendecomposition
        # gives a real zeta and the B channel is IDENTICALLY NULL.  That is the
        # common case for the metal path: M is the AO-level ISDF metric, whose
        # realness is an AO-Gaussian identity independent of the (complex)
        # orbitals -- qcpbc's cthc exports measure |Im M| ~ 4e-16 while |Im X| ~ 2.
        # Dropping the null half HALVES nfields and therefore the dominant
        # propagation/force-bias GEMM cost, with no approximation.  Everything
        # downstream reads nfields from Zc.shape[1], so nothing else changes.
        im_scale = float(numpy.abs(self.zeta.imag).max()) if self.zeta.size else 0.0
        re_scale = max(1.0, float(numpy.abs(self.zeta.real).max()) if self.zeta.size else 1.0)
        self.real_fields = im_scale <= 1e-12 * re_scale
        if self.real_fields:
            self.Zc = numpy.ascontiguousarray(self.zeta.real)      # (Nmu x Ngamma)
        else:
            self.Zc = numpy.ascontiguousarray(
                numpy.concatenate([self.zeta.real, self.zeta.imag], axis=1)
            )                                                      # (Nmu x 2*Ngamma)
        self.nfields = self.Zc.shape[1]

        # One-body reordering (Motta17 Eqn 17), v0_ij = 0.5 sum_{k,gamma} L_ik^g conj(L_jk^g).
        # Substituting the THC L gives, with S = X^H X (the collocation overlap),
        #     v0 = 0.5 * X (M . S) X^H          (Hadamard),
        # the exact complex analog of the real 0.5 X (M . X^T X) X^T.  v0 is
        # Hermitian since M and S are.
        # PHYSICAL operator matrix (coefficient-basis convention c' = h c):
        # v0_pq = 0.5 sum_r (pr|rq)_chem = 0.5 [X* (M .* S^T) X^T]_pq with
        # S = X^H X.  The X (...) X^H form is the TRANSPOSE -- correct only for
        # real X (where they coincide); for complex X it breaks gauge covariance
        # of the one-body (h1 must transform as U^H h0 U).  See frag_gates.py.
        S = self.X.conj().T @ self.X                  # (Nmu x Nmu) Hermitian
        W = self.M * S.T                               # Hadamard, (M .* S^T) Hermitian
        v0 = 0.5 * (self.X.conj() @ (W @ self.X.T))    # (nbasis x nbasis) Hermitian
        h1e_mod = numpy.zeros(self.H1.shape, dtype=numpy.complex128)
        h1e_mod[0] = self.H1[0] - v0
        h1e_mod[1] = self.H1[1] - v0
        self.h1e_mod = xp.array(h1e_mod)

        # Optional reduced-precision PROPAGATION factors (IPIE_THC_MIXED=1):
        # complex64 X and float32 Zc, used by the VHS / force-bias / one-body
        # kernels only.  Energies and the fragment estimator always use fp64.
        self.X32 = None
        self.Zc32 = None
        if _env_flag("IPIE_THC_MIXED"):
            self.X32 = self.X.astype(numpy.complex64)
            self.Zc32 = self.Zc.astype(numpy.float32)
            if not os.environ.get("CUPY_TF32"):
                print(
                    "# GenericComplexTHC: IPIE_THC_MIXED=1 (complex64 propagation factors). "
                    "For TF32 tensor cores on GPU also export CUPY_TF32=1."
                )

        if verbose:
            memX = self.X.nbytes / 1024.0**3
            memM = self.M.nbytes / 1024.0**3
            print("# GenericComplexTHC:")
            print(f"#   nbasis={self.nbasis}  Nmu={self.nmu}  Ngamma={self.nchol}  "
                  f"nfields={self.nfields}"
                  + ("  (REAL zeta -> B channel null, fields HALVED)"
                     if self.real_fields else "  (=2*Ngamma, complex zeta)"))
            print(f"#   X {self.X.shape} ({memX:.3f} GB), M {self.M.shape} ({memM:.3f} GB)")
            print(f"#   M eig range [{w.min():.3e}, {w.max():.3e}] (PSD)")

    # ---- factored two-body building blocks (no L_pq^gamma materialized) ----

    def rho_moment(self, Gmat):
        r"""PHYSICAL density bubble rho_mu = sum_ab conj(X[a,mu]) X[b,mu] G_ab
        (= <rho_hat_mu> for G the mixed 1RDM).  The X * conj(X) pairing is wrong
        for complex X (coincides only on real bases).  Returns (Nmu,) complex."""
        GX = Gmat @ self.X                        # (nbasis x Nmu)
        return numpy.sum(self.X.conj() * GX, axis=0)

    def rho_moment_halfrot(self, Xoccc, Ghalf):
        r"""Same physical rho from half-rotated factors:
        rho_mu = sum_i Xoccc[i,mu] (Ghalf X)[i,mu], Xoccc = psi_occ^H conj(X)."""
        B = Ghalf @ self.X                        # (nocc x Nmu)
        return numpy.sum(Xoccc * B, axis=0)

    def vbias_from_rho(self, rho):
        r"""Force bias in the stacked (A,B) field space: vbias = Zc^T rho.  Returns (2*Ngamma,)."""
        return self.Zc.T @ rho

    def half_rotate(self, psi_occ):
        r"""Half-rotated collocation Xocc = psi_occ^H X  (nocc x Nmu).

        NOTE the CONJUGATE transpose.  ipie builds G = conj(psi0) @ Ghalf
        (greens_function_single_det.py), so
            rho_mu = sum_pq X[p,mu] conj(X[q,mu]) G_pq
                   = sum_i (psi0^H X)[i,mu] (Ghalf conj(X))[i,mu].
        The REAL THC class uses psi_occ.T (valid only because its trial is real);
        with a complex trial the conjugate is required."""
        return psi_occ.conj().T @ self.X


class GenericRealTHCUhf(GenericRealTHC):
    r"""Real THC Hamiltonian with PER-SPIN orbital bases and a SHARED auxiliary index.

    UHF-LNO design (theory/lno_uhf.md section 6): the alpha and beta cluster bases
    differ (n_a_orb vs n_b_orb orbitals) but the ISDF interpolation points mu and
    the central Coulomb metric M are spin-independent (they come from the AO fit).
    Only the collocation matrices split:

        X_a : (n_a_orb x Nmu),   X_b : (n_b_orb x Nmu),   M : (Nmu x Nmu)  shared

    so the spin-sigma block of the two-body operator is
        (pq|rs)^{ss'} = sum_{mu nu} X_s[p,mu] X_s[q,mu] M[mu,nu] X_s'[r,nu] X_s'[s,nu]
    and ONE Hubbard-Stratonovich field x_mu (nfields = Ngamma from the shared
    zeta with M = zeta zeta^T) couples the TOTAL charge density: the sampled
    field builds per-spin potentials

        VHS^s[w] = isqrt_dt * X_s diag(zeta . x[w]) X_s^T        (same x[w] for both spins).

    Subclasses GenericRealTHC so all THC-aware isinstance checks and plum
    dispatch keep routing; every single-X code path is poisoned (self.X = None)
    and replaced by more-specific overloads registered for this type (see
    ipie/lno_thc_uhf.py, ipie/propagation/phaseless_base.py,
    ipie/propagation/phaseless_generic.py, ipie/estimators/energy.py,
    ipie/trial_wavefunction/single_det.py).

    Parameters
    ----------
    h1e_a : ndarray (n_a_orb, n_a_orb)
        Alpha one-body integrals in the alpha cluster basis.
    h1e_b : ndarray (n_b_orb, n_b_orb)
        Beta one-body integrals in the beta cluster basis.
    Xa, Xb : ndarray (n_s_orb, Nmu)
        Per-spin real collocations, SHARED Nmu.
    M : ndarray (Nmu, Nmu)
        Shared real symmetric PSD central matrix.
    ecore : float
        Constant (nuclear + frozen-core) energy.
    M_prop : ndarray, optional
        G=0-free propagation metric (same convention as GenericRealTHC).
    """

    # NOTE: deliberately does NOT call GenericRealTHC.__init__/GenericBase.__init__
    # (both assume one shared basis for the two spins).
    def __init__(self, h1e_a, h1e_b, Xa, Xb, M, ecore=0.0, psd_tol=1e-10, verbose=False,
                 M_prop=None):
        self.verbose = verbose
        self.ecore = ecore

        h1e_a = numpy.ascontiguousarray(numpy.asarray(h1e_a, dtype=numpy.float64))
        h1e_b = numpy.ascontiguousarray(numpy.asarray(h1e_b, dtype=numpy.float64))
        assert h1e_a.ndim == 2 and h1e_a.shape[0] == h1e_a.shape[1]
        assert h1e_b.ndim == 2 and h1e_b.shape[0] == h1e_b.shape[1]
        # H1 as a LIST so the two spin blocks may have different sizes; consumers
        # index hamiltonian.H1[0] / hamiltonian.H1[1] exactly as for the ndarray.
        self.H1 = [h1e_a, h1e_b]
        self.nbasis_a = h1e_a.shape[-1]
        self.nbasis_b = h1e_b.shape[-1]
        # Generic bookkeeping only.  Per-spin code MUST use nbasis_a/nbasis_b;
        # nothing on the THC-UHF path consumes this attribute.
        self.nbasis = max(self.nbasis_a, self.nbasis_b)
        self.nchol = None

        Xa = numpy.asarray(Xa, dtype=numpy.float64)
        Xb = numpy.asarray(Xb, dtype=numpy.float64)
        M = numpy.asarray(M, dtype=numpy.float64)
        assert Xa.shape[0] == self.nbasis_a, f"Xa rows {Xa.shape[0]} != nbasis_a {self.nbasis_a}"
        assert Xb.shape[0] == self.nbasis_b, f"Xb rows {Xb.shape[0]} != nbasis_b {self.nbasis_b}"
        assert Xa.shape[1] == Xb.shape[1], "alpha/beta collocations must share Nmu"
        assert M.shape[0] == M.shape[1] == Xa.shape[1]

        self.Xa = Xa
        self.Xb = Xb
        self.X = None                    # poison the single-X (RHF) code paths
        self.M = 0.5 * (M + M.T)
        self.nmu = Xa.shape[1]

        # Shared auxiliary-field factor (one field set couples the total density).
        M_for_prop = self.M if M_prop is None else numpy.ascontiguousarray(M_prop)
        w, U = _eigh_host_or_gpu(M_for_prop, verbose=verbose)
        wmax = max(1.0, float(w.max()))
        if w.min() < -psd_tol * wmax:
            print(f"# WARNING: THC central matrix M not PSD: min eig = {w.min():.4e}")
        keep = w > psd_tol * wmax
        self.zeta = (U[:, keep] * numpy.sqrt(numpy.clip(w[keep], 0.0, None))[None, :]).copy()
        self.nchol = self.zeta.shape[1]
        self.nfields = self.nchol
        self.chunked = False

        # One-body re-ordering correction, PER SPIN with the spin's own X but
        # exactly the RHF v0 formula:  v0^s = 0.5 [ X_s (M .* (X_s^T X_s)) X_s^T ].
        h1e_mod = []
        for X_s, h_s in ((Xa, h1e_a), (Xb, h1e_b)):
            S = X_s.T @ X_s                          # (Nmu x Nmu)
            W = self.M * S                           # Hadamard
            v0 = 0.5 * (X_s @ (W @ X_s.T))           # (n_s_orb x n_s_orb)
            h1e_mod.append(xp.array(h_s - v0))
        self.h1e_mod = h1e_mod                       # list [alpha, beta]

        # Mixed-precision propagation factors are not supported on the UHF path.
        self.X32 = None
        self.zeta32 = None

        if verbose:
            print("# GenericRealTHCUhf:")
            print(f"#   nbasis_a={self.nbasis_a}  nbasis_b={self.nbasis_b}  "
                  f"Nmu={self.nmu}  nfields(Ngamma)={self.nfields}")
            print(f"#   Xa {self.Xa.shape}, Xb {self.Xb.shape}, M {self.M.shape}")
            print(f"#   M eig range [{w.min():.3e}, {w.max():.3e}] (PSD)")

    # ---- per-spin factored building blocks (shared mu index) ----

    def X_spin(self, spin):
        """Collocation of the requested spin (0 = alpha, 1 = beta)."""
        return self.Xa if spin == 0 else self.Xb

    def rho_moment_spin(self, Gmat, spin):
        r"""rho_mu = sum_pq X_s[p,mu] G_pq X_s[q,mu] with the spin's own collocation."""
        X = self.X_spin(spin)
        GX = Gmat @ X
        return numpy.sum(X * GX, axis=0)

    def half_rotate_spin(self, psi_occ, spin):
        r"""Half-rotated collocation Xocc^s = psi_occ^T X_s  (nocc_s x Nmu)."""
        return psi_occ.T @ self.X_spin(spin)

    def half_rotate(self, psi_occ):
        raise RuntimeError(
            "GenericRealTHCUhf has per-spin collocations; use half_rotate_spin(psi_occ, spin)."
        )

    def rho_moment(self, Gmat):
        raise RuntimeError(
            "GenericRealTHCUhf has per-spin collocations; use rho_moment_spin(Gmat, spin)."
        )
