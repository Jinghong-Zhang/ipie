"""JAX kernels for the discrete single-site Hubbard propagator."""

from functools import partial

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import jax.scipy.linalg as jsp_linalg


def hashable_mpi_comm(comm):
    """Return a hashable mpi4jax communicator wrapper for use in jitted calls."""
    if comm is None:
        return None
    try:
        from mpi4jax._src.utils import wrap_as_hashable
    except ImportError as exc:
        raise ImportError("mpi4jax is required for differentiable MPI Hubbard AD blocks.") from exc
    return wrap_as_hashable(comm)


def as_jax_array(array, dtype=None, force_host=False):
    """Convert NumPy/CuPy/JAX arrays to JAX arrays after enabling x64."""
    if not jax.config.jax_enable_x64:
        raise RuntimeError("The Hubbard JAX backend requires jax_enable_x64=True.")
    if force_host and hasattr(array, "get"):
        array = array.get()
    return jnp.array(array, dtype=dtype, copy=True)


def block_until_ready(tree):
    """Synchronize a JAX pytree and return it."""

    def _block(x):
        if hasattr(x, "block_until_ready"):
            return x.block_until_ready()
        return x

    return jax.tree_util.tree_map(_block, tree)


def _propagate_one_body(phi, bt2):
    return jnp.einsum("ij,wjk->wik", bt2, phi, optimize=True)


def _construct_one_body_propagator(h1e, dt):
    return jnp.stack(
        [
            jsp_linalg.expm(-0.5 * dt * h1e[0]),
            jsp_linalg.expm(-0.5 * dt * h1e[1]),
        ]
    )


def _inverse_overlap(psi0, phi):
    ovlp = jnp.einsum("mi,wmj->wij", jnp.conj(psi0), phi, optimize=True)
    return jnp.linalg.inv(ovlp)


def _calc_overlap_from_inverse(inva, invb, log_shift):
    sign_a, logdet_a = jnp.linalg.slogdet(inva)
    if invb.shape[1] > 0:
        sign_b, logdet_b = jnp.linalg.slogdet(invb)
    else:
        sign_b = jnp.ones_like(sign_a)
        logdet_b = jnp.zeros_like(logdet_a)
    det = sign_a * sign_b * jnp.exp(logdet_a + logdet_b - log_shift)
    return 1.0 / det


def _kinetic_importance_sampling(
    phia,
    phib,
    inva,
    invb,
    weight,
    ovlp,
    log_shift,
    psi0a,
    psi0b,
    expH1,
    rhf,
):
    phia = _propagate_one_body(phia, expH1[0])
    if phib.shape[2] > 0 and not rhf:
        phib = _propagate_one_body(phib, expH1[1])

    inva = _inverse_overlap(psi0a, phia)
    if phib.shape[2] > 0:
        invb = _inverse_overlap(psi0b, phib)

    ovlp_new = _calc_overlap_from_inverse(inva, invb, log_shift)
    ratio = ovlp_new / ovlp
    phase = jnp.angle(ratio)
    weight_factor = jnp.where(jnp.abs(phase) < 0.5 * jnp.pi, ratio.real, 0.0)
    weight = weight * weight_factor
    ovlp = jnp.where(weight_factor > 0.0, ovlp_new, ovlp)
    return phia, phib, inva, invb, weight, ovlp


def _batched_sherman_morrison(ainv, u, vt):
    au = jnp.einsum("wij,j->wi", ainv, u, optimize=True)
    vta = jnp.einsum("wi,wij->wj", vt, ainv, optimize=True)
    denom = 1.0 + jnp.einsum("wi,i->w", vta, u, optimize=True)
    update = jnp.einsum("wi,wj->wij", au, vta, optimize=True) / denom[:, None, None]
    return ainv - update


def _site_greens_diagonal(inv_ovlp, phi, psi0_site, site):
    q = jnp.einsum("wij,wi->wj", inv_ovlp, phi[:, site, :], optimize=True)
    return jnp.einsum("j,wj->w", jnp.conj(psi0_site), q, optimize=True)


def _greens_from_inverse(psi0a, psi0b, phia, phib, inva, invb, rhf=False):
    ga = jnp.einsum("mi,wji,wnj->wmn", jnp.conj(psi0a), inva, phia, optimize=True)
    if phib.shape[2] > 0 and not rhf:
        gb = jnp.einsum("mi,wji,wnj->wmn", jnp.conj(psi0b), invb, phib, optimize=True)
    elif phib.shape[2] > 0 and rhf:
        gb = ga
    else:
        gb = jnp.zeros_like(ga)
    return ga, gb


def _hubbard_local_energy(h1e, U, ecore, ga, gb):
    e1b = (
        jnp.einsum("ij,wji->w", h1e[0], ga, optimize=True)
        + jnp.einsum("ij,wji->w", h1e[1], gb, optimize=True)
        + ecore
    )
    nia = jnp.diagonal(ga, axis1=1, axis2=2)
    nib = jnp.diagonal(gb, axis1=1, axis2=2)
    e2b = U * jnp.sum(nia * nib, axis=1)
    return e1b + e2b


def _density_from_orbitals(psi, nbasis):
    if psi.shape[1] == 0:
        return jnp.zeros(nbasis, dtype=jnp.float64)
    dm = psi @ jnp.conj(psi).T
    return jnp.real(jnp.diagonal(dm))


def _gauge_fix_eigenvectors(vecs):
    max_abs = jnp.argmax(jnp.abs(vecs), axis=0)
    vals = jnp.take_along_axis(vecs, max_abs[None, :], axis=0)[0]
    phase = jnp.where(jnp.abs(vals) > 0.0, vals / jnp.abs(vals), 1.0 + 0.0j)
    return vecs / phase[None, :]


def hubbard_uhf_trial_from_densities(
    h1e,
    U,
    nup,
    ndown,
    niup0,
    nidown0,
    n_scf=50,
    mixing=0.5,
):
    """Fixed-iteration JAX port of the legacy Hubbard UHF mean-field equations.

    The eigenvector derivative is gauge-fixed but remains ill-defined for
    degenerate mean-field spectra.
    """
    nbasis = h1e.shape[-1]
    empty_a = jnp.zeros((nbasis, nup), dtype=h1e.dtype)
    empty_b = jnp.zeros((nbasis, ndown), dtype=h1e.dtype)

    def _scf_step(carry, _):
        niup, nidown, _, _ = carry
        hmf_up = h1e[0] + jnp.diag(U * nidown)
        hmf_down = h1e[1] + jnp.diag(U * niup)
        _, ev_up = jnp.linalg.eigh(hmf_up)
        _, ev_down = jnp.linalg.eigh(hmf_down)
        ev_up = _gauge_fix_eigenvectors(ev_up)
        ev_down = _gauge_fix_eigenvectors(ev_down)
        psi0a = ev_up[:, :nup]
        psi0b = ev_down[:, :ndown]
        niup_new = _density_from_orbitals(psi0a, nbasis)
        nidown_new = _density_from_orbitals(psi0b, nbasis)
        niup_next = (1.0 - mixing) * niup_new + mixing * niup
        nidown_next = (1.0 - mixing) * nidown_new + mixing * nidown
        return (niup_next, nidown_next, psi0a, psi0b), None

    (_, _, psi0a, psi0b), _ = jax.lax.scan(
        _scf_step, (niup0, nidown0, empty_a, empty_b), None, length=n_scf
    )
    return psi0a, psi0b


def _hubbard_single_site_two_body_impl(
    phia,
    phib,
    inva,
    invb,
    weight,
    ovlp,
    psi0a,
    psi0b,
    delta,
    aux_wfac,
    random_fields,
    rhf=False,
):
    def _site_update(carry, site_data):
        phia, phib, inva, invb, weight, ovlp = carry
        site, random_values, psi0a_site, psi0b_site = site_data

        gup = _site_greens_diagonal(inva, phia, psi0a_site, site)
        if phib.shape[2] > 0 and not rhf:
            gdown = _site_greens_diagonal(invb, phib, psi0b_site, site)
        elif phib.shape[2] > 0 and rhf:
            gdown = gup
        else:
            gdown = jnp.zeros_like(gup)

        r1 = (1.0 + delta[0, 0] * gup) * (1.0 + delta[0, 1] * gdown)
        r2 = (1.0 + delta[1, 0] * gup) * (1.0 + delta[1, 1] * gdown)
        probs = 0.5 * jnp.stack([r1, r2], axis=1) * aux_wfac[None, :]
        phaseless_ratio = jnp.maximum(probs.real, 0.0)
        norm = jnp.sum(phaseless_ratio, axis=1)
        live = (norm > 0.0) & (jnp.abs(weight) > 0.0)

        norm_safe = jnp.where(live, norm, 1.0)
        p0 = jnp.where(live, phaseless_ratio[:, 0] / norm_safe, 1.0)
        xi = (random_values >= p0).astype(jnp.int32)
        selected = jnp.take_along_axis(probs, xi[:, None], axis=1)[:, 0]

        weight = weight * jnp.where(live, norm, 0.0)
        ovlp = jnp.where(live, 2.0 * ovlp * selected, ovlp)

        vtup = phia[:, site, :] * delta[xi, 0][:, None]
        vtup = jnp.where(live[:, None], vtup, 0.0)
        phia = phia.at[:, site, :].add(vtup)
        inva = _batched_sherman_morrison(inva, jnp.conj(psi0a_site), vtup)

        if phib.shape[2] > 0 and not rhf:
            vtdown = phib[:, site, :] * delta[xi, 1][:, None]
            vtdown = jnp.where(live[:, None], vtdown, 0.0)
            phib = phib.at[:, site, :].add(vtdown)
            invb = _batched_sherman_morrison(invb, jnp.conj(psi0b_site), vtdown)

        return (phia, phib, inva, invb, weight, ovlp), None

    sites = jnp.arange(phia.shape[1], dtype=jnp.int32)
    carry = (phia, phib, inva, invb, weight, ovlp)
    carry, _ = jax.lax.scan(_site_update, carry, (sites, random_fields, psi0a, psi0b))
    return carry


def _initial_inverse_and_overlap(psi0a, psi0b, phia, phib, log_shift):
    inva = _inverse_overlap(psi0a, phia)
    if phib.shape[2] > 0:
        invb = _inverse_overlap(psi0b, phib)
    else:
        invb = jnp.zeros((phia.shape[0], 0, 0), dtype=phia.dtype)
    ovlp = _calc_overlap_from_inverse(inva, invb, log_shift)
    return inva, invb, ovlp


def _trial_orbitals_for_response(
    h1e,
    U,
    psi0a,
    psi0b,
    nup,
    ndown,
    trial_response,
    uhf_n_scf,
    uhf_mixing,
    uhf_ueff,
):
    if trial_response == "jax_uhf":
        nbasis = h1e.shape[-1]
        h1e_hf = 0.5 * (h1e + jnp.swapaxes(jnp.conj(h1e), -1, -2))
        niup0 = _density_from_orbitals(psi0a, nbasis)
        nidown0 = _density_from_orbitals(psi0b, nbasis)
        return hubbard_uhf_trial_from_densities(
            h1e_hf,
            uhf_ueff,
            nup,
            ndown,
            niup0,
            nidown0,
            n_scf=uhf_n_scf,
            mixing=uhf_mixing,
        )
    return psi0a, psi0b


def _apply_one_body_coupling(h1e, coupling, coupling_mode):
    if coupling_mode == "full":
        return h1e + coupling.astype(h1e.dtype)
    if coupling_mode == "diagonal":
        diag = coupling.astype(h1e.dtype)
        eye = jnp.eye(h1e.shape[-1], dtype=h1e.dtype)
        return h1e + diag[:, :, None] * eye[None, :, :]
    raise ValueError(f"Unknown coupling_mode={coupling_mode!r}.")


def _hubbard_response_objective_impl(
    coupling,
    phia,
    phib,
    weight,
    log_shift,
    psi0a,
    psi0b,
    h1e,
    U,
    ecore,
    delta,
    aux_wfac,
    random_fields,
    dt,
    nup,
    ndown,
    rhf=False,
    trial_response=False,
    uhf_n_scf=50,
    uhf_mixing=0.5,
    uhf_ueff=None,
):
    h1e_lambda = _apply_one_body_coupling(h1e, coupling, "full")
    if uhf_ueff is None:
        uhf_ueff = U.real
    trial_mode = "jax_uhf" if trial_response == "jax_uhf" else False
    psi0a, psi0b = _trial_orbitals_for_response(
        h1e_lambda,
        U,
        psi0a,
        psi0b,
        nup,
        ndown,
        trial_mode,
        uhf_n_scf,
        uhf_mixing,
        uhf_ueff,
    )
    expH1 = _construct_one_body_propagator(h1e_lambda, dt)
    inva, invb, ovlp = _initial_inverse_and_overlap(psi0a, psi0b, phia, phib, log_shift)

    def _step(carry, fields):
        phia, phib, inva, invb, weight, ovlp = carry
        phia, phib, inva, invb, weight, ovlp = _kinetic_importance_sampling(
            phia, phib, inva, invb, weight, ovlp, log_shift, psi0a, psi0b, expH1, rhf
        )
        phia, phib, inva, invb, weight, ovlp = _hubbard_single_site_two_body_impl(
            phia,
            phib,
            inva,
            invb,
            weight,
            ovlp,
            psi0a,
            psi0b,
            delta,
            aux_wfac,
            fields,
            rhf=rhf,
        )
        phia, phib, inva, invb, weight, ovlp = _kinetic_importance_sampling(
            phia, phib, inva, invb, weight, ovlp, log_shift, psi0a, psi0b, expH1, rhf
        )
        return (phia, phib, inva, invb, weight, ovlp), None

    carry = (phia, phib, jnp.asarray(inva), jnp.asarray(invb), weight, ovlp)
    carry, _ = jax.lax.scan(_step, carry, random_fields)
    phia, phib, inva, invb, weight, _ = carry
    ga, gb = _greens_from_inverse(psi0a, psi0b, phia, phib, inva, invb, rhf=rhf)
    energy = _hubbard_local_energy(h1e_lambda, U, ecore, ga, gb)
    enumer = jnp.sum(weight * energy.real)
    edenom = jnp.sum(weight)
    return jnp.stack([enumer, edenom])


def _reorthogonalize_spin(phi):
    q, r = jnp.linalg.qr(phi, mode="reduced")
    diag = jnp.diagonal(r, axis1=1, axis2=2)
    phase = jnp.where(jnp.abs(diag) > 0.0, diag / jnp.abs(diag), 1.0 + 0.0j)
    q = q * phase[:, None, :]
    log_det = jnp.sum(jnp.log(jnp.abs(diag)), axis=1)
    return q, log_det


def _reorthogonalize_walkers(phia, phib, ovlp):
    phia, log_det = _reorthogonalize_spin(phia)
    if phib.shape[2] > 0:
        phib, log_det_b = _reorthogonalize_spin(phib)
        log_det = log_det + log_det_b
    detR = jnp.exp(log_det)
    ovlp = ovlp / detR
    return phia, phib, ovlp


def _local_stochastic_reconfiguration(
    phia,
    phib,
    inva,
    invb,
    weight,
    ovlp,
    zeta,
):
    nwalkers = weight.shape[0]
    abs_weight = jnp.abs(weight)
    total_weight = jnp.sum(abs_weight)

    def _sr(_):
        scaled_weight = abs_weight / total_weight * nwalkers
        cumulative = jnp.cumsum(scaled_weight)
        positions = jnp.arange(nwalkers, dtype=scaled_weight.dtype) + zeta
        indices = jnp.searchsorted(cumulative, positions, side="left")
        indices = jnp.minimum(indices, nwalkers - 1)
        indices = jax.lax.stop_gradient(indices)
        new_weight = jnp.ones_like(weight)
        new_weight = jax.lax.stop_gradient(new_weight)
        return (
            phia[indices],
            phib[indices],
            inva[indices],
            invb[indices],
            new_weight,
            ovlp[indices],
        )

    return jax.lax.cond(
        total_weight > 0.0,
        _sr,
        lambda _: (phia, phib, inva, invb, weight, ovlp),
        operand=None,
    )


def _mpi_allgather_via_allreduce(x, mpi_comm, mpi_rank, mpi_size):
    if mpi_comm is None or int(mpi_size) == 1:
        return x[None, ...]
    try:
        import mpi4jax
        from mpi4py import MPI
    except ImportError as exc:
        raise ImportError("mpi4jax and mpi4py are required for differentiable MPI.") from exc

    gathered = jnp.zeros((int(mpi_size),) + x.shape, dtype=x.dtype)
    gathered = gathered.at[int(mpi_rank)].set(x)
    if jnp.issubdtype(gathered.dtype, jnp.complexfloating):
        gathered_real = mpi4jax.allreduce(gathered.real, MPI.SUM, comm=mpi_comm)
        gathered_imag = mpi4jax.allreduce(gathered.imag, MPI.SUM, comm=mpi_comm)
        return gathered_real + 1.0j * gathered_imag
    return mpi4jax.allreduce(gathered, MPI.SUM, comm=mpi_comm)


@partial(jax.custom_vjp, nondiff_argnums=(1, 2, 3))
def _mpi_allgather_ad(x, mpi_comm, mpi_rank, mpi_size):
    """Allgather with an explicit cross-rank reverse rule.

    mpi4jax 0.9 provides AD rules for ``allreduce``, but its transpose is
    local in the SPMD sense.  Global population control needs gradients from a
    copied walker to flow back to the rank that supplied the parent walker, so
    the backward pass explicitly sums the cotangent buffer and returns this
    rank's slice.
    """
    return _mpi_allgather_via_allreduce(x, mpi_comm, mpi_rank, mpi_size)


def _mpi_allgather_ad_fwd(x, mpi_comm, mpi_rank, mpi_size):
    gathered = _mpi_allgather_via_allreduce(x, mpi_comm, mpi_rank, mpi_size)
    return gathered, None


def _mpi_allgather_ad_bwd(mpi_comm, mpi_rank, mpi_size, _res, gathered_cotangent):
    if mpi_comm is None or int(mpi_size) == 1:
        return (gathered_cotangent[0],)
    try:
        import mpi4jax
        from mpi4py import MPI
    except ImportError as exc:
        raise ImportError("mpi4jax and mpi4py are required for differentiable MPI.") from exc

    if jnp.issubdtype(gathered_cotangent.dtype, jnp.complexfloating):
        summed_real = mpi4jax.allreduce(gathered_cotangent.real, MPI.SUM, comm=mpi_comm)
        summed_imag = mpi4jax.allreduce(gathered_cotangent.imag, MPI.SUM, comm=mpi_comm)
        summed_cotangent = summed_real + 1.0j * summed_imag
    else:
        summed_cotangent = mpi4jax.allreduce(gathered_cotangent, MPI.SUM, comm=mpi_comm)
    return (summed_cotangent[int(mpi_rank)],)


_mpi_allgather_ad.defvjp(_mpi_allgather_ad_fwd, _mpi_allgather_ad_bwd)


def _global_stochastic_reconfiguration_mpi4jax(
    phia,
    phib,
    inva,
    invb,
    weight,
    ovlp,
    zeta,
    mpi_comm,
    mpi_rank,
    mpi_size,
):
    if mpi_comm is None or int(mpi_size) == 1:
        return _local_stochastic_reconfiguration(phia, phib, inva, invb, weight, ovlp, zeta)

    nwalkers = weight.shape[0]
    ntot_walkers = int(mpi_size) * nwalkers

    phia_global = _mpi_allgather_ad(phia, mpi_comm, mpi_rank, mpi_size).reshape(
        (ntot_walkers,) + phia.shape[1:]
    )
    phib_global = _mpi_allgather_ad(phib, mpi_comm, mpi_rank, mpi_size).reshape(
        (ntot_walkers,) + phib.shape[1:]
    )
    inva_global = _mpi_allgather_ad(inva, mpi_comm, mpi_rank, mpi_size).reshape(
        (ntot_walkers,) + inva.shape[1:]
    )
    invb_global = _mpi_allgather_ad(invb, mpi_comm, mpi_rank, mpi_size).reshape(
        (ntot_walkers,) + invb.shape[1:]
    )
    weight_global = _mpi_allgather_ad(weight, mpi_comm, mpi_rank, mpi_size).reshape((ntot_walkers,))
    ovlp_global = _mpi_allgather_ad(ovlp, mpi_comm, mpi_rank, mpi_size).reshape((ntot_walkers,))

    abs_weight = jnp.abs(weight_global)
    total_weight = jnp.sum(abs_weight)

    def _sr(_):
        scaled_weight = abs_weight / total_weight * ntot_walkers
        cumulative = jnp.cumsum(scaled_weight)
        positions = jnp.arange(ntot_walkers, dtype=scaled_weight.dtype) + zeta
        global_indices = jnp.searchsorted(cumulative, positions, side="left")
        global_indices = jnp.minimum(global_indices, ntot_walkers - 1)
        global_indices = jax.lax.stop_gradient(global_indices)
        local_slots = int(mpi_rank) * nwalkers + jnp.arange(nwalkers, dtype=jnp.int32)
        indices = global_indices[local_slots]
        new_weight = jax.lax.stop_gradient(jnp.ones_like(weight))
        return (
            phia_global[indices],
            phib_global[indices],
            inva_global[indices],
            invb_global[indices],
            new_weight,
            ovlp_global[indices],
        )

    return jax.lax.cond(
        total_weight > 0.0,
        _sr,
        lambda _: (phia, phib, inva, invb, weight, ovlp),
        operand=None,
    )


def _hubbard_ad_block_objective_impl(
    coupling,
    phia,
    phib,
    weight,
    log_shift,
    psi0a,
    psi0b,
    h1e,
    U,
    ecore,
    delta,
    aux_wfac,
    random_fields,
    pop_control_randoms,
    dt,
    nup,
    ndown,
    rhf=False,
    trial_response=False,
    uhf_n_scf=50,
    uhf_mixing=0.5,
    uhf_ueff=None,
    stabilize_freq=5,
    pop_control_freq=5,
    measure_freq=25,
    local_pop_control=True,
    global_pop_control=False,
    mpi_comm=None,
    mpi_rank=0,
    mpi_size=1,
    checkpoint_steps=True,
    coupling_mode="full",
):
    h1e_lambda = _apply_one_body_coupling(h1e, coupling, coupling_mode)
    if uhf_ueff is None:
        uhf_ueff = U.real
    trial_mode = "jax_uhf" if trial_response == "jax_uhf" else False
    psi0a, psi0b = _trial_orbitals_for_response(
        h1e_lambda,
        U,
        psi0a,
        psi0b,
        nup,
        ndown,
        trial_mode,
        uhf_n_scf,
        uhf_mixing,
        uhf_ueff,
    )
    expH1 = _construct_one_body_propagator(h1e_lambda, dt)
    inva, invb, ovlp = _initial_inverse_and_overlap(psi0a, psi0b, phia, phib, log_shift)

    def _identity_state(state):
        return state

    def _stabilize_state(state):
        phia, phib, inva, invb, weight, ovlp, enumer, edenom = state
        phia, phib, ovlp = _reorthogonalize_walkers(phia, phib, ovlp)
        return phia, phib, inva, invb, weight, ovlp, enumer, edenom

    def _measure_state(state):
        phia, phib, inva, invb, weight, ovlp, enumer, edenom = state
        ga, gb = _greens_from_inverse(psi0a, psi0b, phia, phib, inva, invb, rhf=rhf)
        energy = _hubbard_local_energy(h1e_lambda, U, ecore, ga, gb)
        return (
            phia,
            phib,
            inva,
            invb,
            weight,
            ovlp,
            enumer + jnp.sum(weight * energy.real),
            edenom + jnp.sum(weight),
        )

    def _pop_control_state(state, zeta):
        phia, phib, inva, invb, weight, ovlp, enumer, edenom = state
        phia, phib, inva, invb, weight, ovlp = _local_stochastic_reconfiguration(
            phia, phib, inva, invb, weight, ovlp, zeta
        )
        return phia, phib, inva, invb, weight, ovlp, enumer, edenom

    def _global_pop_control_state(state, zeta):
        phia, phib, inva, invb, weight, ovlp, enumer, edenom = state
        phia, phib, inva, invb, weight, ovlp = _global_stochastic_reconfiguration_mpi4jax(
            phia,
            phib,
            inva,
            invb,
            weight,
            ovlp,
            zeta,
            mpi_comm,
            mpi_rank,
            mpi_size,
        )
        return phia, phib, inva, invb, weight, ovlp, enumer, edenom

    def _step(state, data):
        step, fields, zeta = data
        stabilize_now = (step % stabilize_freq) == (stabilize_freq - 1)
        state = jax.lax.cond(stabilize_now, _stabilize_state, _identity_state, state)

        phia, phib, inva, invb, weight, ovlp, enumer, edenom = state
        phia, phib, inva, invb, weight, ovlp = _kinetic_importance_sampling(
            phia, phib, inva, invb, weight, ovlp, log_shift, psi0a, psi0b, expH1, rhf
        )
        phia, phib, inva, invb, weight, ovlp = _hubbard_single_site_two_body_impl(
            phia,
            phib,
            inva,
            invb,
            weight,
            ovlp,
            psi0a,
            psi0b,
            delta,
            aux_wfac,
            fields,
            rhf=rhf,
        )
        phia, phib, inva, invb, weight, ovlp = _kinetic_importance_sampling(
            phia, phib, inva, invb, weight, ovlp, log_shift, psi0a, psi0b, expH1, rhf
        )
        state = (phia, phib, inva, invb, weight, ovlp, enumer, edenom)

        measure_now = (step % measure_freq) == (measure_freq - 1)
        state = jax.lax.cond(measure_now, _measure_state, _identity_state, state)

        pop_now = (step % pop_control_freq) == (pop_control_freq - 1)
        if global_pop_control:
            state = jax.lax.cond(
                pop_now,
                lambda x: _global_pop_control_state(x, zeta),
                _identity_state,
                state,
            )
        elif local_pop_control:
            state = jax.lax.cond(
                pop_now,
                lambda x: _pop_control_state(x, zeta),
                _identity_state,
                state,
            )
        return state, None

    initial_state = (
        phia,
        phib,
        inva,
        invb,
        weight,
        ovlp,
        jnp.asarray(0.0, dtype=jnp.float64),
        jnp.asarray(0.0, dtype=jnp.float64),
    )
    steps = jnp.arange(random_fields.shape[0], dtype=jnp.int32)
    scan_step = jax.checkpoint(_step) if checkpoint_steps else _step
    final_state, _ = jax.lax.scan(
        scan_step,
        initial_state,
        (steps, random_fields, pop_control_randoms),
        _split_transpose=True,
    )
    phia, phib, inva, invb, weight, ovlp, enumer, edenom = final_state
    final_walkers = (phia, phib, inva, invb, weight, ovlp)
    return enumer, edenom, final_walkers


@partial(
    jax.jit,
    static_argnames=(
        "nup",
        "ndown",
        "rhf",
        "trial_response",
        "uhf_n_scf",
        "stabilize_freq",
        "pop_control_freq",
        "measure_freq",
        "local_pop_control",
        "global_pop_control",
        "mpi_comm",
        "mpi_rank",
        "mpi_size",
        "checkpoint_steps",
        "coupling_mode",
    ),
)
def hubbard_ad_block_raw(
    coupling,
    phia,
    phib,
    weight,
    log_shift,
    psi0a,
    psi0b,
    h1e,
    U,
    ecore,
    delta,
    aux_wfac,
    random_fields,
    pop_control_randoms,
    dt,
    nup,
    ndown,
    rhf=False,
    trial_response=False,
    uhf_n_scf=50,
    uhf_mixing=0.5,
    uhf_ueff=None,
    stabilize_freq=5,
    pop_control_freq=5,
    measure_freq=25,
    local_pop_control=True,
    global_pop_control=False,
    mpi_comm=None,
    mpi_rank=0,
    mpi_size=1,
    checkpoint_steps=True,
    coupling_mode="full",
):
    """Return AD-block raw response values and detached-ready final walker arrays.

    The differentiated objective is the weighted average over measurements
    inside one AD block.  Local stochastic reconfiguration is part of the
    forward trajectory, but its resampling indices and reset weights are treated
    as nondifferentiable control flow.  With ``global_pop_control=True``, the
    walker copy/gather uses mpi4jax collectives with a custom reverse rule so
    copied-walker cotangents flow back to the rank that supplied the parent.
    """

    def _objective(cpl):
        enumer, edenom, final_walkers = _hubbard_ad_block_objective_impl(
            cpl,
            phia,
            phib,
            weight,
            log_shift,
            psi0a,
            psi0b,
            h1e,
            U,
            ecore,
            delta,
            aux_wfac,
            random_fields,
            pop_control_randoms,
            dt,
            nup,
            ndown,
            rhf=rhf,
            trial_response=trial_response,
            uhf_n_scf=uhf_n_scf,
            uhf_mixing=uhf_mixing,
            uhf_ueff=uhf_ueff,
            stabilize_freq=stabilize_freq,
            pop_control_freq=pop_control_freq,
            measure_freq=measure_freq,
            local_pop_control=local_pop_control,
            global_pop_control=global_pop_control,
            mpi_comm=mpi_comm,
            mpi_rank=mpi_rank,
            mpi_size=mpi_size,
            checkpoint_steps=checkpoint_steps,
            coupling_mode=coupling_mode,
        )
        values = jnp.stack([enumer, edenom])
        return values, (values, final_walkers)

    jac, (values, final_walkers) = jax.jacrev(_objective, has_aux=True)(coupling)
    phia, phib, inva, invb, weight, ovlp = final_walkers
    return values[0], values[1], jac[0], jac[1], phia, phib, inva, invb, weight, ovlp


@partial(
    jax.jit,
    static_argnames=(
        "nup",
        "ndown",
        "rhf",
        "trial_response",
        "uhf_n_scf",
        "stabilize_freq",
        "pop_control_freq",
        "measure_freq",
        "local_pop_control",
        "global_pop_control",
        "mpi_comm",
        "mpi_rank",
        "mpi_size",
        "checkpoint_steps",
        "coupling_mode",
    ),
)
def hubbard_ad_block_value(
    coupling,
    phia,
    phib,
    weight,
    log_shift,
    psi0a,
    psi0b,
    h1e,
    U,
    ecore,
    delta,
    aux_wfac,
    random_fields,
    pop_control_randoms,
    dt,
    nup,
    ndown,
    rhf=False,
    trial_response=False,
    uhf_n_scf=50,
    uhf_mixing=0.5,
    uhf_ueff=None,
    stabilize_freq=5,
    pop_control_freq=5,
    measure_freq=25,
    local_pop_control=True,
    global_pop_control=False,
    mpi_comm=None,
    mpi_rank=0,
    mpi_size=1,
    checkpoint_steps=True,
    coupling_mode="full",
):
    """Return AD-block numerator/denominator without building a Jacobian."""
    enumer, edenom, _ = _hubbard_ad_block_objective_impl(
        coupling,
        phia,
        phib,
        weight,
        log_shift,
        psi0a,
        psi0b,
        h1e,
        U,
        ecore,
        delta,
        aux_wfac,
        random_fields,
        pop_control_randoms,
        dt,
        nup,
        ndown,
        rhf=rhf,
        trial_response=trial_response,
        uhf_n_scf=uhf_n_scf,
        uhf_mixing=uhf_mixing,
        uhf_ueff=uhf_ueff,
        stabilize_freq=stabilize_freq,
        pop_control_freq=pop_control_freq,
        measure_freq=measure_freq,
        local_pop_control=local_pop_control,
        global_pop_control=global_pop_control,
        mpi_comm=mpi_comm,
        mpi_rank=mpi_rank,
        mpi_size=mpi_size,
        checkpoint_steps=checkpoint_steps,
        coupling_mode=coupling_mode,
    )
    return enumer, edenom


@partial(
    jax.jit,
    static_argnames=("nup", "ndown", "rhf", "trial_response", "uhf_n_scf"),
)
def hubbard_response_objective(
    coupling,
    phia,
    phib,
    weight,
    log_shift,
    psi0a,
    psi0b,
    h1e,
    U,
    ecore,
    delta,
    aux_wfac,
    random_fields,
    dt,
    nup,
    ndown,
    rhf=False,
    trial_response=False,
    uhf_n_scf=50,
    uhf_mixing=0.5,
    uhf_ueff=None,
):
    """Return ``[ENumer, EDenom]`` for a fixed-field differentiable response block."""
    return _hubbard_response_objective_impl(
        coupling,
        phia,
        phib,
        weight,
        log_shift,
        psi0a,
        psi0b,
        h1e,
        U,
        ecore,
        delta,
        aux_wfac,
        random_fields,
        dt,
        nup,
        ndown,
        rhf=rhf,
        trial_response=trial_response,
        uhf_n_scf=uhf_n_scf,
        uhf_mixing=uhf_mixing,
        uhf_ueff=uhf_ueff,
    )


@partial(
    jax.jit,
    static_argnames=("nup", "ndown", "rhf", "trial_response", "uhf_n_scf"),
)
def hubbard_response_raw(
    coupling,
    phia,
    phib,
    weight,
    log_shift,
    psi0a,
    psi0b,
    h1e,
    U,
    ecore,
    delta,
    aux_wfac,
    random_fields,
    dt,
    nup,
    ndown,
    rhf=False,
    trial_response=False,
    uhf_n_scf=50,
    uhf_mixing=0.5,
    uhf_ueff=None,
):
    """Return raw numerator/denominator values and reverse-AD derivatives."""

    def _objective(cpl):
        return _hubbard_response_objective_impl(
            cpl,
            phia,
            phib,
            weight,
            log_shift,
            psi0a,
            psi0b,
            h1e,
            U,
            ecore,
            delta,
            aux_wfac,
            random_fields,
            dt,
            nup,
            ndown,
            rhf=rhf,
            trial_response=trial_response,
            uhf_n_scf=uhf_n_scf,
            uhf_mixing=uhf_mixing,
            uhf_ueff=uhf_ueff,
        )

    values = _objective(coupling)
    jac = jax.jacrev(_objective)(coupling)
    return values[0], values[1], jac[0], jac[1]


@partial(jax.jit, static_argnames=("rhf",))
def hubbard_single_site_two_body(
    phia,
    phib,
    inva,
    invb,
    weight,
    ovlp,
    psi0a,
    psi0b,
    delta,
    aux_wfac,
    random_fields,
    rhf=False,
):
    """Apply the single-site Hubbard two-body propagator with fixed fields."""
    return _hubbard_single_site_two_body_impl(
        phia,
        phib,
        inva,
        invb,
        weight,
        ovlp,
        psi0a,
        psi0b,
        delta,
        aux_wfac,
        random_fields,
        rhf=rhf,
    )


@partial(jax.jit, static_argnames=("rhf",))
def hubbard_single_site_full_step(
    phia,
    phib,
    inva,
    invb,
    weight,
    ovlp,
    log_shift,
    psi0a,
    psi0b,
    expH1,
    delta,
    aux_wfac,
    random_fields,
    dt,
    eshift,
    rhf=False,
):
    """Apply the fused kinetic/two-body/kinetic Hubbard propagation step."""
    ovlp_old = ovlp
    phia, phib, inva, invb, weight, ovlp = _kinetic_importance_sampling(
        phia, phib, inva, invb, weight, ovlp, log_shift, psi0a, psi0b, expH1, rhf
    )
    phia, phib, inva, invb, weight, ovlp = _hubbard_single_site_two_body_impl(
        phia,
        phib,
        inva,
        invb,
        weight,
        ovlp,
        psi0a,
        psi0b,
        delta,
        aux_wfac,
        random_fields,
        rhf=rhf,
    )
    phia, phib, inva, invb, weight, ovlp = _kinetic_importance_sampling(
        phia, phib, inva, invb, weight, ovlp, log_shift, psi0a, psi0b, expH1, rhf
    )
    hybrid_energy = (-(jnp.log(ovlp / ovlp_old)) / dt).real
    weight = weight * jnp.exp(dt * eshift)
    return phia, phib, inva, invb, weight, ovlp, hybrid_energy
