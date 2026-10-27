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


def extrapolate_eta(eta, values, degree=None):
    """Value at eta = 0 of the polynomial in eta fitted to the members.

    With the default degree this is Lagrange interpolation through all
    members; for eta = beta/2, beta/4, beta/8 it is the three-point Richardson
    combination. A lower degree gives the least-squares fit.

    Parameters
    ----------
    eta : :class:`numpy.ndarray`
        Trial times of the members.
    values : :class:`numpy.ndarray`
        Member values, members on the last axis.
    degree : int
        Polynomial degree. Default len(eta) - 1.

    Returns
    -------
    value : :class:`numpy.ndarray`
        Extrapolated values, shape values.shape[:-1].
    """
    eta = numpy.asarray(eta, dtype=float)
    if degree is None:
        degree = len(eta) - 1
    if not 0 <= degree < len(eta):
        raise ValueError("degree must satisfy 0 <= degree < number of members.")
    # Row of the pseudo-inverse that returns the constant coefficient.
    weights = numpy.linalg.pinv(numpy.vander(eta, degree + 1))[-1]
    return numpy.asarray(values) @ weights


def pooled_ratio(numer, denom, log_scale):
    r"""Ratio of means over blocks and its delete-one jackknife error.

    .. math::

        \mathrm{Re} \frac{\sum_b N_b e^{s_b}}{\sum_b D_b e^{s_b}}

    Parameters
    ----------
    numer, denom : :class:`numpy.ndarray`
        Block sums, blocks on axis 0; trailing axes (e.g. members) are kept.
    log_scale : :class:`numpy.ndarray`
        Log scales :math:`s_b` of the blocks.

    Returns
    -------
    value, error : :class:`numpy.ndarray`
        Ratio and jackknife standard error.
    """
    numer, denom = numpy.asarray(numer), numpy.asarray(denom)
    s = numpy.asarray(log_scale, dtype=float)
    fac = numpy.exp(s - s.max(axis=0))
    N, D = numer * fac, denom * fac
    nblocks = N.shape[0]
    value = (N.sum(axis=0) / D.sum(axis=0)).real
    if nblocks < 2:
        return value, numpy.full_like(value, numpy.nan)
    jack = ((N.sum(axis=0) - N) / (D.sum(axis=0) - D)).real
    error = numpy.sqrt((nblocks - 1) * ((jack - jack.mean(axis=0)) ** 2).mean(axis=0))
    return value, error


def summarize(results, degree=None, guide_time=None):
    """Pooled estimates with jackknife errors from :meth:`ThermofieldAFQMC.run`.

    The extrapolation is applied to the pooled members, and its error comes
    from the same delete-one jackknife.

    Parameters
    ----------
    results : dict
        Block sums returned by :meth:`ThermofieldAFQMC.run`.
    degree : int
        Degree of the extrapolation, as in :func:`extrapolate_eta`. Default None.
    guide_time : float
        Guide time; the member at eta = guide_time is reported as "mixed".
        Default None.

    Returns
    -------
    estimates : dict
        "members" (energy and particle number per eta), "eta_extrapolated"
        (at eta = 0), "mixed" and "trace", each as (value, error) pairs.
    """
    out = {}
    eta = results["eta"]
    if len(eta):
        s = results["log_scale_mix"]
        e, de = pooled_ratio(results["numer_mix"], results["denom_mix"], s)
        n, dn = pooled_ratio(results["navnum_mix"], results["denom_mix"], s)
        out["members"] = {"eta": eta, "energy": (e, de), "nav": (n, dn)}
        nblocks = s.shape[0]
        fac = numpy.exp(s - s.max(axis=0))
        D = results["denom_mix"] * fac
        extrap = {}
        for key, numer in (("energy", results["numer_mix"]), ("nav", results["navnum_mix"])):
            N = numer * fac
            value = float(extrapolate_eta(eta, (N.sum(axis=0) / D.sum(axis=0)).real, degree))
            error = numpy.nan
            if nblocks > 1:
                jack_ratio = ((N.sum(axis=0) - N) / (D.sum(axis=0) - D)).real
                jack = extrapolate_eta(eta, jack_ratio, degree)
                error = float(numpy.sqrt((nblocks - 1) * ((jack - jack.mean()) ** 2).mean()))
            extrap[key] = (value, error)
        out["eta_extrapolated"] = extrap
        if guide_time is not None:
            k = numpy.flatnonzero(numpy.isclose(eta, guide_time))
            if k.size:
                out["mixed"] = {"energy": (e[k[0]], de[k[0]]), "nav": (n[k[0]], dn[k[0]])}
    if "denom_trace" in results:
        s = results["log_scale_trace"]
        out["trace"] = {
            "energy": pooled_ratio(results["numer_trace"], results["denom_trace"], s),
            "nav": pooled_ratio(results["navnum_trace"], results["denom_trace"], s),
        }
    return out
