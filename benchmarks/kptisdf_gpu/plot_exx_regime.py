#!/usr/bin/env python
"""Plot the exx algorithm regime sweep (sweep_exx_algo.py BENCH_RESULT lines).

Produces, from the nw=8 main grid:
  exx_speedup_vs_Nbsf.pdf : old kernel (its nbsf > 8*nk heuristic) over new
                            kernel (calibrated cost model), per nk
  exx_regime_vs_Nbsf.pdf  : t_new_lowk / t_new_largek per nk -- above 1 means
                            the dense-k (Nk^2 N^4) path wins
Usage: plot_exx_regime.py results/*.out
"""

import json
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

C_BW = 330  # must match kpt_isdf_exx_kernel_gpu


def model_algo(nk, nbsf, nocc, nw, nisdf):
    f_lgk = nk * nocc * nbsf * nisdf**2 + 3 * nk**2 * nw * nocc * nbsf * nisdf
    f_lowk = 2 * nk**2 * nw * nocc * nisdf**2 + 2 * nk**2 * nw * nocc * nbsf * nisdf
    return "largek" if f_lgk <= f_lowk + C_BW * nw * nk**2 * nisdf**2 else "lowk"


rows = []
for path in sys.argv[1:]:
    for line in open(path):
        if line.startswith("BENCH_RESULT "):
            r = json.loads(line[len("BENCH_RESULT ") :])
            if "skipped" not in r:
                rows.append(r)

main = [r for r in rows if r["nwalkers"] == 8 and r["nocc"] == max(2, round(0.09 * r["nbsf"]))]
nks = sorted({r["nk"] for r in main})
markers = dict(zip(nks, "osD^vP"))

fig, ax = plt.subplots(figsize=(5, 4))
for nk in nks:
    pts = []
    for r in sorted(main, key=lambda r: r["nbsf"]):
        if r["nk"] != nk:
            continue
        t_old = r[f"t_old_{r['old_heuristic_algo']}_s"]
        algo = model_algo(r["nk"], r["nbsf"], r["nocc"], r["nwalkers"], r["nisdf"])
        t_new = r[f"t_new_{algo}_s"]
        if t_old is not None and t_new is not None:
            pts.append((r["nbsf"], t_old / t_new))
    if pts:
        ax.plot(*zip(*pts), marker=markers[nk], label=f"$N_k$ = {nk}")
ax.set_xscale("log")
ax.set_yscale("log")
ax.axhline(1.0, color="gray", ls="--", lw=0.8)
ax.set_xlabel(r"$N_{\mathrm{bsf}}$")
ax.set_ylabel("speedup (old heuristic / new selected)")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig("exx_speedup_vs_Nbsf.pdf")

fig, ax = plt.subplots(figsize=(5, 4))
for nk in nks:
    pts = [
        (r["nbsf"], r["t_new_lowk_s"] / r["t_new_largek_s"])
        for r in sorted(main, key=lambda r: r["nbsf"])
        if r["nk"] == nk
        and r.get("t_new_lowk_s") is not None
        and r.get("t_new_largek_s") is not None
    ]
    if pts:
        ax.plot(*zip(*pts), marker=markers[nk], label=f"$N_k$ = {nk}")
ax.set_xscale("log")
ax.set_yscale("log")
ax.axhline(1.0, color="gray", ls="--", lw=0.8)
ax.set_xlabel(r"$N_{\mathrm{bsf}}$")
ax.set_ylabel(r"$t_{\mathrm{low}k}\,/\,t_{\mathrm{dense}k}$ (new kernel)")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig("exx_regime_vs_Nbsf.pdf")
print("wrote exx_speedup_vs_Nbsf.pdf exx_regime_vs_Nbsf.pdf")
