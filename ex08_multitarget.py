"""Module 8 companion: multi-target association (assignment; PDA vs GNN).

Run from the project root:
    streamlit run exercises/ex08_multitarget.py

Part A solves a measurement-to-track assignment optimally and compares it to
greedy nearest-neighbor. Part B runs a single target in clutter under GNN vs PDA
and reports track error and loss. Both use the shared, verified compute in
appcommon.py, so the app agrees with the course figures.
"""
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

import appcommon as ac

ac.page_header("Multi-Target Association",
               "assignment vs greedy, and PDA vs GNN in clutter",
               "Module 8")

C = ac.COURSE_COLORS
part = st.radio("Choose a part", ["Part A - assignment vs greedy",
                                  "Part B - PDA vs GNN in clutter"],
                horizontal=True)

# =====================================================================
if part.startswith("Part A"):
    st.markdown(
        "**What you are doing:** a scan gives a **cost** for updating each track "
        "with each measurement (the negative log-likelihood: the Mahalanobis "
        "distance plus a normalizer). The optimal one-to-one assignment minimizes "
        "the total cost (**Munkres / `scipy.optimize.linear_sum_assignment`**). "
        "**Greedy** takes the cheapest cell first and repeats - fast, but it can be "
        "forced into an expensive swap.")

    c1, c2 = st.columns(2)
    with c1:
        n = st.slider("Scene size (tracks = measurements)", 2, 6, 3)
    with c2:
        amb = st.slider("Ambiguity (cost spread)", 0.0, 1.0, 0.5, 0.05,
                        help="low = costs well separated; high = many near-ties")
    seed = st.number_input("Scene seed", 0, 9999, 7, 1)

    rng = np.random.default_rng(int(seed))
    base = rng.integers(1, 10, (n, n)).astype(float)
    # blend toward a near-tie matrix as ambiguity rises
    Cm = (1 - amb) * base + amb * (5 + rng.normal(0, 0.6, (n, n)))
    Cm = np.round(Cm, 1)

    opt = ac.optimal_assign(Cm)
    grd = ac.greedy_assign(Cm)
    oc, gc = ac.assign_cost(Cm, opt), ac.assign_cost(Cm, grd)

    fig, ax = plt.subplots(figsize=(1.1 * n + 1.5, 1.1 * n + 0.5))
    ac.apply_style()
    ax.imshow(Cm, cmap="Blues", alpha=0.35)
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{Cm[i, j]:.1f}", ha="center", va="center",
                    color=C["navy"], fontsize=11)
    for i, j in enumerate(opt):
        ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False,
                     edgecolor=C["accent"], lw=3))
    for i, j in enumerate(grd):
        ax.add_patch(plt.Rectangle((j - 0.38, i - 0.38), 0.76, 0.76, fill=False,
                     edgecolor=C["accent2"], lw=2, ls="--"))
    ax.set_xticks(range(n)); ax.set_xticklabels([f"$m_{j+1}$" for j in range(n)])
    ax.set_yticks(range(n)); ax.set_yticklabels([f"$t_{i+1}$" for i in range(n)])
    ax.set_title("teal = optimal   ·   orange dashed = greedy")
    st.pyplot(fig)

    cA, cB = st.columns(2)
    cA.metric("Optimal total cost (Munkres)", f"{oc:.1f}")
    cB.metric("Greedy total cost", f"{gc:.1f}",
              delta=f"+{100*(gc-oc)/oc:.0f}% vs optimal" if oc > 0 else None,
              delta_color="inverse")
    if gc > oc + 1e-6:
        st.info("Greedy paid a penalty here: its cheapest-first pick forced a more "
                "expensive assignment elsewhere.")
    else:
        st.success("Greedy happened to match the optimal assignment this time - "
                   "raise the ambiguity or reseed to find a case where it doesn't.")
    st.caption("Deliverable: the optimal vs greedy comparison with a sentence on "
               "when greedy's shortcut costs a swap.")

# =====================================================================
else:
    st.markdown(
        "**What you are doing:** one target moves through Poisson **clutter**. "
        "**GNN** hard-picks the nearest gated return each scan; **PDA** softly "
        "weights all of them. Push the clutter up and watch GNN start losing the "
        "track while PDA holds on.")

    c1, c2, c3 = st.columns(3)
    with c1:
        nclut = st.slider("Clutter (false returns in gate)", 0.0, 6.0, 2.0, 0.5)
    with c2:
        pd = st.slider("Detection prob $P_D$", 0.50, 0.99, 0.90, 0.01)
    with c3:
        n_mc = st.slider("Monte-Carlo runs", 40, 400, 150, 20)

    res = {}
    curves = {}
    for kind in ("gnn", "pda"):
        e = None; lost = 0; rms = []
        for s in range(n_mc):
            o = ac.simulate_assoc_track(kind, seed=s, pd=pd, n_clutter=nclut)
            e = o["err"] if e is None else e + o["err"]
            lost += o["lost"]; rms.append(o["rmse"])
        curves[kind] = e / n_mc
        res[kind] = dict(loss=100 * lost / n_mc, rmse=float(np.mean(rms)))

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(9.0, 3.6),
                                   gridspec_kw={"width_ratios": [1.4, 1.0]})
    ac.apply_style()
    t = np.arange(len(curves["gnn"]))
    axL.plot(t, curves["gnn"], color=C["accent2"], lw=1.9, label="GNN (hard)")
    axL.plot(t, curves["pda"], color=C["accent"], lw=2.1, label="PDA (soft)")
    axL.set_xlabel("scan"); axL.set_ylabel("position error (m)")
    axL.legend(loc="upper right"); axL.set_title("Track error in clutter")
    axL.grid(True, alpha=0.4)
    axR.bar([0, 1], [res["gnn"]["loss"], res["pda"]["loss"]], 0.6,
            color=[C["accent2"], C["accent"]])
    axR.set_xticks([0, 1]); axR.set_xticklabels(["GNN", "PDA"])
    axR.set_ylabel("track lost (% of runs)")
    for x, k in zip([0, 1], ["gnn", "pda"]):
        axR.text(x, res[k]["loss"] + 1, f"{res[k]['loss']:.0f}%", ha="center",
                 color=C["navy"])
    axR.set_ylim(0, max(res["gnn"]["loss"], res["pda"]["loss"], 1) * 1.25)
    axR.set_title("Track loss")
    fig.tight_layout()
    st.pyplot(fig)

    m1, m2 = st.columns(2)
    m1.metric("GNN track loss", f"{res['gnn']['loss']:.0f}%",
              delta=f"RMSE {res['gnn']['rmse']:.0f} m", delta_color="off")
    m2.metric("PDA track loss", f"{res['pda']['loss']:.0f}%",
              delta=f"RMSE {res['pda']['rmse']:.0f} m", delta_color="off")
    st.caption("Deliverable: the error / loss comparison with a short paragraph on "
               "why soft association survives clutter that breaks the hard pick.")
