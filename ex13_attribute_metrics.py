"""Module 13 companion: attribute fusion & performance metrics (capstone).

Run from the project root:
    streamlit run exercises/ex13_attribute_metrics.py

Two parts, chosen with the selector at the top:
  A. Attribute fusion - fuse noisy type declarations with Bayes and with
     Dempster-Shafer, including the high-conflict (Zadeh) case.
  B. Scoring a run - score a multi-target run with RMSE, track purity, and OSPA,
     and watch OSPA react to a dropped or spurious track.

Uses the shared, verified compute in appcommon.py so the app and the course
figures agree.
"""
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

import appcommon as ac

ac.page_header("Attribute Fusion & Performance Metrics",
               "class fusion (Bayes / Dempster-Shafer) and tracking metrics (RMSE / purity / OSPA)",
               "Module 13")

C = ac.COURSE_COLORS

part = st.radio("Choose a part",
                ["A - Attribute fusion: Bayes vs Dempster-Shafer",
                 "B - Scoring a run: RMSE, purity, OSPA"],
                horizontal=True)

# =====================================================================
# PART A - attribute fusion
# =====================================================================
if part.startswith("A"):
    st.markdown(
        "**What you are doing:** a classifier emits noisy **type declarations** about a "
        "target. Fuse them with a **Bayesian** class-probability update (a point posterior) "
        "and compare to **Dempster-Shafer** belief/plausibility (an interval that models "
        "ignorance). Then try the **high-conflict** case, where Dempster's rule misbehaves.")

    c1, c2, c3 = st.columns(3)
    with c1:
        correct = st.slider("Classifier accuracy (% correct)", 40, 95, 70, 5) / 100.0
    with c2:
        n = st.slider("Number of declarations", 1, 20, 10, 1)
    with c3:
        conflict = st.checkbox("High-conflict (Zadeh) case", value=False)

    conf = ac.default_confusion(3, correct)
    h = ac.bayes_run(0, conf, n, seed=3)   # true class = A (index 0)

    fig, ax = ac.new_fig(figsize=(7.4, 3.4))
    names = ["class A (true)", "class B", "class C"]
    cols = [C["accent"], C["accent2"], C["mid"]]
    for i in range(3):
        ax.plot(range(n + 1), h[:, i], "-o", ms=3,
                lw=(2.6 if i == 0 else 1.8), color=cols[i], label=names[i])
    ax.axhline(0.95, color=C["muted"], ls=":", lw=1.2)
    ax.set_ylim(-0.03, 1.05)
    ax.set_xlabel("number of declarations"); ax.set_ylabel("class probability")
    ax.legend(loc="center right", fontsize=8)
    ax.set_title("Bayesian class fusion")
    st.pyplot(fig)

    avg = ac.mean_decls_to_confidence(0, conf)
    m1, m2 = st.columns(2)
    m1.metric("P(true class) after all declarations", f"{h[-1, 0]:.3f}")
    m2.metric("Avg declarations to 0.95", f"{avg:.1f}")

    st.markdown("**Dempster-Shafer on the same class set:**")
    A, B, Cc = frozenset("A"), frozenset("B"), frozenset("C")
    if conflict:
        m_1 = {A: 0.99, Cc: 0.01}
        m_2 = {B: 0.99, Cc: 0.01}
        st.caption("Sensor 1: A=0.99, C=0.01   ·   Sensor 2: B=0.99, C=0.01")
    else:
        m_1 = {A: 0.6, A | B: 0.3, A | B | Cc: 0.1}
        m_2 = {A: 0.5, Cc: 0.2, A | B | Cc: 0.3}
        st.caption("Sensor 1: A=0.6, {A,B}=0.3, ignorance=0.1   ·   "
                   "Sensor 2: A=0.5, C=0.2, ignorance=0.3")
    comb, K = ac.dempster_combine(m_1, m_2)
    d1, d2, d3 = st.columns(3)
    d1.metric("Conflict K", f"{K:.3f}",
              delta="HIGH - rule unreliable" if K > 0.9 else "ok", delta_color="off")
    d2.metric("Combined belief(A)", f"{ac.belief(comb, A):.2f}")
    d3.metric("Combined plausibility(A)", f"{ac.plausibility(comb, A):.2f}")
    combined_str = ", ".join(f"{''.join(sorted(k))}={v:.2f}" for k, v in comb.items())
    st.write("Combined masses:", combined_str)

    if conflict:
        st.info(f"Conflict K = {K:.3f}. Both sensors put almost all their mass on DIFFERENT "
                "classes (A and B) and only 0.01 each on C, yet Dempster's rule, normalizing "
                "by the near-total conflict, certifies **C**, the class both nearly ruled "
                "out. Bayes would not do this; it is the classic Dempster-Shafer pitfall.")
    else:
        st.success("Low conflict: Bayes gives a sharp point posterior, Dempster-Shafer gives "
                   "a belief-plausibility interval that also holds some explicit ignorance. "
                   "Both agree on the leading class. Turn on high-conflict to break the rule.")

# =====================================================================
# PART B - scoring a run
# =====================================================================
else:
    st.markdown(
        "**What you are doing:** score a two-target run with **RMSE** (accuracy), **track "
        "purity** (continuity), and **OSPA** (a composite of localization and cardinality). "
        "Inject a **dropped** or **spurious** track and watch OSPA react where RMSE, over the "
        "matched track, barely moves.")

    c1, c2, c3 = st.columns(3)
    with c1:
        c_cut = st.slider("OSPA cutoff $c$ (m)", 50.0, 300.0, 100.0, 10.0)
    with c2:
        noise = st.slider("Position noise (m)", 5.0, 60.0, 22.0, 1.0)
    with c3:
        defect = st.selectbox("Inject", ["nothing", "dropped track", "spurious track"])

    rng = np.random.default_rng(2)
    T = np.array([[0.0, 0.0], [500.0, 0.0]])
    steps = 40
    tot, loc, card, rmse_matched = [], [], [], []
    for k in range(steps):
        est = T + rng.normal(0, noise, T.shape)
        window = (16 <= k < 24)
        if defect == "dropped track" and window:
            est = est[:1]
        elif defect == "spurious track" and window:
            est = np.vstack([est, [250.0, 300.0]])
        t_, l_, cd_ = ac.ospa(T, est, c=c_cut, p=2.0)
        tot.append(t_); loc.append(l_); card.append(cd_)
        # RMSE over the MATCHED first track only (what a naive report shows)
        rmse_matched.append(np.linalg.norm(est[0] - T[0]))
    ks = np.arange(steps)

    fig, ax = ac.new_fig(figsize=(7.6, 3.6))
    ax.fill_between(ks, 0, loc, color=C["accent"], alpha=0.3, label="OSPA localization")
    ax.fill_between(ks, loc, np.array(loc) + np.array(card), color=C["accent2"],
                    alpha=0.3, label="OSPA cardinality")
    ax.plot(ks, tot, "-", color=C["navy"], lw=2.2, label="OSPA total")
    ax.plot(ks, rmse_matched, "--", color=C["mid"], lw=1.6,
            label="RMSE (matched track)")
    if defect != "nothing":
        ax.axvspan(16, 24, color=C["muted"], alpha=0.10)
        ax.text(20, max(tot) * 0.95, defect, ha="center", fontsize=8, color=C["muted"])
    ax.set_xlabel("scan"); ax.set_ylabel(f"distance (m, c={c_cut:.0f})")
    ax.legend(loc="upper left", fontsize=7.5, ncol=2)
    ax.set_title("OSPA vs RMSE over the run")
    st.pyplot(fig)

    # a track-purity illustration (one track swaps mid-run)
    labels = [0] * 13 + [1] * 7
    m1, m2, m3 = st.columns(3)
    m1.metric("Mean RMSE (matched)", f"{np.mean(rmse_matched):.0f} m")
    m2.metric("Mean OSPA", f"{np.mean(tot):.0f} m",
              delta="feels the defect" if defect != "nothing" else "loc only",
              delta_color="off")
    m3.metric("Example track purity", f"{ac.track_purity(labels):.2f}",
              delta="one swap", delta_color="off")

    if defect != "nothing":
        st.info("Note the split: RMSE over the matched track hardly changes, but OSPA jumps "
                "during the defect because the cardinality term charges for the missing or "
                "extra track. A single composite metric is needed precisely because accuracy "
                "and cardinality are separate failure modes.")
    else:
        st.success("With both targets tracked, OSPA follows the localization error alone and "
                   "sits near the RMSE. Inject a dropped or spurious track to separate them.")

st.caption("Deliverables: (a) the class-confidence comparison including the high-conflict "
           "case, with a line on when each framework is appropriate; (b) the metrics report "
           "(RMSE, purity, OSPA) with a paragraph on why a composite metric is needed and what "
           "a fair Monte-Carlo evaluation requires.")
