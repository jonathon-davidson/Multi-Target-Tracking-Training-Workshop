"""Module 6 companion: gating and nearest-neighbor association in clutter.

Run from the project root:
    streamlit run exercises/ex06_association.py

A single constant-velocity target moves through Poisson clutter. An ellipsoidal
(chi-square) gate is drawn around the filter's prediction, and nearest-neighbor
picks the gated return of smallest Mahalanobis distance. Change the clutter
density, detection probability, gate probability, and gate shape, and watch the
mis-association rate.
"""
import numpy as np
import streamlit as st

import appcommon as ac

ac.page_header("Single-Target Correlation & Association",
               "gating and nearest-neighbor in clutter",
               "Module 6")

st.markdown(
    "**What you are doing:** each scan the sensor returns the target (with "
    "probability $P_D$) plus **clutter**. A gate around the prediction discards "
    "implausible returns; **nearest-neighbor** then updates the track with the "
    "gated return of smallest **Mahalanobis** distance. Push the clutter up and "
    "watch nearest-neighbor start picking the wrong return.")

DEFAULTS = dict(beta5=1.0, pd=0.95, pg=0.99, gate="ellipse", seed=3)
for _k, _v in DEFAULTS.items():
    st.session_state.setdefault(_k, _v)

PRESETS = {
    "Custom (set the controls yourself)": None,
    "Light clutter": dict(beta5=0.5, pd=0.97, pg=0.99, gate="ellipse"),
    "Moderate clutter": dict(beta5=2.0, pd=0.95, pg=0.99, gate="ellipse"),
    "Dense clutter": dict(beta5=6.0, pd=0.95, pg=0.99, gate="ellipse"),
    "Low detection probability": dict(beta5=2.0, pd=0.75, pg=0.99, gate="ellipse"),
}
preset = st.selectbox("Scenario preset", list(PRESETS.keys()), key="preset")
if PRESETS[preset] is not None and st.session_state.get("_applied") != preset:
    for k, v in PRESETS[preset].items():
        st.session_state[k] = v
    st.session_state["_applied"] = preset

col = st.columns(3)
with col[0]:
    beta5 = st.slider("Clutter density (1e-5 returns / m²)", 0.0, 8.0, step=0.5,
                      key="beta5")
    seed = st.number_input("Random seed", 0, 9999, step=1, key="seed")
with col[1]:
    pd = st.slider("Detection probability P_D", 0.50, 1.0, step=0.05, key="pd")
    pg = st.slider("Gate probability P_G", 0.90, 0.999, step=0.005, key="pg")
with col[2]:
    gate = st.radio("Gate shape", ["ellipse", "rect"],
                    format_func=lambda s: "Ellipsoidal" if s == "ellipse" else "Rectangular",
                    key="gate")

beta = beta5 * 1e-5
# elongated measurement covariance (good along-track, poor cross-track) so the
# gate shape is visibly non-circular
R = ac.cov_ellipse_R(150.0, 60.0, 20.0)
dt, steps = 1.0, 60

res = ac.simulate_clutter_track(steps, dt, R, beta, pd, pg=pg, seed=int(seed),
                                gate=gate)
gamma = res["gamma"]

# choose an illustrative frame from the EARLY scans (before any divergence
# inflates the gate): the one with the most returns in view
frames = res["frames"]
early = min(15, len(frames))
fi = int(np.argmax([len(frames[i]["returns"]) for i in range(early)])) if frames else 0
fr = frames[fi]

c1, c2 = st.columns(2)

# ---- geometry of one scan ----
with c1:
    fig, ax = ac.new_fig(figsize=(5.6, 4.6))
    zhat, S, returns = fr["zhat"], fr["S"], fr["returns"]
    ell = ac.gate_ellipse(zhat, S, gamma)
    ax.plot(ell[:, 0] / 1e3, ell[:, 1] / 1e3, color=ac.C["accent"], lw=2.0,
            label="gate")
    ax.plot(zhat[0] / 1e3, zhat[1] / 1e3, "P", color=ac.C["navy"], ms=12,
            label="prediction")
    if len(returns):
        mask = (ac.ellipsoidal_gate(returns, zhat, S, gamma) if gate == "ellipse"
                else ac.rectangular_gate(returns, zhat, S, gamma))
        # clutter vs target return
        for j, z in enumerate(returns):
            is_t = (fr["target_idx"] is not None and j == fr["target_idx"])
            if is_t:
                ax.plot(z[0] / 1e3, z[1] / 1e3, "*", color=ac.C["accent2"], ms=17,
                        zorder=5, label="true detection")
            else:
                ax.plot(z[0] / 1e3, z[1] / 1e3, "o",
                        color=ac.C["mid"] if mask[j] else "none",
                        markeredgecolor=ac.C["muted"], ms=7, zorder=3)
        if fr["pick"] is not None:
            zp = returns[fr["pick"]]
            ax.plot(zp[0] / 1e3, zp[1] / 1e3, "o", mfc="none",
                    mec=ac.C["navy"], mew=2.2, ms=16, zorder=6, label="NN pick")
    ax.set_xlabel("East (km)"); ax.set_ylabel("North (km)")
    ax.set_aspect("equal", adjustable="datalim")
    ax.legend(loc="best", fontsize=8)
    ax.set_title(f"one scan (of {steps}): {len(returns)} returns in view", fontsize=10)
    st.pyplot(fig)

# ---- mis-association rate vs clutter density ----
@st.cache_data(show_spinner=False)
def sweep(pd, pg, gate, sr, sc, ang):
    Rr = ac.cov_ellipse_R(sr, sc, ang)
    betas = np.array([0.0, 1e-6, 2e-6, 5e-6, 1e-5, 2e-5, 4e-5, 6e-5, 8e-5])
    rates = ac.misassoc_vs_density(betas, R=Rr, pd=pd, pg=pg, steps=60,
                                   n_mc=30, gate=gate)
    return betas, rates

with c2:
    betas, rates = sweep(pd, pg, gate, 150.0, 60.0, 20.0)
    fig2, ax2 = ac.new_fig(figsize=(5.6, 4.6))
    ax2.fill_between(betas * 1e5, 0, rates, color=ac.C["accent"], alpha=0.10)
    ax2.plot(betas * 1e5, rates, color=ac.C["accent"], lw=2.2, marker="o", ms=4)
    ax2.axvline(beta5, color=ac.C["accent2"], lw=1.6, ls="--")
    ax2.text(beta5, 0.03, " you are here", color=ac.C["accent2"], fontsize=8)
    ax2.axhline(0.5, color=ac.C["muted"], lw=0.8, ls=":")
    ax2.set_xlabel("clutter density (1e-5 returns / m²)")
    ax2.set_ylabel("mis-association rate")
    ax2.set_ylim(-0.02, 1.0)
    st.pyplot(fig2)

# ---- metrics ----
ncand = np.mean([int((ac.ellipsoidal_gate(f["returns"], f["zhat"], f["S"], gamma)
                      if gate == "ellipse" else
                      ac.rectangular_gate(f["returns"], f["zhat"], f["S"], gamma)).sum())
                 if len(f["returns"]) else 0 for f in frames[:early]]) if frames else 0.0
m = st.columns(3)
m[0].metric("Mis-association rate (this run)", f"{res['misassoc_rate']*100:.0f}%")
m[1].metric("Gate threshold γ", f"{gamma:.2f}")
m[2].metric("Avg returns in gate / scan", f"{ncand:.1f}")
st.caption("Mis-association rate = fraction of detected scans whose nearest-neighbor "
           "pick was NOT the true target return (a clutter return won, or the target "
           "was missed and clutter was picked).")

with st.expander("Questions to answer (these match the student guide)", expanded=True):
    st.markdown(
        "1. **Light clutter** preset. Does nearest-neighbor almost always pick the "
        "true return? Watch the mis-association rate.\n"
        "2. Raise the **clutter density** in steps and record the rate at each. Where "
        "does it start climbing steeply?\n"
        "3. Lower the **detection probability** $P_D$. How does missing the target more "
        "often change the rate at a fixed clutter density?\n"
        "4. Raise the **gate probability** $P_G$ to widen the gate. Does a bigger gate "
        "help or hurt in clutter, and why?\n"
        "5. Switch the gate to **rectangular**. Does the rate change much? What does "
        "that say about where the extra clutter the rectangle admits actually lands?")

st.info("Gating and nearest-neighbor are cheap and work well when clutter is light. "
        "As the gate fills with clutter, the nearest return is increasingly a false "
        "one, and a single hard pick each scan is no longer enough: that is what "
        "drives the assignment and probabilistic methods (PDA, MHT) of Module 8.")
