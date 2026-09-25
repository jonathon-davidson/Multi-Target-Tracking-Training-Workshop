"""Module 2 companion: linear filters (no coding required).

Run from the project root:
    streamlit run exercises/ex02_linear_filters.py

Drive a Kalman filter on a target you configure, feel the process-noise (Q) vs
measurement-noise (R) trade-off (Exercise 2a), then switch to a fixed-gain
alpha-beta filter and tune its gains to match the Kalman filter (Exercise 2b).
"""
import numpy as np
import streamlit as st

import appcommon as ac

ac.page_header("Linear Filters: Kalman and Alpha-Beta",
               "tune Q and R, then match an alpha-beta filter to the Kalman filter",
               "Module 2")

st.markdown(
    "**What you are doing:** a target moves along a line and a noisy sensor "
    "reports its position. You run a filter on those measurements and compare the "
    "estimate to the truth. Use **Kalman** mode for Exercise 2a (tune Q and R) and "
    "**Alpha-beta** mode for Exercise 2b (match the fixed-gain filter to the "
    "Kalman filter)."
)

DEFAULTS = dict(target="Maneuvering", man_accel=9.0, r=90.0, q=3.0,
                alpha=0.5, beta=0.1, steps=60, dt=1.0, seed=2)
for _k, _v in DEFAULTS.items():
    st.session_state.setdefault(_k, _v)

PRESETS = {
    "Custom (set the controls yourself)": None,
    "Quiet target, noisy sensor": dict(target="Constant velocity", man_accel=9.0,
                                       r=160.0, q=3.0),
    "Maneuvering target": dict(target="Maneuvering", man_accel=9.0, r=90.0, q=3.0),
    "Sluggish filter (low Q)": dict(target="Maneuvering", man_accel=9.0,
                                    r=90.0, q=0.3),
    "Twitchy filter (high Q)": dict(target="Maneuvering", man_accel=9.0,
                                    r=90.0, q=20.0),
}

preset = st.selectbox("Scenario preset", list(PRESETS.keys()), key="preset")
if PRESETS[preset] is not None and st.session_state.get("_applied_preset") != preset:
    for k, v in PRESETS[preset].items():
        st.session_state[k] = v
    st.session_state["_applied_preset"] = preset

col = st.columns(3)
with col[0]:
    target = st.selectbox("Target motion", ["Constant velocity", "Maneuvering"],
                          key="target")
    man_accel = st.slider("Maneuver acceleration (m/s^2)", 0.0, 25.0, step=1.0,
                          key="man_accel",
                          help="Only used when the target is maneuvering.")
    r = st.slider("Measurement noise R: sensor std (m)", 20.0, 300.0, step=10.0,
                  key="r")
with col[1]:
    filt = st.radio("Filter", ["Kalman", "Alpha-beta"], key="filter",
                    horizontal=True)
    if filt == "Kalman":
        q = st.slider("Process noise Q: accel std (m/s^2)", 0.1, 25.0, step=0.1,
                      key="q")
        alpha = st.session_state["alpha"]
        beta = st.session_state["beta"]
    else:
        q = st.session_state["q"]
        alpha = st.slider("alpha (position gain)", 0.02, 0.99, step=0.01,
                          key="alpha")
        beta = st.slider("beta (velocity gain)", 0.001, 0.90, step=0.001,
                         key="beta", format="%.3f")
with col[2]:
    steps = st.slider("Number of steps", 20, 200, step=5, key="steps")
    dt = st.slider("Time per step (s)", 0.2, 2.0, step=0.1, key="dt")
    seed = st.number_input("Random seed", 0, 9999, step=1, key="seed")

# ---- simulate the truth + measurements, run the filters -----------------------
maneuver = dict(start=0.45, stop=0.55, accel=man_accel) if target == "Maneuvering" else None
t, pos, vel, meas = ac.simulate_cv_track(steps, dt, r, speed=200.0,
                                         maneuver=maneuver, seed=int(seed))

# Kalman is always run so it can serve as the reference for Exercise 2b
kf_p, kf_v, kf_pvar, kf_gain = ac.run_kf(meas, dt, q, r)
kf_rmse = ac.rmse(kf_p[15:], pos[15:])
kf_gain_ss = float(kf_gain[-1])

if filt == "Kalman":
    est = kf_p
    est_label = "Kalman estimate"
else:
    ab_p, ab_v = ac.run_alpha_beta(meas, dt, alpha, beta)
    est = ab_p
    est_label = f"alpha-beta estimate ($\\alpha$={alpha:.2f}, $\\beta$={beta:.3f})"
sel_rmse = ac.rmse(est[15:], pos[15:])

fig, ax = ac.new_fig(figsize=(7.4, 4.4))
ax.plot(t, meas / 1e3, "o", color=ac.C["muted"], ms=3, alpha=0.5,
        label="measurements")
ax.plot(t, pos / 1e3, color=ac.C["accent2"], lw=1.6, ls="--", label="truth")
ax.plot(t, est / 1e3, color=ac.C["accent"], lw=2.0, label=est_label)
if maneuver is not None:
    ax.axvline(t[steps // 2], color=ac.C["muted"], lw=0.8, ls=":")
    ax.text(t[steps // 2], ax.get_ylim()[0], " maneuver", fontsize=8,
            color=ac.C["muted"], va="bottom")
ax.set_xlabel("time (s)")
ax.set_ylabel("position (km)")
ax.legend(loc="upper left")
st.pyplot(fig)

m = st.columns(3)
m[0].metric(f"{filt} position RMSE", f"{sel_rmse:.0f} m")
m[1].metric("Kalman RMSE (reference)", f"{kf_rmse:.0f} m")
m[2].metric("Kalman steady-state gain", f"{kf_gain_ss:.3f}")

if filt == "Alpha-beta":
    st.caption(f"Match target: get the alpha-beta RMSE down to about the Kalman "
               f"RMSE of {kf_rmse:.0f} m. Try setting alpha near the Kalman gain "
               f"{kf_gain_ss:.3f}.")

with st.expander("Questions to answer (these match the student guide)", expanded=True):
    st.markdown(
        "**Exercise 2a (Kalman mode).** Use the *Maneuvering target* preset.\n\n"
        "1. Turn **Q** to its lowest setting. What does the estimate do at the "
        "maneuver, and why (in terms of the gain)?\n"
        "2. Turn **Q** high. What happens between maneuvers, and why?\n"
        "3. Find a **Q** that balances the two and note the RMSE.\n"
        "4. Raise **R** and watch the steady-state gain. Which way does it move, "
        "and what is the filter then doing?\n\n"
        "**Exercise 2b (Alpha-beta mode).** Note the Kalman RMSE and gain first.\n\n"
        "5. Switch to **Alpha-beta** and tune **alpha, beta** until the RMSE "
        "matches the Kalman filter. How does your **alpha** compare to the Kalman "
        "gain?\n"
        "6. Which filter settles faster from a cold start, and why?"
    )

st.info("Key idea: alpha is the Kalman filter's steady-state position gain and "
        "beta/dt its steady-state velocity gain, so a matched alpha-beta filter "
        "equals the Kalman filter once both have settled. The Kalman filter still "
        "wins the first few seconds because its gain starts high and adapts down.")
