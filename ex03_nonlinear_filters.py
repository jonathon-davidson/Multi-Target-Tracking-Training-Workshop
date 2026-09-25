"""Module 3 companion: EKF vs UKF on range/bearing tracking (no coding required).

Run from the project root:
    streamlit run exercises/ex03_nonlinear_filters.py

A constant-velocity target flies past a fixed sensor that reports range and
bearing (both nonlinear in the Cartesian state). An EKF (Jacobian linearization)
and a UKF (sigma points, no Jacobians) run on the same measurements. Increase the
bearing uncertainty or tighten the geometry and watch where the EKF becomes
overconfident and less accurate while the UKF holds.
"""
import numpy as np
import streamlit as st

import appcommon as ac

ac.page_header("Nonlinear Filters: EKF vs UKF",
               "range/bearing tracking, and where linearization degrades",
               "Module 3")

st.markdown(
    "**What you are doing:** the sensor at the origin measures the target's "
    "**range** and **bearing** every step. Both filters estimate the Cartesian "
    "track. When bearings are precise they agree; as bearing uncertainty (the "
    "nonlinearity) grows, the EKF's tangent-line approximation and shrinking "
    "covariance start to hurt."
)

DEFAULTS = dict(cpa=1200.0, speed=250.0, sr=25.0, sb=3.0, q=3.0,
                steps=60, dt=1.0, seed=2)
for _k, _v in DEFAULTS.items():
    st.session_state.setdefault(_k, _v)

PRESETS = {
    "Custom (set the controls yourself)": None,
    "Precise sensor (both agree)": dict(cpa=1500.0, speed=250.0, sr=25.0,
                                        sb=0.5, q=3.0),
    "Coarse bearings (EKF degrades)": dict(cpa=1500.0, speed=250.0, sr=25.0,
                                           sb=10.0, q=3.0),
    "Close, fast pass": dict(cpa=300.0, speed=400.0, sr=20.0, sb=4.0, q=4.0),
}

preset = st.selectbox("Scenario preset", list(PRESETS.keys()), key="preset")
if PRESETS[preset] is not None and st.session_state.get("_applied_preset") != preset:
    for k, v in PRESETS[preset].items():
        st.session_state[k] = v
    st.session_state["_applied_preset"] = preset

col = st.columns(3)
with col[0]:
    cpa = st.slider("Miss distance / CPA (m)", 100.0, 4000.0, step=100.0, key="cpa",
                    help="How close the target passes the sensor. Smaller = more nonlinear.")
    speed = st.slider("Target speed (m/s)", 100.0, 500.0, step=25.0, key="speed")
with col[1]:
    sb = st.slider("Bearing noise (deg)", 0.5, 15.0, step=0.5, key="sb")
    sr = st.slider("Range noise (m)", 5.0, 100.0, step=5.0, key="sr")
with col[2]:
    q = st.slider("Process noise (m/s^2)", 0.5, 15.0, step=0.5, key="q")
    steps = st.slider("Steps", 30, 120, step=10, key="steps")
    seed = st.number_input("Random seed", 0, 9999, step=1, key="seed")

dt = float(st.session_state["dt"])
sb_rad = np.deg2rad(sb)
start = (-speed * dt * steps * 0.5, cpa)   # symmetric pass, CPA ~ cpa at x=0
vel = (speed, 0.0)

t, X, meas = ac.simulate_rb_track(steps, dt, sr, sb_rad, start, vel, seed=int(seed))
truth = X[:, [0, 2]]
ekf, trP_e = ac.run_ekf_rb(meas, dt, q, sr, sb_rad)
ukf, trP_u = ac.run_ukf_rb(meas, dt, q, sr, sb_rad)

# measurements converted to xy for display
mx = meas[:, 0] * np.cos(meas[:, 1])
my = meas[:, 0] * np.sin(meas[:, 1])

fig, ax = ac.new_fig(figsize=(7.4, 5.0))
ax.plot(mx / 1e3, my / 1e3, "o", color=ac.C["muted"], ms=3, alpha=0.4,
        label="measurements")
ax.plot(truth[:, 0] / 1e3, truth[:, 1] / 1e3, color=ac.C["navy"], lw=2.0,
        label="truth")
ax.plot(ekf[:, 0] / 1e3, ekf[:, 1] / 1e3, color=ac.C["accent2"], lw=1.8,
        label="EKF")
ax.plot(ukf[:, 0] / 1e3, ukf[:, 1] / 1e3, color=ac.C["accent"], lw=1.8, ls="--",
        label="UKF")
ax.plot(0, 0, "^", color=ac.C["navy"], ms=11)
ax.annotate("sensor", (0, 0), textcoords="offset points", xytext=(6, 6),
            fontsize=9, color=ac.C["navy"])
ax.set_xlabel("East (km)")
ax.set_ylabel("North (km)")
ax.set_aspect("equal", adjustable="datalim")
ax.legend(loc="best", fontsize=8)
st.pyplot(fig)

# metrics: RMSE and a consistency ratio (actual pos error^2 / reported pos var)
def rmse_xy(est):
    return float(np.sqrt(np.mean(np.sum((est - truth) ** 2, axis=1))))

def consistency(est, trP):
    err2 = np.sum((est - truth) ** 2, axis=1)
    return float(np.mean(err2[5:]) / np.mean(trP[5:]))

m = st.columns(4)
m[0].metric("EKF RMSE", f"{rmse_xy(ekf):.0f} m")
m[1].metric("UKF RMSE", f"{rmse_xy(ukf):.0f} m")
m[2].metric("EKF consistency", f"{consistency(ekf, trP_e):.1f}")
m[3].metric("UKF consistency", f"{consistency(ukf, trP_u):.1f}")
st.caption("Consistency = actual error^2 / filter-reported variance. Near 1 is "
           "honest; much greater than 1 means the filter is overconfident "
           "(reporting less uncertainty than it actually has).")

with st.expander("Questions to answer (these match the student guide)", expanded=True):
    st.markdown(
        "1. Load **Precise sensor**. Confirm EKF and UKF give the same track and "
        "RMSE. Why should they agree when bearings are precise?\n"
        "2. Raise the **bearing noise** in stages. Watch RMSE and consistency for "
        "each filter. Which degrades first, and which holds up?\n"
        "3. Load **Close, fast pass**. Where does the EKF become overconfident "
        "(consistency well above 1) compared with the UKF?\n"
        "4. The UKF uses no Jacobians. When is the EKF still the right choice, and "
        "when would you reach for the UKF?"
    )

st.info("Key idea: the EKF linearizes the range/bearing map with a tangent; as "
        "bearing uncertainty grows, that tangent misses the curved likelihood, so "
        "the EKF gets biased and overconfident. The UKF pushes sigma points "
        "through the true map and stays more consistent, with no Jacobians. The "
        "starkest version of this is bearing-only tracking, Module 4.")
