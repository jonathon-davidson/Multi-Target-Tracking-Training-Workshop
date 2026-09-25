"""Module 4 companion: bearings-only tracking and observer-maneuver observability.

Run from the project root:
    streamlit run exercises/ex04_angle_only.py

A passive observer measures only the bearing to a constant-velocity target. With
the observer on a straight course, range is unobservable; a course change restores
it. Toggle the maneuver and watch the range error.
"""
import numpy as np
import streamlit as st

import appcommon as ac

ac.page_header("Angle-Only Tracking",
               "how an observer maneuver restores range observability",
               "Module 4")

st.markdown(
    "**What you are doing:** the sensor at the moving observer reports only the "
    "**bearing** to the target, never range. On a straight observer course range "
    "cannot be recovered; a **course change** makes it observable. Toggle the "
    "maneuver and watch the range error.")

DEFAULTS = dict(maneuver_on=True, man_time=0.5, obs_speed=10.0, man_heading=90,
                sb=1.0, init_range=2800.0, filt="Cartesian EKF", seed=1)
for _k, _v in DEFAULTS.items():
    st.session_state.setdefault(_k, _v)

PRESETS = {
    "Custom (set the controls yourself)": None,
    "Straight leg (range lost)": dict(maneuver_on=False, obs_speed=10.0, sb=1.0),
    "Mid-track turn": dict(maneuver_on=True, man_time=0.5, man_heading=90,
                           obs_speed=10.0, sb=1.0),
    "Early sharp turn": dict(maneuver_on=True, man_time=0.3, man_heading=120,
                             obs_speed=14.0, sb=1.0),
}
preset = st.selectbox("Scenario preset", list(PRESETS.keys()), key="preset")
if PRESETS[preset] is not None and st.session_state.get("_applied") != preset:
    for k, v in PRESETS[preset].items():
        st.session_state[k] = v
    st.session_state["_applied"] = preset

col = st.columns(3)
with col[0]:
    maneuver_on = st.checkbox("Observer maneuvers", key="maneuver_on")
    man_time = st.slider("Maneuver time (fraction of run)", 0.2, 0.8, step=0.05,
                         key="man_time")
    man_heading = st.slider("New heading after turn (deg)", 0, 350, step=10,
                            key="man_heading")
with col[1]:
    obs_speed = st.slider("Observer speed (m/s)", 4.0, 16.0, step=1.0, key="obs_speed")
    sb = st.slider("Bearing noise (deg)", 0.25, 3.0, step=0.25, key="sb")
    init_range = st.slider("Initial range guess (m)", 1000.0, 6000.0, step=200.0,
                           key="init_range")
with col[2]:
    filt = st.radio("Filter", ["Cartesian EKF", "Modified polar", "Both"], key="filt")
    seed = st.number_input("Random seed", 0, 9999, step=1, key="seed")

# fixed scenario frame (target crossing ahead, constant velocity)
dt, steps = 20.0, 50
tgt0, tvel = (4000.0, 2000.0), (0.0, -6.0)
obs0, ovel = (0.0, 0.0), (obs_speed, 0.0)
sb_rad = np.deg2rad(sb)
h = np.deg2rad(man_heading)
maneuver = (dict(at=man_time, vel=(obs_speed * np.cos(h), obs_speed * np.sin(h)))
            if maneuver_on else None)

t, T, O, Ov, bearings = ac.simulate_bo(steps, dt, sb_rad, tgt0, tvel, obs0, ovel,
                                       maneuver=maneuver, seed=int(seed))
truth = T[:, [0, 2]]
true_r = np.hypot(truth[:, 0] - O[:, 0], truth[:, 1] - O[:, 1])

results = {}
if filt in ("Cartesian EKF", "Both"):
    est, ps = ac.run_ekf_bo(bearings, dt, O, 0.02, sb_rad, init_range)
    results["Cartesian EKF"] = (est, ps, ac.C["accent2"])
if filt in ("Modified polar", "Both"):
    est, ps = ac.run_mpc_ekf(bearings, dt, O, Ov, 0.02, sb_rad, init_range)
    results["Modified polar"] = (est, ps, ac.C["accent"])

c1, c2 = st.columns(2)

# geometry
with c1:
    fig, ax = ac.new_fig(figsize=(5.4, 4.6))
    ax.plot(O[:, 0] / 1e3, O[:, 1] / 1e3, color=ac.C["mid"], lw=2.2, label="observer")
    ax.plot(truth[:, 0] / 1e3, truth[:, 1] / 1e3, color=ac.C["navy"], lw=2.2,
            label="target (truth)")
    for name, (est, ps, col) in results.items():
        ax.plot(est[:, 0] / 1e3, est[:, 1] / 1e3, color=col, lw=1.6, ls="--",
                label=name)
    if maneuver_on:
        km = int(man_time * steps)
        ax.plot(O[km, 0] / 1e3, O[km, 1] / 1e3, "s", color=ac.C["accent2"], ms=7)
    ax.set_xlabel("East (km)"); ax.set_ylabel("North (km)")
    ax.set_aspect("equal", adjustable="datalim")
    ax.legend(loc="best", fontsize=8)
    st.pyplot(fig)

# range error over time
with c2:
    fig2, ax2 = ac.new_fig(figsize=(5.4, 4.6))
    for name, (est, ps, col) in results.items():
        est_r = np.hypot(est[:, 0] - O[:, 0], est[:, 1] - O[:, 1])
        ax2.plot(t / 60, np.abs(est_r - true_r) / 1e3, color=col, lw=2.0, label=name)
    if maneuver_on:
        ax2.axvline(t[int(man_time * steps)] / 60, color=ac.C["muted"], lw=0.9, ls=":")
        ax2.text(t[int(man_time * steps)] / 60 + 0.2, ax2.get_ylim()[1] * 0.9,
                 "maneuver", fontsize=8, color=ac.C["muted"])
    ax2.set_xlabel("time (min)"); ax2.set_ylabel("range error (km)")
    ax2.legend(loc="best", fontsize=8)
    st.pyplot(fig2)

# metrics
cols = st.columns(len(results) * 2 if results else 1)
i = 0
for name, (est, ps, col) in results.items():
    poserr = np.hypot(est[:, 0] - truth[:, 0], est[:, 1] - truth[:, 1])
    est_r = np.hypot(est[:, 0] - O[:, 0], est[:, 1] - O[:, 1])
    cols[i].metric(f"{name}: final range err", f"{abs(est_r[-1]-true_r[-1]):.0f} m"); i += 1
    cols[i].metric(f"{name}: error / reported sigma", f"{poserr[-1]/max(ps[-1],1e-9):.1f}"); i += 1
st.caption("error / reported sigma near 1 is an honest filter; much greater than 1 "
           "means it is overconfident (reporting less uncertainty than it has).")

with st.expander("Questions to answer (these match the student guide)", expanded=True):
    st.markdown(
        "1. **Straight leg (range lost)** preset. Does the range error ever settle? "
        "Compare the Cartesian EKF's reported sigma to its actual error.\n"
        "2. Turn the **observer maneuver on**. What happens to the range error at and "
        "after the turn? Record the final range error with and without the maneuver.\n"
        "3. Make the maneuver **earlier or sharper**. How does that change how fast "
        "range is resolved?\n"
        "4. Switch to **Modified polar** (or Both). Which filter's reported "
        "uncertainty is honest about what it knows?"
    )

st.info("Observability, not filtering, is the limit here: a constant-velocity "
        "observer leaves range unobservable, so no filter can recover it. The "
        "course change is what makes range recoverable; the modified-polar filter "
        "then reports its uncertainty honestly, while the Cartesian EKF tends to be "
        "overconfident.")
