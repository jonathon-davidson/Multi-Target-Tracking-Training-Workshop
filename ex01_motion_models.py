"""Module 1 companion: explore the motion models (no coding required).

Run from the project root:
    streamlit run exercises/ex01_motion_models.py

This is the hands-on tool for Exercise 1. Students learn by moving the controls
and picking scenario presets, then watch how a constant-velocity (CV) path and a
coordinated-turn (CT) path diverge from the same start state. Nothing here needs
to be edited to complete the exercise.
"""
import numpy as np
import streamlit as st

import appcommon as ac

ac.page_header("Explore the Motion Models",
               "constant velocity vs coordinated turn, and what process noise does",
               "Module 1")

st.markdown(
    "**What you are doing:** a target starts at the origin and you choose how it "
    "moves. The plot shows a **constant-velocity (CV)** path and a "
    "**coordinated-turn (CT)** path from that same start, so you can see the "
    "difference a maneuver makes and what the process-noise level does to each. "
    "Use the presets for ready-made cases, or set the controls yourself."
)

# ---- control defaults live in session_state so the presets can set them -------
DEFAULTS = dict(speed=200, heading=0, omega=3.0, q=2.0, steps=60, dt=1.0, seed=7)
for _k, _v in DEFAULTS.items():
    st.session_state.setdefault(_k, _v)

PRESETS = {
    "Custom (set the controls yourself)": None,
    "Benign airliner (nearly straight)": dict(speed=250, heading=90, omega=0.0,
                                              q=1.0, steps=60, dt=1.0, seed=7),
    "Fighter in a hard bank": dict(speed=320, heading=0, omega=7.0,
                                   q=2.0, steps=60, dt=1.0, seed=7),
    "Turn rate = 0 (CT becomes CV)": dict(speed=200, heading=0, omega=0.0,
                                          q=0.0, steps=60, dt=1.0, seed=7),
    "Heavy process noise": dict(speed=200, heading=0, omega=3.0,
                                q=8.0, steps=60, dt=1.0, seed=7),
}

preset = st.selectbox("Scenario preset", list(PRESETS.keys()), key="preset")
# apply a preset only when the selection actually changes, so manual tweaks stick
if PRESETS[preset] is not None and st.session_state.get("_applied_preset") != preset:
    for k, v in PRESETS[preset].items():
        st.session_state[k] = v
    st.session_state["_applied_preset"] = preset

col = st.columns(3)
with col[0]:
    speed = st.slider("Speed (m/s)", 50, 400, step=10, key="speed")
    heading = st.slider("Initial heading (deg): 0 = east, 90 = north",
                        0, 360, step=5, key="heading")
with col[1]:
    omega = st.slider("Turn rate (deg/s)  -  0 means straight ahead",
                      0.0, 9.0, step=0.5, key="omega")
    q = st.slider("Process-noise level (m/s^2)", 0.0, 10.0, step=0.5, key="q")
with col[2]:
    steps = st.slider("Number of steps", 10, 200, step=5, key="steps")
    dt = st.slider("Time per step (s)", 0.2, 2.0, step=0.1, key="dt")
    seed = st.number_input("Random seed", 0, 9999, step=1, key="seed")

show_err = st.checkbox(
    "Highlight the error a straight-line (CV) model makes on this target",
    value=True)

# ---- propagate both models from the SAME start state and process noise --------
xcv, ycv = ac.propagate_cv(steps, dt, speed, heading, q, seed=int(seed))
xct, yct = ac.propagate_ct(steps, dt, speed, omega, heading, q, seed=int(seed))

fig, ax = ac.new_fig(figsize=(7.0, 5.0))
ax.plot(xct / 1e3, yct / 1e3, color=ac.C["mid"], lw=2.2,
        label="Target's true path (coordinated turn)")
ax.plot(xcv / 1e3, ycv / 1e3, color=ac.C["accent"], lw=2.0, ls="--",
        label="Constant-velocity model")
ax.plot(0, 0, "o", color=ac.C["navy"], ms=8)
ax.annotate("start", (0, 0), textcoords="offset points", xytext=(8, 8),
            color=ac.C["navy"], fontsize=9)

sep_km = float(np.hypot(xcv[-1] - xct[-1], ycv[-1] - yct[-1]) / 1e3)
if show_err and sep_km > 1e-6:
    ax.annotate("", xy=(xct[-1] / 1e3, yct[-1] / 1e3),
                xytext=(xcv[-1] / 1e3, ycv[-1] / 1e3),
                arrowprops=dict(arrowstyle="<->", color=ac.C["accent2"], lw=1.6))
    mx = (xcv[-1] + xct[-1]) / 2e3
    my = (ycv[-1] + yct[-1]) / 2e3
    ax.annotate(f"CV model error\nafter {steps * dt:.0f} s: {sep_km:.1f} km",
                (mx, my), textcoords="offset points", xytext=(10, 0),
                color=ac.C["accent2"], fontsize=9, va="center")

ax.set_xlabel("East (km)")
ax.set_ylabel("North (km)")
ax.set_aspect("equal", adjustable="datalim")
ax.legend(loc="best")
st.pyplot(fig)

# ---- live read-outs -----------------------------------------------------------
turn_radius_km = float(speed / np.deg2rad(omega) / 1e3) if omega > 0 else float("inf")
m = st.columns(3)
m[0].metric("CV-CT separation at end", f"{sep_km:.2f} km")
m[1].metric("CT turn radius",
            "straight" if omega == 0 else f"{turn_radius_km:.2f} km")
m[2].metric("Total time", f"{steps * dt:.0f} s")

with st.expander("Questions to answer (these match the student guide)", expanded=True):
    st.markdown(
        "1. Set **Turn rate = 0**. The two paths land on top of each other. Why "
        "does the coordinated-turn model become the constant-velocity model when "
        "the turn rate is zero?\n"
        "2. Set the turn rate to about **3 deg/s** and read the **CV-CT "
        "separation** above. That is roughly the error a CV tracker builds up on "
        "this maneuvering target. Where does the error come from?\n"
        "3. Slide the **process-noise level** from 0 up to about 8. Describe how "
        "the paths change, and name one real source of that uncertainty.\n"
        "4. Load the **Benign airliner** and **Fighter in a hard bank** presets. "
        "Which motion model fits each, and why?"
    )

st.info("The dashed line is what a constant-velocity tracker would expect; the "
        "solid line is what a maneuvering target actually does. That growing gap "
        "is the maneuver problem, which Module 5 (adaptive and IMM filters) "
        "solves.")
