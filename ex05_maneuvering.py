"""Module 5 companion: maneuver detection and the IMM estimator.

Run from the project root:
    streamlit run exercises/ex05_maneuvering.py

A target flies straight, turns at a constant rate, then straightens out. Part A
runs a single Kalman filter and shows its normalized innovation squared (NIS) as
a maneuver detector. Part B runs a two-model IMM (constant velocity + coordinated
turn) and shows the mode probabilities switching through the turn.
"""
import numpy as np
import streamlit as st

import appcommon as ac

ac.page_header("Maneuvering Targets",
               "maneuver detection and the Interacting Multiple Model estimator",
               "Module 5")

view = st.radio(
    "Choose a view",
    ["Part A - Maneuver detector (single filter)",
     "Part B - Two-model IMM (CV + coordinated turn)"],
    key="view", horizontal=True)

st.markdown(
    "**The scenario:** the target flies straight, executes a constant-rate "
    "**coordinated turn** in the shaded window, then flies straight again. A "
    "single constant-velocity filter has no term for the turn, so it lags. Watch "
    "how the filter reveals the maneuver (Part A) and how a model bank tracks "
    "through it (Part B).")

# ------------------------------------------------------------------ scenario controls
DEFAULTS = dict(speed=200.0, turn_rate=3.0, meas_noise=80.0, seed=3,
                thr=6.0, win=3, q_cv=2.0, imm_turn=3.0, stay=0.95)
for _k, _v in DEFAULTS.items():
    st.session_state.setdefault(_k, _v)

dt, steps = 1.0, 80
turn_window = (0.375, 0.625)

sc = st.columns(3)
with sc[0]:
    speed = st.slider("Target speed (m/s)", 100.0, 300.0, step=10.0, key="speed")
with sc[1]:
    turn_rate = st.slider("True turn rate (deg/s)", 1.0, 6.0, step=0.5, key="turn_rate")
with sc[2]:
    meas_noise = st.slider("Measurement noise (m)", 20.0, 200.0, step=10.0,
                           key="meas_noise")

t, X, meas = ac.simulate_maneuver_2d(steps, dt, meas_noise, speed=speed,
                                     heading_deg=0.0, turn_rate_deg=turn_rate,
                                     turn_window=turn_window,
                                     seed=int(st.session_state["seed"]))
truth = X[:, [0, 2]]
ks, ke = int(turn_window[0] * steps), int(turn_window[1] * steps)


def _err(est):
    return np.hypot(est[:, 0] - truth[:, 0], est[:, 1] - truth[:, 1])


# ================================================================== Part A
if view.startswith("Part A"):
    st.subheader("Part A - the innovation as a maneuver detector")
    ca = st.columns(2)
    with ca[0]:
        thr = st.slider("Detection threshold (NIS)", 2.0, 20.0, step=0.5, key="thr")
    with ca[1]:
        win = st.slider("Detector window (steps, moving average)", 1, 8, step=1,
                        key="win")

    est, nis = ac.run_kf2d(meas, dt, meas_noise, model="cv",
                           q=st.session_state["q_cv"])
    # moving-average NIS for the detector
    k = int(win)
    if k > 1:
        kernel = np.ones(k) / k
        nis_s = np.convolve(nis, kernel, mode="same")
    else:
        nis_s = nis
    detected = nis_s > thr

    c1, c2 = st.columns(2)
    with c1:
        fig, axg = ac.new_fig(figsize=(5.4, 4.4))
        axg.plot(meas[:, 0] / 1e3, meas[:, 1] / 1e3, ".", color=ac.C["muted"],
                 ms=3, alpha=0.5, label="measurements")
        axg.plot(truth[:, 0] / 1e3, truth[:, 1] / 1e3, color=ac.C["navy"],
                 lw=2.2, label="truth")
        axg.plot(est[:, 0] / 1e3, est[:, 1] / 1e3, color=ac.C["accent2"],
                 lw=1.9, ls="--", label="CV filter")
        axg.plot(truth[ks:ke + 1, 0] / 1e3, truth[ks:ke + 1, 1] / 1e3,
                 color=ac.C["accent"], lw=3.2, alpha=0.5, label="turn")
        axg.set_xlabel("East (km)"); axg.set_ylabel("North (km)")
        axg.set_aspect("equal", adjustable="datalim")
        axg.legend(loc="best", fontsize=8)
        st.pyplot(fig)
    with c2:
        fig2, ax2 = ac.new_fig(figsize=(5.4, 4.4))
        ax2.axvspan(t[ks], t[ke], color=ac.C["accent"], alpha=0.10)
        ax2.plot(t[1:], nis_s[1:], color=ac.C["accent2"], lw=1.9,
                 label=f"NIS ({'window '+str(k) if k>1 else 'per step'})")
        ax2.axhline(2.0, color=ac.C["mid"], lw=1.2, ls="--", label="expected (2)")
        ax2.axhline(thr, color=ac.C["navy"], lw=1.3, ls=":", label="threshold")
        # mark detections
        td = t[detected]
        if td.size:
            ax2.plot(td, np.full(td.size, thr), "v", color=ac.C["navy"], ms=5,
                     label="detected")
        ax2.set_xlabel("time (s)"); ax2.set_ylabel("NIS")
        ax2.legend(loc="upper left", fontsize=8)
        st.pyplot(fig2)

    # detection delay
    delay_txt = "not detected"
    fired = np.where(detected[ks:])[0]
    if fired.size:
        delay_txt = f"{int(fired[0])} steps ({fired[0]*dt:.0f} s)"
    # false alarms: detections on the PRE-maneuver straight leg only (the elevated
    # NIS just after the turn is the filter still recovering, not a false alarm)
    fa = int(detected[3:ks].sum())
    m = st.columns(3)
    m[0].metric("Pre-turn mean NIS", f"{nis_s[5:ks].mean():.1f}")
    m[1].metric("Detection delay", delay_txt)
    m[2].metric("False alarms (before turn)", f"{fa}")
    st.caption("NIS averages the measurement dimension (2 here) when the model is "
               "correct; a sustained excursion above the threshold is the maneuver. "
               "The elevated NIS just after the turn is the filter recovering, not a "
               "false alarm.")

    with st.expander("Questions to answer (these match the student guide, Part A)",
                     expanded=True):
        st.markdown(
            "1. On the straight legs, is the NIS near its expected value of 2?\n"
            "2. Find the smallest **threshold** and **window** that flag the turn "
            "with no false alarms on the straight legs. Record the detection delay.\n"
            "3. Raise the **measurement noise**. What happens to the detector's "
            "reliability, and why?")

    st.info("The filter detects its own model mismatch: a maneuver makes the "
            "innovations biased and correlated, so NIS climbs above its chi-square "
            "expectation. A longer window is steadier but slower to fire.")

# ================================================================== Part B
else:
    st.subheader("Part B - the Interacting Multiple Model estimator")
    cb = st.columns(2)
    with cb[0]:
        imm_turn = st.slider("IMM turn-model rate (deg/s)", 1.0, 6.0, step=0.5,
                             key="imm_turn")
    with cb[1]:
        stay = st.slider("Stay-in-mode probability (TPM diagonal)", 0.80, 0.99,
                         step=0.01, key="stay")

    models = [dict(kind="cv", q=st.session_state["q_cv"]),
              dict(kind="ct", q=st.session_state["q_cv"], omega_deg=imm_turn)]
    off = 1.0 - stay
    tpm = np.array([[stay, off], [off, stay]])
    imm_est, mu, _ = ac.run_imm(meas, dt, meas_noise, models=models, tpm=tpm)
    cv_est, _ = ac.run_kf2d(meas, dt, meas_noise, model="cv",
                            q=st.session_state["q_cv"])

    c1, c2 = st.columns(2)
    with c1:
        fig, axg = ac.new_fig(figsize=(5.4, 4.4))
        axg.plot(meas[:, 0] / 1e3, meas[:, 1] / 1e3, ".", color=ac.C["muted"],
                 ms=3, alpha=0.5, label="measurements")
        axg.plot(truth[:, 0] / 1e3, truth[:, 1] / 1e3, color=ac.C["navy"],
                 lw=2.2, label="truth")
        axg.plot(cv_est[:, 0] / 1e3, cv_est[:, 1] / 1e3, color=ac.C["accent2"],
                 lw=1.6, ls="--", label="single CV")
        axg.plot(imm_est[:, 0] / 1e3, imm_est[:, 1] / 1e3, color=ac.C["accent"],
                 lw=1.9, label="IMM")
        axg.set_xlabel("East (km)"); axg.set_ylabel("North (km)")
        axg.set_aspect("equal", adjustable="datalim")
        axg.legend(loc="best", fontsize=8)
        st.pyplot(fig)
    with c2:
        fig2, ax2 = ac.new_fig(figsize=(5.4, 4.4))
        ax2.axvspan(t[ks], t[ke], color=ac.C["accent"], alpha=0.10)
        ax2.plot(t, mu[:, 0], color=ac.C["mid"], lw=2.2, label="constant velocity")
        ax2.plot(t, mu[:, 1], color=ac.C["accent"], lw=2.2, label="coordinated turn")
        ax2.set_xlabel("time (s)"); ax2.set_ylabel("mode probability")
        ax2.set_ylim(-0.03, 1.08)
        ax2.legend(loc="center left", fontsize=8)
        st.pyplot(fig2)

    # error-over-time comparison
    fig3, ax3 = ac.new_fig(figsize=(11.0, 3.0))
    ax3.axvspan(t[ks], t[ke], color=ac.C["accent"], alpha=0.10)
    ax3.plot(t, _err(cv_est), color=ac.C["accent2"], lw=2.0, label="single CV filter")
    ax3.plot(t, _err(imm_est), color=ac.C["accent"], lw=2.0, label="IMM (CV + turn)")
    ax3.text((t[ks] + t[ke]) / 2, ax3.get_ylim()[1] * 0.9, "maneuver",
             ha="center", fontsize=9, color=ac.C["navy"])
    ax3.set_xlabel("time (s)"); ax3.set_ylabel("position error (m)")
    ax3.legend(loc="upper left", fontsize=9)
    st.pyplot(fig3)

    def rmse_xy(est):
        return float(np.sqrt(np.mean(_err(est) ** 2)))

    def rmse_win(est):
        e = _err(est)[ks:ke + 1]
        return float(np.sqrt(np.mean(e ** 2)))

    m = st.columns(4)
    m[0].metric("Single CV: overall RMSE", f"{rmse_xy(cv_est):.0f} m")
    m[1].metric("IMM: overall RMSE", f"{rmse_xy(imm_est):.0f} m")
    m[2].metric("Single CV: turn-window RMSE", f"{rmse_win(cv_est):.0f} m")
    m[3].metric("IMM: turn-window RMSE", f"{rmse_win(imm_est):.0f} m")

    with st.expander("Questions to answer (these match the student guide, Part B)",
                     expanded=True):
        st.markdown(
            "1. When does the weight shift to the **turn model**, and how quickly "
            "does it shift back after the maneuver?\n"
            "2. Change the **turn-model rate** so it does or does not match the "
            "truth. How does a mismatched model affect the switch and the error?\n"
            "3. Adjust the **stay-in-mode probability**. How do stickier vs looser "
            "transitions trade response speed against jitter?\n"
            "4. Compare the IMM error against the single CV filter, overall and in "
            "the turn window.")

    st.info("The IMM needs no hard detection decision: the mode probabilities move "
            "continuously as the fit changes, giving the turn model's agility only "
            "during the turn and the CV model's smoothness on the straight legs.")
