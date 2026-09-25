"""Module 7 companion: track management (M-of-N) and range-rate (Doppler).

Run from the project root:
    streamlit run exercises/ex07_track_mgmt.py

Part A tunes an M-of-N initiation rule against a false-track budget. Part B
switches a range-rate measurement on and off in an EKF and shows which velocity
component it improves. Both use the shared, verified compute in appcommon.py, so
the app agrees with the course figures.
"""
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

import appcommon as ac

ac.page_header("Track Management & Range-Rate",
               "M-of-N initiation and the Doppler benefit",
               "Module 7")

part = st.radio("Choose a part", ["Part A - M-of-N initiation",
                                  "Part B - range-rate in the EKF"],
                horizontal=True)

C = ac.COURSE_COLORS
RULES = [(2, 2), (2, 3), (3, 4), (3, 5), (4, 5), (5, 6), (4, 6)]

# =====================================================================
if part.startswith("Part A"):
    st.markdown(
        "**What you are doing:** after an initiating detection, a tentative track "
        "is confirmed once it gets **M detections in a window of N scans**. The "
        "same binomial evaluated at the target's $P_D$ gives the chance a real "
        "target confirms; at the clutter-in-gate rate $P_{FA}$ it gives the chance "
        "clutter confirms into a **false track**. Find a rule that meets your "
        "false-track budget without dropping too many real targets.")

    c1, c2, c3 = st.columns(3)
    with c1:
        pd = st.slider("Target detection prob $P_D$", 0.50, 0.99, 0.90, 0.01)
    with c2:
        pfa = st.slider("Clutter-in-gate prob $P_{FA}$", 0.01, 0.30, 0.05, 0.01)
    with c3:
        budget = st.select_slider(
            "False-track budget",
            options=[1e-1, 1e-2, 1e-3, 1e-4, 1e-5],
            value=1e-3, format_func=lambda v: f"{v:.0e}")
    floor = st.slider("Target-confirmation floor", 0.80, 0.999, 0.95, 0.005)
    M = st.slider("M (required detections)", 1, 6, 3)
    N = st.slider("N (window length)", M, 8, 5)

    labels, pc, pf = ac.mofn_curve(RULES, pd, pfa)
    selc = ac.m_of_n(M, N, pd)
    self_ = ac.m_of_n(M, N, pfa)

    fig, ax = ac.new_fig(figsize=(7.4, 4.2))
    ax.scatter(pf, pc, s=70, color=C["accent"], edgecolor="white",
               linewidth=1.3, zorder=3)
    for l, x, y in zip(labels, pf, pc):
        ax.annotate(l, (x, y), (7, 4), textcoords="offset points",
                    fontsize=9, color=C["navy"])
    ax.scatter([self_], [selc], s=170, marker="*", color=C["accent2"],
               edgecolor="white", linewidth=1.2, zorder=5,
               label=f"your rule {M}/{N}")
    ax.axvline(budget, color=C["muted"], ls="--", lw=1.2)
    ax.axhline(floor, color=C["muted"], ls=":", lw=1.2)
    ax.axvspan(ax.get_xlim()[0], budget, ymin=0, ymax=1, color=C["accent"],
               alpha=0.05)
    ax.set_xscale("log")
    ax.set_xlabel("false-track probability (clutter confirms)")
    ax.set_ylabel("target-confirm probability")
    ax.set_title("M-of-N detection / false-track trade")
    ax.legend(loc="lower right")
    st.pyplot(fig)

    ok_budget = self_ <= budget
    ok_floor = selc >= floor
    st.markdown(
        f"**Rule {M}/{N}:** confirms **{selc:.3f}** of real targets, "
        f"false-track probability **{self_:.2e}**.")
    cA, cB = st.columns(2)
    cA.metric("Meets false-track budget?", "yes" if ok_budget else "no",
              delta=f"budget {budget:.0e}", delta_color="off")
    cB.metric("Meets confirmation floor?", "yes" if ok_floor else "no",
              delta=f"floor {floor:.2f}", delta_color="off")
    if ok_budget and ok_floor:
        st.success("This rule clears both the budget and the floor.")
    else:
        st.info("Adjust M, N (or accept a looser budget/floor) to clear both "
                "dashed lines: below the vertical, above the horizontal.")

    st.caption("Deliverable: this trade-off plot with your chosen (M, N) marked, "
               "and a sentence justifying it against the budget and floor.")

# =====================================================================
else:
    st.markdown(
        "**What you are doing:** an EKF tracks a target from noisy **position** "
        "measurements. Switch the measured **range-rate** (Doppler) on and watch "
        "the error drop. The velocity error is split into the component **along** "
        "the line of sight (radial) and **across** it (tangential): range-rate "
        "observes only the radial part.")

    c1, c2, c3 = st.columns(3)
    with c1:
        geom = st.selectbox("Geometry", ["Closing", "Crossing"])
    with c2:
        sig_rr = st.slider("Range-rate noise (m/s)", 0.5, 12.0, 2.0, 0.5)
    with c3:
        sig_pos = st.slider("Position noise (m)", 10.0, 100.0, 40.0, 5.0)
    n_mc = st.slider("Monte-Carlo runs", 20, 300, 120, 20)

    kw = dict(sig_pos=sig_pos, sig_rr=sig_rr)
    if geom == "Crossing":
        kw.update(start=(3000.0, -1400.0), vel=(0.0, 70.0))

    # Monte-Carlo error curves + radial/tangential split
    pe0 = pe1 = None
    er0 = et0 = er1 = et1 = 0.0
    for s in range(n_mc):
        t, X, Zp, Zr = ac.simulate_rr_track(seed=s, **kw)
        X0 = ac.run_ekf_rr(t, Zp, Zr, use_rr=False, sig_pos=sig_pos, sig_rr=sig_rr)
        X1 = ac.run_ekf_rr(t, Zp, Zr, use_rr=True, sig_pos=sig_pos, sig_rr=sig_rr)
        p0, _ = ac.track_errors(X, X0)
        p1, _ = ac.track_errors(X, X1)
        pe0 = p0 if pe0 is None else pe0 + p0
        pe1 = p1 if pe1 is None else pe1 + p1
        a0, b0 = ac.radtan_velocity_error(X, X0)
        a1, b1 = ac.radtan_velocity_error(X, X1)
        w = slice(10, None)
        er0 += a0[w].mean(); et0 += b0[w].mean()
        er1 += a1[w].mean(); et1 += b1[w].mean()
    pe0 /= n_mc; pe1 /= n_mc
    er0, et0, er1, et1 = [v / n_mc for v in (er0, et0, er1, et1)]

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(9.2, 3.6),
                                   gridspec_kw={"width_ratios": [1.35, 1.0]})
    ac.apply_style()
    axL.plot(t, pe0, color=C["muted"], lw=1.8, label="position only")
    axL.plot(t, pe1, color=C["accent"], lw=2.0, label="+ range-rate")
    axL.set_xlabel("time (s)"); axL.set_ylabel("position error (m)")
    axL.legend(loc="upper right"); axL.set_title(f"Track error, {geom.lower()} target")
    axL.grid(True, alpha=0.4)
    x = np.arange(2); wd = 0.36
    axR.bar(x - wd / 2, [er0, et0], wd, color=C["muted"], label="position only")
    axR.bar(x + wd / 2, [er1, et1], wd, color=C["accent"], label="+ range-rate")
    axR.set_xticks(x); axR.set_xticklabels(["radial\n(along LOS)",
                                            "tangential\n(cross-range)"], fontsize=9)
    axR.set_ylabel("velocity error (m/s)")
    axR.set_ylim(0, max(er0, et0, er1, et1) * 1.28)
    axR.legend(loc="upper center", ncol=2, fontsize=8, columnspacing=1.0)
    axR.set_title("Which velocity component improves")
    fig.tight_layout()
    st.pyplot(fig)

    pos_pct = 100 * (1 - pe1[10:].mean() / pe0[10:].mean())
    vr_pct = 100 * (1 - er1 / er0)
    vt_pct = 100 * (1 - et1 / max(et0, 1e-9))
    m1, m2, m3 = st.columns(3)
    m1.metric("Position error", f"{pe1[10:].mean():.1f} m",
              delta=f"-{pos_pct:.0f}% vs no range-rate")
    m2.metric("Radial velocity error", f"{er1:.2f} m/s",
              delta=f"-{vr_pct:.0f}%")
    m3.metric("Tangential velocity error", f"{et1:.2f} m/s",
              delta=f"{-vt_pct:+.0f}%", delta_color="off")
    st.caption("Deliverable: this before/after error plot with a short paragraph "
               "on why range-rate improves the radial velocity and not the "
               "tangential.")
