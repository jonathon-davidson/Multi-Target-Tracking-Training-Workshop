"""Module 11 companion: multi-radar fusion & angle-only triangulation.

Run from the project root:
    streamlit run exercises/ex11_fusion.py

Two parts, chosen with the selector at the top:
  A. Multi-radar fusion - fuse two radar tracks and watch the fused covariance
     shrink with the look-angle separation (the gain is geometric).
  B. Triangulation & ghosts - cross-fix two angle-only sensors, watch the fix
     ellipse stretch as the cut angle drops, reveal ghost targets with a second
     target, and resolve them with a third sensor.

Uses the shared, verified compute in appcommon.py so the app and the course
figures agree.
"""
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

import appcommon as ac

ac.page_header("Multiple Radars & Angle-Only Sensors",
               "homogeneous fusion: radar geometry and bearings-only triangulation",
               "Module 11")

C = ac.COURSE_COLORS


def ellipse(ax, mean, cov, color, label, nsig=2.0, lw=2.4):
    vals, vecs = np.linalg.eigh(cov)
    ang = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
    ax.add_patch(Ellipse(mean, 2 * nsig * np.sqrt(vals[0]),
                 2 * nsig * np.sqrt(vals[1]), angle=ang, fill=False,
                 edgecolor=color, lw=lw, zorder=5, label=label))


part = st.radio("Choose a part",
                ["A - Multi-radar fusion & geometry",
                 "B - Triangulation & ghost targets"],
                horizontal=True)

# =====================================================================
# PART A - multi-radar fusion
# =====================================================================
if part.startswith("A"):
    st.markdown(
        "**What you are doing:** two radars track the **same** target. Each measures "
        "**range** well but **angle** only to $\\sigma_\\theta$, so its error ellipse is "
        "thin in range and fat in cross-range ($r\\,\\sigma_\\theta$). Fusing two radars "
        "from **different bearings** crosses their good range axes. Watch the fused "
        "ellipse collapse when the looks are well separated, and barely shrink when they "
        "are near-collinear.")

    c1, c2, c3 = st.columns(3)
    with c1:
        r_km = st.slider("Target range (km)", 10.0, 100.0, 50.0, 5.0)
        b1 = st.slider("Radar 1 bearing (deg)", 0.0, 180.0, 20.0, 5.0)
    with c2:
        b2 = st.slider("Radar 2 bearing (deg)", 0.0, 180.0, 110.0, 5.0)
        sth = st.slider("Angular noise $\\sigma_\\theta$ (deg)", 0.1, 1.0, 0.3, 0.05)
    with c3:
        sr = st.slider("Range noise $\\sigma_r$ (m)", 10.0, 100.0, 30.0, 5.0)
        method = st.selectbox("Fusion", ["naive (independent)",
                                         "covariance intersection"])

    r = r_km * 1000.0
    P1 = ac.radar_cov(r, b1, sr, sth)
    P2 = ac.radar_cov(r, b2, sr, sth)
    m = "naive" if method.startswith("naive") else "ci"
    Pf = ac.fuse_radars([P1, P2], m)

    fig, ax = ac.new_fig(figsize=(6.6, 5.2))
    k = 3.0
    ellipse(ax, (0, 0), (k ** 2) * P1, C["mid"],
            f"radar 1 ({np.sqrt(np.trace(P1)):.0f} m)")
    ellipse(ax, (0, 0), (k ** 2) * P2, C["accent2"],
            f"radar 2 ({np.sqrt(np.trace(P2)):.0f} m)")
    ellipse(ax, (0, 0), (k ** 2) * Pf, C["accent"],
            f"fused ({np.sqrt(np.trace(Pf)):.0f} m)", lw=2.8)
    for b, col in [(b1, C["mid"]), (b2, C["accent2"])]:
        d = np.array([np.sin(np.deg2rad(b)), np.cos(np.deg2rad(b))])
        ax.annotate("", xy=-1500 * d, xytext=-3400 * d,
                    arrowprops=dict(arrowstyle="-|>", color=col, lw=1.6))
    ax.scatter(0, 0, marker="*", s=170, color=C["navy"], zorder=7,
               edgecolor="white", linewidth=1.0)
    lim = 3.4 * k * np.sqrt(max(np.linalg.eigvalsh(P1).max(),
                                np.linalg.eigvalsh(P2).max()))
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("East offset (m)"); ax.set_ylabel("North offset (m)")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_title("Two-radar fusion (ellipses at 3$\\times$ scale)")
    st.pyplot(fig)

    sep = abs(b2 - b1)
    m1, m2, m3 = st.columns(3)
    m1.metric("Single-radar rms", f"{np.sqrt(np.trace(P1)):.0f} m")
    m2.metric("Fused rms", f"{np.sqrt(np.trace(Pf)):.0f} m",
              delta=f"{np.sqrt(np.trace(P1)) / np.sqrt(np.trace(Pf)):.1f}x better",
              delta_color="off")
    m3.metric("Look-angle separation", f"{sep:.0f} deg",
              delta="good" if sep > 45 else "weak", delta_color="off")
    if sep < 25:
        st.info("The two looks are nearly collinear, so the radars' fat cross-range axes "
                "point the same way and fusion barely helps. Separate the bearings toward "
                "ninety degrees to cross the good range axes and collapse the ellipse.")
    else:
        st.success("Well-separated looks: the two precise range axes cross, so the fused "
                   "ellipse is far smaller than either radar alone. This geometric gain, "
                   "not the sensor count, is the point.")
    if m == "ci":
        st.caption("Covariance intersection is looser than naive fusion but stays "
                   "consistent for the unknown correlation between two radar tracks of "
                   "the same target (Module 10). Naive fusion is tighter but optimistic.")

# =====================================================================
# PART B - triangulation & ghosts
# =====================================================================
else:
    st.markdown(
        "**What you are doing:** angle-only sensors measure **bearing, not range**, so a "
        "single one only gives a line of position. Two or more **cross-fix** "
        "(triangulate) to a point, with an error ellipse set by the **cut angle**. With a "
        "**second target**, the four bearing crossings include two **ghost** targets, "
        "resolved by a **third sensor**.")

    c1, c2, c3 = st.columns(3)
    with c1:
        base_km = st.slider("Sensor baseline (km)", 5.0, 100.0, 60.0, 5.0)
        sth = st.slider("Bearing noise $\\sigma_\\theta$ (deg)", 0.2, 3.0, 1.0, 0.1)
    with c2:
        tx = st.slider("Target 1 East (km)", -30.0, 30.0, -6.0, 1.0)
        ty = st.slider("Target 1 North (km)", 15.0, 80.0, 45.0, 1.0)
    with c3:
        two = st.checkbox("Add a second target (ghosts)", value=True)
        third = st.checkbox("Add a third sensor (resolve)", value=True)

    s1 = np.array([-base_km * 500.0, 0.0])   # km/2 in metres
    s2 = np.array([base_km * 500.0, 0.0])
    s3 = np.array([0.0, -5000.0])
    T = [np.array([tx * 1000.0, ty * 1000.0])]
    if two:
        T.append(np.array([10000.0, 38000.0]))

    fig, ax = ac.new_fig(figsize=(7.0, 5.4))

    def ray(s, tgt, col, lw=1.5, alpha=1.0, extend=1.25):
        b = ac.bearing(s, tgt)
        d = np.array([np.sin(b), np.cos(b)])
        end = s + extend * np.hypot(*(tgt - s)) * d
        ax.plot([s[0], end[0]], [s[1], end[1]], "-", color=col, lw=lw,
                alpha=alpha, zorder=2)

    for t in T:
        ray(s1, t, C["mid"]); ray(s2, t, C["accent2"])

    # fix ellipse(s) for the real target(s)
    for t in T:
        P = ac.triangulate_cov([s1, s2], t, sth)
        ellipse(ax, t, P, C["accent"], None, nsig=2.0, lw=2.2)

    cand = ac.enumerate_intersections(s1, s2, T)
    if third:
        ac.resolve_with_third(s3, cand, T)
        for t in T:
            ray(s3, t, C["navy"], lw=1.4, alpha=0.8, extend=1.15)
    for c in cand:
        if c["real"]:
            ax.scatter(*c["pos"], s=170, marker="*", color=C["navy"], zorder=6,
                       edgecolor="white", linewidth=1.0)
        else:
            ax.scatter(*c["pos"], s=110, marker="X", color=C["muted"], zorder=6)

    ax.scatter(*s1, marker="^", s=90, color=C["mid"], zorder=7, label="sensor 1")
    ax.scatter(*s2, marker="^", s=90, color=C["accent2"], zorder=7, label="sensor 2")
    if third:
        ax.scatter(*s3, marker="^", s=90, color=C["navy"], zorder=7, label="sensor 3")
    ax.scatter([], [], marker="*", color=C["navy"], s=120, label="real fix")
    if two:
        ax.scatter([], [], marker="X", color=C["muted"], s=100, label="ghost")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("East (m)"); ax.set_ylabel("North (m)")
    ax.legend(loc="lower right", fontsize=8, ncol=2)
    cut = ac.cut_angle_deg(s1, s2, T[0])
    P0 = ac.triangulate_cov([s1, s2], T[0], sth)
    ax.set_title(f"Cross-fix: target 1 cut angle {cut:.0f}°, "
                 f"rms {np.sqrt(np.trace(P0)):.0f} m")
    st.pyplot(fig)

    m1, m2, m3 = st.columns(3)
    m1.metric("Target 1 cut angle", f"{cut:.0f} deg",
              delta="strong" if cut > 60 else "weak", delta_color="off")
    m2.metric("Target 1 fix rms", f"{np.sqrt(np.trace(P0)):.0f} m")
    n_ghost = sum(1 for c in cand if not c["real"])
    m3.metric("Ghost crossings", f"{n_ghost}",
              delta="resolved" if (third and two) else ("present" if two else "none"),
              delta_color="off")

    if two and not third:
        st.info("The crosses are ghost targets: real intersections of bearings that "
                "belong to DIFFERENT targets. On two sensors alone they are "
                "indistinguishable from real fixes. Turn on the third sensor to resolve "
                "them.")
    elif two and third:
        st.success("The third sensor's bearings (navy) pass through the two real targets "
                   "and miss the ghosts, so the ambiguity is resolved. This is the "
                   "data association of Modules 6 to 8, applied across sensors.")
    else:
        st.success("One target, one clean fix. The ellipse stretches as the cut angle "
                   "falls (shrink the baseline or move the target aside) - that is GDOP.")

st.caption("Deliverables: (a) the radar-fusion covariance plot at a good and a poor "
           "geometry with one line on why look-angle separation matters; (b) the "
           "triangulation/ghost plot with and without the third sensor, identifying the "
           "ghosts and how the third sensor rejects them.")
