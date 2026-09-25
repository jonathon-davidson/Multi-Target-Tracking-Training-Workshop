"""Module 12 companion: sensor registration.

Run from the project root:
    streamlit run exercises/ex12_registration.py

One target, two sensors. Inject a known azimuth bias into sensor 2 and watch the
fused picture split into two tracks once the bias-induced shift (R*dtheta) exceeds
the association gate. Then estimate the bias from N common-target observations
(least-squares) and remove it, re-merging the tracks. The estimate tightens as
sigma_theta / sqrt(N). Uses the shared, verified compute in appcommon.py.
"""
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

import appcommon as ac

ac.page_header("Sensor Registration",
               "inject an azimuth bias, split the target, estimate and re-merge",
               "Module 12")

C = ac.COURSE_COLORS

st.markdown(
    "**What you are doing:** two sensors track one target. A systematic **azimuth "
    "bias** on sensor 2 shifts its track cross-range by $R\\,\\Delta\\theta$. Once that "
    "shift exceeds the **association gate**, the fusion splits one target into **two** "
    "tracks. Estimate the bias from **common targets** (least-squares) and remove it to "
    "**re-merge** the tracks. The estimate tightens as $\\sigma_\\theta/\\sqrt{N}$.")

c1, c2, c3 = st.columns(3)
with c1:
    bias = st.slider("Injected azimuth bias (deg)", 0.0, 3.0, 1.5, 0.1)
with c2:
    N = st.slider("Common observations $N$ for the estimate", 1, 40, 10, 1)
with c3:
    r_km = st.slider("Target range (km)", 20.0, 100.0, 50.0, 5.0)
sth = st.slider("Sensor angular noise $\\sigma_\\theta$ (deg)", 0.1, 1.0, 0.3, 0.05)

reg = ac.simulate_registration(bias_deg=bias, N=N, r=r_km * 1000.0, sth_deg=sth)
tgt = reg["tgt"]


def ell(ax, mean, cov, color, label, ls="-", lw=2.2, k=6.0):
    vals, vecs = np.linalg.eigh((k ** 2) * cov)
    ang = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
    ax.add_patch(Ellipse(mean, 2 * np.sqrt(vals[0]), 2 * np.sqrt(vals[1]),
                 angle=ang, fill=False, edgecolor=color, lw=lw, ls=ls, zorder=5,
                 label=label))


col_a, col_b = st.columns(2)

# ---- before: biased / split ----
with col_a:
    fig1, ax1 = ac.new_fig(figsize=(5.2, 4.6))
    ell(ax1, reg["rep1"], reg["P1"], C["accent"], "sensor 1")
    ell(ax1, reg["rep2_biased"], reg["P2"], C["accent2"], "sensor 2 (biased)")
    ax1.annotate("", xy=reg["rep2_biased"], xytext=reg["rep1"],
                 arrowprops=dict(arrowstyle="<->", color=C["muted"], lw=1.4))
    ax1.scatter(*tgt, marker="*", s=180, color=C["navy"], zorder=7,
                edgecolor="white", linewidth=1.0, label="true target")
    span = max(2200, reg["shift"] * 0.9)
    ax1.set_xlim(tgt[0] - span, tgt[0] + reg["shift"] + span)
    ax1.set_ylim(tgt[1] - span, tgt[1] + span)
    ax1.set_aspect("equal", adjustable="box")
    ax1.set_xlabel("East (m)"); ax1.set_ylabel("North (m)")
    ax1.legend(loc="lower left", fontsize=8)
    ax1.set_title("Before: biased" + (" - SPLIT" if reg["split"] else " - gated as one"))
    st.pyplot(fig1)

# ---- after: corrected / merged ----
with col_b:
    fig2, ax2 = ac.new_fig(figsize=(5.2, 4.6))
    ell(ax2, reg["rep1"], reg["P1"], C["accent"], "sensor 1")
    ell(ax2, reg["rep2_corr"], reg["P2"], C["accent2"], "sensor 2 (corrected)",
        ls=(0, (4, 2)))
    ax2.scatter(*tgt, marker="*", s=180, color=C["navy"], zorder=7,
                edgecolor="white", linewidth=1.0, label="true target")
    ax2.set_xlim(tgt[0] - 2400, tgt[0] + 2400)
    ax2.set_ylim(tgt[1] - 2400, tgt[1] + 2400)
    ax2.set_aspect("equal", adjustable="box")
    ax2.set_xlabel("East (m)"); ax2.set_ylabel("North (m)")
    ax2.legend(loc="lower left", fontsize=8)
    ax2.set_title("After: corrected - re-merged")
    st.pyplot(fig2)

m1, m2, m3, m4 = st.columns(4)
m1.metric("Track shift $R\\,\\Delta\\theta$", f"{reg['shift']:.0f} m")
m2.metric("Association gate", f"{reg['gate']:.0f} m",
          delta="SPLIT" if reg["split"] else "one track", delta_color="off")
m3.metric("Estimated bias", f"{reg['bias_est']:.2f}°",
          delta=f"truth {bias:.2f}°", delta_color="off")
m4.metric("Residual after fix", f"{reg['residual']:.0f} m",
          delta="inside gate" if reg["residual"] < reg["gate"] else "still out",
          delta_color="off")

if reg["split"]:
    st.info(f"The {bias:.1f}° bias shifts sensor 2's track by {reg['shift']:.0f} m, more "
            f"than the {reg['gate']:.0f} m gate, so the fusion reports TWO tracks for one "
            f"target. The least-squares estimate ({reg['bias_est']:.2f}°) removes it and "
            f"the residual ({reg['residual']:.0f} m) falls inside the gate, re-merging "
            "the tracks.")
else:
    st.success(f"At {bias:.1f}° the shift ({reg['shift']:.0f} m) is within the "
               f"{reg['gate']:.0f} m gate, so the target stays one track, though the "
               "fused estimate is biased. Raise the bias (or the range) until it splits.")

st.caption("Deliverable: the before/after fused-track plot (split, then re-merged) and "
           "the estimated-bias-versus-truth value, with a short paragraph on why "
           "registration must drive the residual shift well inside the association gate.")
