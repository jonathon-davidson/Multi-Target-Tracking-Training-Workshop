"""Module 9 companion: debiased converted measurements.

Run from the project root:
    streamlit run exercises/ex09_conversion.py

Draw a Monte-Carlo cloud of a target's converted position from noisy polar
measurements, overlay the naive and debiased converted-measurement covariance
ellipses, and report the NEES for each. The naive conversion is biased and
over-confident; the debiased one is consistent (NEES = 2). Uses the shared,
verified compute in appcommon.py.
"""
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

import appcommon as ac

ac.page_header("Coordinate Conversions",
               "debiased, consistent converted measurements",
               "Module 9")

C = ac.COURSE_COLORS

st.markdown(
    "**What you are doing:** a sensor measures a target in **polar** (range, "
    "bearing) with noise. Converting to Cartesian is nonlinear, so the naive "
    "converted point is **biased** toward the sensor and its Jacobian covariance "
    "is **over-confident**. Compare it to the **debiased** converted measurement "
    "and its exact covariance. **NEES** should equal 2 for a consistent 2-D "
    "measurement.")

c1, c2, c3 = st.columns(3)
with c1:
    r_km = st.slider("Range (km)", 5.0, 100.0, 20.0, 1.0)
with c2:
    sb_deg = st.slider("Angular noise $\\sigma_\\theta$ (deg)", 0.5, 10.0, 4.0, 0.5)
with c3:
    sr = st.slider("Range noise $\\sigma_r$ (m)", 10.0, 200.0, 50.0, 5.0)
n = st.slider("Monte-Carlo samples", 1000, 20000, 6000, 1000)

r = r_km * 1000.0
b = np.deg2rad(35.0)
sb = np.deg2rad(sb_deg)
rng = np.random.default_rng(9)
truth = ac.polar_to_cart(r, b)
rr = r + rng.normal(0, sr, n)
bb = b + rng.normal(0, sb, n)
xc = np.vstack([rr * np.cos(bb), rr * np.sin(bb)])
xu = np.exp(sb ** 2 / 2) * xc

nees_naive = ac.nees(xc, truth, ac.naive_cov(r, b, sr, sb))
nees_deb = ac.nees(xu, truth, ac.unbiased_cov(r, b, sr, sb))
bias = float(np.linalg.norm(xc.mean(1) - truth))

# local radial / cross-range frame about truth
cA, sA = np.cos(b), np.sin(b)
Rot = np.array([[cA, sA], [-sA, cA]])
cloud = Rot @ (xc - truth[:, None])
naive_mean = Rot @ (xc.mean(1) - truth)
Rn = Rot @ ac.naive_cov(r, b, sr, sb) @ Rot.T
Ru = Rot @ ac.unbiased_cov(r, b, sr, sb) @ Rot.T

fig, ax = ac.new_fig(figsize=(7.6, 4.4))
ax.scatter(cloud[0], cloud[1], s=4, color=C["muted"], alpha=0.2, zorder=1)


def ell(mean, cov, color, label):
    vals, vecs = np.linalg.eigh(cov)
    ang = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
    ax.add_patch(Ellipse(mean, 4 * np.sqrt(vals[0]), 4 * np.sqrt(vals[1]),
                 angle=ang, fill=False, edgecolor=color, lw=2.4, zorder=4,
                 label=label))


ell(naive_mean, Rn, C["accent2"], f"naive (NEES {nees_naive:.1f})")
ell([0, 0], Ru, C["accent"], f"debiased (NEES {nees_deb:.1f})")
ax.scatter(0, 0, marker="*", s=180, color=C["navy"], zorder=6,
           edgecolor="white", linewidth=1.0, label="true position")
ax.scatter(*naive_mean, marker="X", s=80, color=C["accent2"], zorder=6)
lim_v = 3.2 * r * sb
ax.set_xlim(-max(300, 6 * sr) - bias, max(200, 4 * sr))
ax.set_ylim(-lim_v, lim_v)
ax.set_xlabel("radial offset (m)   [toward sensor $\\leftarrow$]")
ax.set_ylabel("cross-range offset (m)")
ax.legend(loc="upper right", fontsize=9)
ax.set_title("Naive vs debiased converted measurement")
st.pyplot(fig)

m1, m2, m3 = st.columns(3)
m1.metric("Naive NEES", f"{nees_naive:.1f}",
          delta="over-confident" if nees_naive > 3 else "ok", delta_color="off")
m2.metric("Debiased NEES", f"{nees_deb:.1f}", delta="consistent (=2)",
          delta_color="off")
m3.metric("Naive mean bias", f"{bias:.0f} m", delta="toward sensor",
          delta_color="off")

if nees_naive > 3:
    st.info("The naive converted covariance is inconsistent here: its NEES is well "
            "above 2, so a filter would trust this measurement far more than it "
            "deserves. The debiasing removes the bias and restores NEES to 2.")
else:
    st.success("At this angular noise the nonlinearity is mild, so even the naive "
               "conversion is near-consistent. Raise $\\sigma_\\theta$ to see it break.")

st.caption("Deliverable: this naive-vs-debiased ellipse comparison with the two "
           "NEES values, and a short paragraph on why the naive converted covariance "
           "is inconsistent and what the debiasing fixes.")
