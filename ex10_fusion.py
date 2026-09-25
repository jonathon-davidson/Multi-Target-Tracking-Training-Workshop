"""Module 10 companion: correlated track fusion.

Run from the project root:
    streamlit run exercises/ex10_fusion.py

Two local tracks of the same target share a tunable fraction of their error. Fuse
them (a) naively, assuming independence, and (b) with covariance intersection, and
compare the reported covariance against the true fused error. The naive fusion is
over-confident when the tracks are correlated (NEES climbs above 2); covariance
intersection stays consistent for any unknown correlation (NEES at or below 2).
Uses the shared, verified fusion compute in appcommon.py.
"""
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

import appcommon as ac

ac.page_header("Multi-Sensor Fusion",
               "correlated track fusion: naive vs covariance intersection",
               "Module 10")

C = ac.COURSE_COLORS

st.markdown(
    "**What you are doing:** two sensors each track the **same** target and report a "
    "local track (position + covariance). Their errors are **correlated** because they "
    "share error sources (common target motion, a shared prior). Fuse them **naively** "
    "(assuming independence) and with **covariance intersection (CI)**, and compare each "
    "to the **true** fused error. **NEES** should equal 2 for a consistent 2-D estimate: "
    "above 2 is over-confident, at or below 2 is safe.")

c1, c2, c3 = st.columns(3)
with c1:
    rho = st.slider("Shared-error correlation $\\rho$", 0.0, 0.95, 0.7, 0.05)
with c2:
    s1x = st.slider("Track 1 std, x (m)", 1.0, 6.0, 2.0, 0.5)
    s1y = st.slider("Track 1 std, y (m)", 1.0, 6.0, 3.0, 0.5)
with c3:
    s2x = st.slider("Track 2 std, x (m)", 1.0, 6.0, 3.0, 0.5)
    s2y = st.slider("Track 2 std, y (m)", 1.0, 6.0, 2.0, 0.5)

P1 = np.array([[s1x ** 2, 0.0], [0.0, s1y ** 2]])
P2 = np.array([[s2x ** 2, 0.0], [0.0, s2y ** 2]])

Pn = ac.naive_fuse(P1, P2)
Pci_opt, w_opt = ac.ci_fuse(P1, P2)
manual = st.checkbox("Set the CI weight $\\omega$ by hand", value=False)
if manual:
    w = st.slider("CI weight $\\omega$ (1.0 = trust track 1 only)",
                  0.0, 1.0, float(round(w_opt, 2)), 0.02)
    Pci = np.linalg.inv(w * np.linalg.inv(P1) + (1 - w) * np.linalg.inv(P2))
else:
    w, Pci = w_opt, Pci_opt

P12 = ac.crosscov(P1, P2, rho)
Ptrue = ac.bc_fuse(P1, P2, P12)
nees_naive = ac.sim_fused_nees(P1, P2, rho, Pn, weights=(1.0, 1.0))
nees_ci = ac.sim_fused_nees(P1, P2, rho, Pci, weights=(w, 1.0 - w))

# --- Monte-Carlo cloud of the ACTUAL naive-fused estimate error ---
rng = np.random.default_rng(3)
L = np.linalg.cholesky(np.block([[P1, P12], [P12.T, P2]]))
z = L @ rng.standard_normal((4, 4000))
e1, e2 = z[:2], z[2:]
I1, I2 = np.linalg.inv(P1), np.linalg.inv(P2)
cloud = Pn @ (I1 @ e1 + I2 @ e2)

fig, ax = ac.new_fig(figsize=(7.4, 5.0))
ax.scatter(cloud[0], cloud[1], s=4, color=C["muted"], alpha=0.18, zorder=1,
           label="naive-fused error cloud")


def ell(cov, color, label, ls="-"):
    vals, vecs = np.linalg.eigh(cov)
    ang = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
    ax.add_patch(Ellipse((0, 0), 2 * 2 * np.sqrt(vals[0]),
                 2 * 2 * np.sqrt(vals[1]), angle=ang, fill=False,
                 edgecolor=color, lw=2.4, ls=ls, zorder=4, label=label))


ell(P1, C["mid"], "track 1 input", ls=(0, (2, 2)))
ell(P2, C["navy"], "track 2 input", ls=(0, (2, 2)))
ell(Ptrue, C["navy"], "true fused error", ls=(0, (5, 2)))
ell(Pn, C["accent2"], f"naive (NEES {nees_naive:.1f})")
ell(Pci, C["accent"], f"CI, $\\omega$={w:.2f} (NEES {nees_ci:.1f})")
ax.scatter(0, 0, marker="*", s=170, color=C["navy"], zorder=6,
           edgecolor="white", linewidth=1.0, label="true position")
lim = 2.4 * np.sqrt(np.trace(Pci))
ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
ax.set_aspect("equal", adjustable="box")
ax.set_xlabel("x error (m)"); ax.set_ylabel("y error (m)")
ax.legend(loc="upper right", fontsize=8)
ax.set_title("Naive vs covariance-intersection fusion (2$\\sigma$ ellipses)")
st.pyplot(fig)

m1, m2, m3 = st.columns(3)
m1.metric("Naive NEES", f"{nees_naive:.1f}",
          delta="over-confident" if nees_naive > 2.3 else "ok",
          delta_color="off")
m2.metric("CI NEES", f"{nees_ci:.1f}", delta="consistent (<=2)",
          delta_color="off")
m3.metric("Fused-cov size (trace)", f"naive {np.trace(Pn):.1f} / CI {np.trace(Pci):.1f}",
          delta="CI is conservative", delta_color="off")

if nees_naive > 2.3:
    st.info("The naive fusion is over-confident here: it assumes the two tracks are "
            "independent, but they share an error, so the true fused error is larger "
            "than the naive covariance claims (NEES above 2). Covariance intersection "
            "stays consistent without knowing the correlation, at the cost of a larger, "
            "conservative covariance.")
else:
    st.success("At this correlation the naive fusion is near-consistent, and it is the "
               "tightest option when the tracks really are independent. Raise $\\rho$ to "
               "watch it become over-confident while CI stays safe.")

st.caption("Deliverable: this naive-vs-CI ellipse comparison with the two NEES values, "
           "and a short paragraph on what makes the two local tracks correlated and why "
           "covariance intersection is the safe default in a distributed fusion system.")
