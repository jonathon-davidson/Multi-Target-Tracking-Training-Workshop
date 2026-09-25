"""Shared helpers for the Multi-Target Tracking Workshop Streamlit companion apps.

Every ex*.py app imports from this module. Streamlit inserts the running
script's directory (exercises/) onto sys.path, so `import appcommon` resolves
when you run, from the project root:

    streamlit run exercises/ex01_motion_models.py

Dependencies: streamlit, numpy, matplotlib (the standard scientific stack in
the `radar` conda/mamba environment). No scipy is required.

The compute functions here mirror the ones behind the course figures
(figures/mod*.py), so the interactive apps and the printed slides/guides agree.
"""
from __future__ import annotations

import math

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams


# ------------------------------------------------------------------ style
COURSE_COLORS = {
    "navy": "#0B1C2E",
    "mid": "#1F3864",
    "accent": "#1A9B8E",
    "accent2": "#E07A3D",
    "muted": "#5A6A7A",
    "grid": "#D0D7DE",
    "bg": "#FFFFFF",
}
C = COURSE_COLORS


def apply_style() -> None:
    """Match the course figure look (see figures/common.py)."""
    rcParams.update({
        "figure.facecolor": C["bg"],
        "axes.facecolor": C["bg"],
        "axes.edgecolor": C["muted"],
        "axes.labelcolor": C["navy"],
        "axes.titlecolor": C["navy"],
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "xtick.color": C["muted"],
        "ytick.color": C["muted"],
        "grid.color": C["grid"],
        "grid.linewidth": 0.7,
        "grid.alpha": 0.85,
        "legend.frameon": False,
        "font.family": "sans-serif",
        "lines.linewidth": 1.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def new_fig(figsize=(6.6, 4.2)):
    apply_style()
    fig, ax = plt.subplots(figsize=figsize)
    ax.grid(True, alpha=0.4)
    return fig, ax


def page_header(title: str, subtitle: str, module: str) -> None:
    """Standard app title block. Import streamlit lazily so this file stays
    importable for unit tests without a Streamlit runtime."""
    import streamlit as st
    st.set_page_config(page_title=title, layout="wide")
    st.title(title)
    st.caption(f"{module} companion  ·  {subtitle}")


# ------------------------------------------------------------------ motion models
def propagate_cv(steps, dt, speed, heading_deg=0.0, q=0.0, seed=0):
    """Constant-velocity target with process noise q (accel std, m/s^2).

    Returns arrays (x, y) of length steps+1 in metres. A small random
    acceleration ~ N(0, q^2) is injected in each axis every step; that is the
    process noise the tracker's Q term budgets for.
    """
    rng = np.random.default_rng(seed)
    h = np.deg2rad(heading_deg)
    v = np.array([speed * np.cos(h), speed * np.sin(h)], dtype=float)
    p = np.zeros(2)
    xs, ys = [p[0]], [p[1]]
    for _ in range(steps):
        a = rng.normal(0.0, q, size=2) if q > 0 else np.zeros(2)
        v = v + a * dt
        p = p + v * dt
        xs.append(p[0]); ys.append(p[1])
    return np.array(xs), np.array(ys)


def propagate_ct(steps, dt, speed, omega_deg=3.0, heading_deg=0.0, q=0.0, seed=0):
    """Coordinated-turn target: constant speed, constant turn rate omega
    (deg/s), with the same process-noise model as propagate_cv.
    """
    rng = np.random.default_rng(seed)
    h = np.deg2rad(heading_deg)
    w = np.deg2rad(omega_deg)
    v = np.array([speed * np.cos(h), speed * np.sin(h)], dtype=float)
    p = np.zeros(2)
    xs, ys = [p[0]], [p[1]]
    cos_w, sin_w = np.cos(w * dt), np.sin(w * dt)
    R = np.array([[cos_w, -sin_w], [sin_w, cos_w]])   # rotate velocity each step
    for _ in range(steps):
        v = R @ v
        a = rng.normal(0.0, q, size=2) if q > 0 else np.zeros(2)
        v = v + a * dt
        p = p + v * dt
        xs.append(p[0]); ys.append(p[1])
    return np.array(xs), np.array(ys)


# ------------------------------------------------------------------ linear filters (Module 2)
def cv_matrices(dt, q):
    """State-space matrices for a 1-D constant-velocity model, state [pos, vel].

    Q is the discrete white-noise-acceleration process-noise covariance; q is the
    acceleration standard deviation (m/s^2), the "how much the target can depart
    from constant velocity" knob.
    """
    F = np.array([[1.0, dt], [0.0, 1.0]])
    H = np.array([[1.0, 0.0]])
    Q = (q ** 2) * np.array([[dt ** 4 / 4, dt ** 3 / 2],
                             [dt ** 3 / 2, dt ** 2]])
    return F, H, Q


def simulate_cv_track(steps, dt, r, speed=200.0, x0=0.0, maneuver=None, seed=0):
    """Truth path + noisy 1-D position measurements.

    maneuver=None gives a pure constant-velocity target. Passing
    maneuver=dict(start=0.4, stop=0.5, accel=25.0) applies a constant acceleration
    (m/s^2) only in the window [start, stop) of the track, i.e. a brief velocity
    change and then constant velocity again. This finite maneuver is what gives
    the Q tuning trade-off a real sweet spot (low Q spikes at the maneuver, high Q
    is jittery throughout). Omit "stop" to accelerate through to the end.
    Returns (t, truth_pos, truth_vel, meas), each length steps+1.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(steps + 1) * dt
    pos = np.zeros(steps + 1)
    vel = np.zeros(steps + 1)
    p, v = float(x0), float(speed)
    pos[0], vel[0] = p, v
    ks = int((maneuver or {}).get("start", 1.0) * steps) if maneuver else steps + 2
    ke = int((maneuver or {}).get("stop", 1.0) * steps) if (maneuver and "stop" in maneuver) else steps + 2
    a_man = float((maneuver or {}).get("accel", 0.0))
    for k in range(1, steps + 1):
        a = a_man if (maneuver and ks <= k < ke) else 0.0
        v = v + a * dt
        p = p + v * dt
        pos[k], vel[k] = p, v
    meas = pos + rng.normal(0.0, r, size=steps + 1)
    return t, pos, vel, meas


def run_kf(meas, dt, q, r):
    """Run the 1-D constant-velocity Kalman filter over a measurement sequence.

    Returns (est_pos, est_vel, pos_var, gain_pos): the position and velocity
    estimates, the position variance P[0,0] each step, and the position element
    of the Kalman gain each step.
    """
    F, H, Q = cv_matrices(dt, q)
    R = np.array([[r ** 2]])
    x = np.array([[meas[0]], [0.0]])
    P = np.diag([r ** 2, 50.0 ** 2]).astype(float)
    I = np.eye(2)
    est_p = [float(x[0, 0])]
    est_v = [float(x[1, 0])]
    pvar = [float(P[0, 0])]
    gain = [0.0]
    for z in meas[1:]:
        # predict
        x = F @ x
        P = F @ P @ F.T + Q
        # update
        S = H @ P @ H.T + R
        K = P @ H.T @ np.linalg.inv(S)
        x = x + K @ (np.array([[z]]) - H @ x)
        P = (I - K @ H) @ P
        est_p.append(float(x[0, 0]))
        est_v.append(float(x[1, 0]))
        pvar.append(float(P[0, 0]))
        gain.append(float(K[0, 0]))
    return np.array(est_p), np.array(est_v), np.array(pvar), np.array(gain)


def run_alpha_beta(meas, dt, alpha, beta):
    """Run a fixed-gain alpha-beta filter (steady-state CV Kalman) over meas.

    Returns (est_pos, est_vel). alpha weights the position residual into position;
    beta weights it into velocity (scaled by 1/dt).
    """
    # two-point initialization: seed velocity from the first two measurements
    x = float(meas[0])
    v = (float(meas[1]) - float(meas[0])) / dt if len(meas) > 1 else 0.0
    est_p = [x]
    est_v = [v]
    for z in meas[1:]:
        xp = x + v * dt          # predict
        resid = z - xp           # innovation
        x = xp + alpha * resid   # update position
        v = v + (beta / dt) * resid  # update velocity
        est_p.append(x)
        est_v.append(v)
    return np.array(est_p), np.array(est_v)


def rmse(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return float(np.sqrt(np.mean((a - b) ** 2)))


# ================================================================== Module 3
# Nonlinear filters on a 2-D constant-velocity target observed by a fixed sensor
# at the origin through range + bearing (both nonlinear in the Cartesian state).
# State is x = [px, vx, py, vy].
# ------------------------------------------------------------------------------
def wrap_angle(a):
    """Wrap an angle (radians) to [-pi, pi]."""
    return np.arctan2(np.sin(a), np.cos(a))


def cv_matrices_2d(dt, q):
    """4x4 constant-velocity transition F and process-noise Q (DWNA per axis)."""
    F = np.array([[1, dt, 0, 0],
                  [0, 1, 0, 0],
                  [0, 0, 1, dt],
                  [0, 0, 0, 1]], dtype=float)
    qd = (q ** 2) * np.array([[dt ** 4 / 4, dt ** 3 / 2],
                              [dt ** 3 / 2, dt ** 2]])
    Q = np.zeros((4, 4))
    Q[0:2, 0:2] = qd
    Q[2:4, 2:4] = qd
    return F, Q


def h_rb(px, py):
    """Range/bearing measurement of a point from a sensor at the origin."""
    return np.array([np.hypot(px, py), np.arctan2(py, px)])


def simulate_rb_track(steps, dt, sr, sb, start, vel, seed=0):
    """Constant-velocity truth past a sensor at the origin, with noisy
    range/bearing measurements.

    start, vel: 2-vectors (m, m/s). sr: range-noise std (m). sb: bearing-noise
    std (rad). Returns (t, X, meas) with X the (steps+1, 4) truth states
    [px,vx,py,vy] and meas the (steps+1, 2) [range, bearing] measurements.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(steps + 1) * dt
    X = np.zeros((steps + 1, 4))
    X[0] = [start[0], vel[0], start[1], vel[1]]
    F, _ = cv_matrices_2d(dt, 0.0)
    for k in range(1, steps + 1):
        X[k] = F @ X[k - 1]
    meas = np.zeros((steps + 1, 2))
    for k in range(steps + 1):
        z = h_rb(X[k, 0], X[k, 2])
        meas[k, 0] = z[0] + rng.normal(0.0, sr)
        meas[k, 1] = wrap_angle(z[1] + rng.normal(0.0, sb))
    return t, X, meas


def _rb_to_xy(r, b):
    return np.array([r * np.cos(b), r * np.sin(b)])


def _init_from_meas(meas, dt, sr, sb):
    """Two-point initialization of [px,vx,py,vy] and a starting covariance."""
    p0 = _rb_to_xy(*meas[0])
    p1 = _rb_to_xy(*meas[1]) if len(meas) > 1 else p0
    v0 = (p1 - p0) / dt
    x = np.array([p0[0], v0[0], p0[1], v0[1]], dtype=float)
    P = np.diag([ (sr + meas[0, 0] * sb) ** 2, (150.0) ** 2,
                  (sr + meas[0, 0] * sb) ** 2, (150.0) ** 2]).astype(float)
    return x, P


def run_ekf_rb(meas, dt, q, sr, sb):
    """Extended Kalman filter (Jacobian linearization) for range/bearing.

    Returns (est_xy, trace_pos): (steps+1, 2) position estimates and the trace of
    the position covariance each step (a consistency indicator).
    """
    F, Q = cv_matrices_2d(dt, q)
    R = np.diag([sr ** 2, sb ** 2])
    x, P = _init_from_meas(meas, dt, sr, sb)
    I = np.eye(4)
    est = [[x[0], x[2]]]
    trP = [P[0, 0] + P[2, 2]]
    for z in meas[1:]:
        # predict
        x = F @ x
        P = F @ P @ F.T + Q
        px, py = x[0], x[2]
        r2 = px * px + py * py
        r = np.sqrt(r2) + 1e-9
        # measurement Jacobian
        H = np.array([[px / r, 0, py / r, 0],
                      [-py / r2, 0, px / r2, 0]])
        zhat = np.array([r, np.arctan2(py, px)])
        y = z - zhat
        y[1] = wrap_angle(y[1])
        S = H @ P @ H.T + R
        K = P @ H.T @ np.linalg.inv(S)
        x = x + K @ y
        P = (I - K @ H) @ P
        est.append([x[0], x[2]])
        trP.append(P[0, 0] + P[2, 2])
    return np.array(est), np.array(trP)


def _sigma_points(x, P, kappa=0.0):
    """Plain unscented transform sigma points (Julier), nonnegative weights.
    Returns (X sigma matrix 2n+1 x n, Wm, Wc)."""
    n = len(x)
    lam = n + kappa
    try:
        Scol = np.linalg.cholesky(lam * P)
    except np.linalg.LinAlgError:
        # nudge to positive definite
        Scol = np.linalg.cholesky(lam * (P + 1e-6 * np.eye(n)))
    X = np.zeros((2 * n + 1, n))
    X[0] = x
    for i in range(n):
        X[i + 1] = x + Scol[:, i]
        X[i + 1 + n] = x - Scol[:, i]
    Wm = np.full(2 * n + 1, 1.0 / (2 * lam))
    Wm[0] = kappa / lam
    Wc = Wm.copy()
    return X, Wm, Wc


def run_ukf_rb(meas, dt, q, sr, sb, kappa=0.0):
    """Unscented Kalman filter (sigma points, no Jacobians) for range/bearing.

    Angle-safe: the predicted bearing mean is a circular mean and every bearing
    residual is wrapped. Returns (est_xy, trace_pos) like run_ekf_rb.
    """
    F, Q = cv_matrices_2d(dt, q)
    R = np.diag([sr ** 2, sb ** 2])
    x, P = _init_from_meas(meas, dt, sr, sb)
    n = 4
    est = [[x[0], x[2]]]
    trP = [P[0, 0] + P[2, 2]]
    for z in meas[1:]:
        # ---- predict ----
        X, Wm, Wc = _sigma_points(x, P, kappa)
        Xp = (F @ X.T).T
        xp = Wm @ Xp
        Pp = Q.copy()
        for i in range(len(Wm)):
            d = Xp[i] - xp
            Pp += Wc[i] * np.outer(d, d)
        # ---- update ----
        X2, Wm, Wc = _sigma_points(xp, Pp, kappa)
        Z = np.array([h_rb(s[0], s[2]) for s in X2])
        # circular mean for bearing, plain mean for range
        zr = Wm @ Z[:, 0]
        zb = np.arctan2(Wm @ np.sin(Z[:, 1]), Wm @ np.cos(Z[:, 1]))
        zhat = np.array([zr, zb])
        S = R.copy()
        Pxz = np.zeros((n, 2))
        for i in range(len(Wm)):
            dz = Z[i] - zhat
            dz[1] = wrap_angle(dz[1])
            dx = X2[i] - xp
            S += Wc[i] * np.outer(dz, dz)
            Pxz += Wc[i] * np.outer(dx, dz)
        K = Pxz @ np.linalg.inv(S)
        y = z - zhat
        y[1] = wrap_angle(y[1])
        x = xp + K @ y
        P = Pp - K @ S @ K.T
        est.append([x[0], x[2]])
        trP.append(P[0, 0] + P[2, 2])
    return np.array(est), np.array(trP)


def run_pf_rb(meas, dt, q, sr, sb, n_particles=500, seed=0):
    """Bootstrap particle filter for range/bearing tracking.

    Propagates particles with the CV model plus process noise, weights them by
    the range/bearing likelihood, and systematically resamples when the effective
    sample size drops below half. Returns (est_xy, ess_frac) where ess_frac is the
    effective-sample-size fraction each step (a degeneracy indicator).
    """
    rng = np.random.default_rng(seed)
    F, Q = cv_matrices_2d(dt, q)
    x0, P0 = _init_from_meas(meas, dt, sr, sb)
    P = np.stack([rng.multivariate_normal(x0, P0) for _ in range(n_particles)])
    w = np.full(n_particles, 1.0 / n_particles)
    # process-noise sampling covariance (a little floor so particles keep spread)
    Qs = Q + 1e-6 * np.eye(4)
    est = [[x0[0], x0[2]]]
    ess = [1.0]
    for z in meas[1:]:
        # propagate
        noise = rng.multivariate_normal(np.zeros(4), Qs, size=n_particles)
        P = (F @ P.T).T + noise
        # weight by likelihood
        r = np.hypot(P[:, 0], P[:, 2])
        b = np.arctan2(P[:, 2], P[:, 0])
        dr = z[0] - r
        db = wrap_angle(z[1] - b)
        logw = -0.5 * ((dr / sr) ** 2 + (db / sb) ** 2)
        logw -= logw.max()
        w = w * np.exp(logw)
        s = w.sum()
        w = w / s if s > 0 else np.full(n_particles, 1.0 / n_particles)
        xhat = w @ P
        est.append([xhat[0], xhat[2]])
        neff = 1.0 / np.sum(w ** 2)
        ess.append(neff / n_particles)
        # systematic resample when degenerate, with regularized roughening so the
        # cloud does not collapse to a few points (particle impoverishment)
        if neff < n_particles / 2:
            positions = (rng.random() + np.arange(n_particles)) / n_particles
            idx = np.searchsorted(np.cumsum(w), positions)
            idx = np.clip(idx, 0, n_particles - 1)
            P = P[idx]
            stds = P.std(axis=0)
            P = P + rng.normal(0.0, 1.0, P.shape) * (0.25 * stds)
            w = np.full(n_particles, 1.0 / n_particles)
    return np.array(est), np.array(ess)


# ================================================================== Module 4
# Bearings-only (angle-only) tracking. A moving observer measures only the
# bearing to a constant-velocity target; range is unobservable unless the
# observer maneuvers. State conventions: target absolute state [Tx,Tvx,Ty,Tvy];
# observer position/velocity known (own-ship data). Bearing measured from the
# +x axis: beta = atan2(Ty-Oy, Tx-Ox).
# ------------------------------------------------------------------------------
def simulate_bo(steps, dt, sb, target0, tvel, obs0, ovel,
                maneuver=None, seed=0):
    """Bearings-only scenario. target0/tvel, obs0/ovel are 2-vectors (m, m/s).
    maneuver=dict(at=frac, vel=(vx,vy)) changes the OBSERVER velocity to vel at
    step int(at*steps) (a course change that restores observability); None keeps
    the observer on a straight constant-velocity leg. Returns
    (t, T, O, Ovel, bearings): target states (steps+1,4), observer positions
    (steps+1,2), observer velocities (steps+1,2), noisy bearings (steps+1,)."""
    rng = np.random.default_rng(seed)
    t = np.arange(steps + 1) * dt
    T = np.zeros((steps + 1, 4))
    O = np.zeros((steps + 1, 2))
    Ov = np.zeros((steps + 1, 2))
    T[0] = [target0[0], tvel[0], target0[1], tvel[1]]
    O[0] = [obs0[0], obs0[1]]
    ov = np.array(ovel, dtype=float)
    Ov[0] = ov
    kman = int((maneuver or {}).get("at", 2.0) * steps) if maneuver else steps + 2
    for k in range(1, steps + 1):
        # target constant velocity
        T[k, 0] = T[k - 1, 0] + T[k - 1, 1] * dt
        T[k, 2] = T[k - 1, 2] + T[k - 1, 3] * dt
        T[k, 1] = T[k - 1, 1]; T[k, 3] = T[k - 1, 3]
        # observer velocity change at the maneuver step
        if maneuver and k == kman:
            ov = np.array(maneuver["vel"], dtype=float)
        O[k] = O[k - 1] + ov * dt
        Ov[k] = ov
    bearings = np.zeros(steps + 1)
    for k in range(steps + 1):
        b = np.arctan2(T[k, 2] - O[k, 1], T[k, 0] - O[k, 0])
        bearings[k] = wrap_angle(b + rng.normal(0.0, sb))
    return t, T, O, Ov, bearings


def run_ekf_bo(bearings, dt, O, q, sb, init_range, init_speed=10.0):
    """Cartesian bearings-only EKF on the target state [Tx,Tvx,Ty,Tvy]. The
    observer positions O (steps+1,2) are known. Initialized along the first
    bearing at range init_range (range poorly known -> large covariance). Returns
    (est_xy, pos_sigma): target position estimate (steps+1,2) and the reported
    position 1-sigma sqrt(P_xx+P_yy) each step (its collapse below the actual
    error is the EKF's inconsistency)."""
    F, Q = cv_matrices_2d(dt, q)
    R = np.array([[sb ** 2]])
    b0 = bearings[0]
    x = np.array([O[0, 0] + init_range * np.cos(b0), 0.0,
                  O[0, 1] + init_range * np.sin(b0), 0.0])
    P = np.diag([init_range ** 2, init_speed ** 2,
                 init_range ** 2, init_speed ** 2]).astype(float)
    I = np.eye(4)
    est = [[x[0], x[2]]]
    psig = [np.sqrt(P[0, 0] + P[2, 2])]
    for k in range(1, len(bearings)):
        x = F @ x
        P = F @ P @ F.T + Q
        dx = x[0] - O[k, 0]; dy = x[2] - O[k, 1]
        r2 = dx * dx + dy * dy + 1e-9
        zhat = np.arctan2(dy, dx)
        H = np.array([[-dy / r2, 0.0, dx / r2, 0.0]])
        y = np.array([wrap_angle(bearings[k] - zhat)])
        S = H @ P @ H.T + R
        Kk = P @ H.T @ np.linalg.inv(S)
        x = x + (Kk @ y).ravel()
        P = (I - Kk @ H) @ P
        est.append([x[0], x[2]])
        psig.append(np.sqrt(P[0, 0] + P[2, 2]))
    return np.array(est), np.array(psig)


# ---- modified polar coordinates (Aidala-Hammel) EKF --------------------------
def _mpc_from_rel(p, w):
    """Relative Cartesian (p position, w velocity) -> MPC state [beta, betadot,
    rdot/r, 1/r]."""
    r = np.hypot(p[0], p[1])
    beta = np.arctan2(p[1], p[0])
    rdot = (p[0] * w[0] + p[1] * w[1]) / r
    betadot = (p[0] * w[1] - p[1] * w[0]) / (r * r)
    return np.array([beta, betadot, rdot / r, 1.0 / r])


def _rel_from_mpc(y):
    """MPC state -> relative Cartesian (p, w)."""
    beta, betadot, rho, alpha = y
    r = 1.0 / alpha
    c, s = np.cos(beta), np.sin(beta)
    p = np.array([r * c, r * s])
    rdot = rho * r
    w = rdot * np.array([c, s]) + r * betadot * np.array([-s, c])
    return p, w


def _mpc_propagate(y, dt, dv_obs):
    """Exact MPC mean propagation over dt via a Cartesian round-trip, applying a
    known observer velocity change dv_obs (relative velocity shifts by -dv_obs)."""
    p, w = _rel_from_mpc(y)
    p = p + w * dt
    w = w - dv_obs
    return _mpc_from_rel(p, w)


def run_mpc_ekf(bearings, dt, O, Ov, q, sb, init_range, init_speed=10.0):
    """Modified-polar-coordinates EKF for bearings-only tracking. The bearing
    measurement is LINEAR in the MPC state (H = [1,0,0,0]) and the unobservable
    inverse-range is a separate state, so the filter stays consistent where the
    Cartesian EKF does not. Mean propagation is exact (Cartesian round-trip);
    the covariance uses a numerical Jacobian. Returns (est_xy, pos_sigma)."""
    R = np.array([[sb ** 2]])
    # init MPC state from the first two bearings
    b0, b1 = bearings[0], bearings[1]
    betadot0 = wrap_angle(b1 - b0) / dt
    y = np.array([b0, betadot0, 0.0, 1.0 / init_range])
    # generous covariance, especially on inverse range (unobservable at first)
    P = np.diag([sb ** 2, (betadot0 ** 2 + (sb / dt) ** 2),
                 (0.02) ** 2, (1.0 / init_range) ** 2]).astype(float)
    Qm = np.diag([1e-9, 1e-8, (q / init_range * dt) ** 2, 1e-14])
    I = np.eye(4)
    H = np.array([[1.0, 0.0, 0.0, 0.0]])

    def est_pos(yv, Ok):
        p, _ = _rel_from_mpc(yv)
        return Ok + p

    est = [est_pos(y, O[0])]
    psig = [0.0]
    for k in range(1, len(bearings)):
        dv = Ov[k] - Ov[k - 1]                 # observer velocity change
        # predict mean (exact) and covariance (numerical Jacobian)
        f0 = _mpc_propagate(y, dt, dv)
        Fj = np.zeros((4, 4))
        for j in range(4):
            dy = np.zeros(4)
            step = 1e-6 * max(1.0, abs(y[j]))
            dy[j] = step
            Fj[:, j] = (_mpc_propagate(y + dy, dt, dv) - f0) / step
        y = f0
        P = Fj @ P @ Fj.T + Qm
        # linear bearing update
        zhat = y[0]
        resid = np.array([wrap_angle(bearings[k] - zhat)])
        S = H @ P @ H.T + R
        Kk = P @ H.T @ np.linalg.inv(S)
        y = y + (Kk @ resid).ravel()
        y[0] = wrap_angle(y[0])
        P = (I - Kk @ H) @ P
        # reported Cartesian position sigma via numerical Jacobian of est_pos
        pos = est_pos(y, O[k])
        Jp = np.zeros((2, 4))
        for j in range(4):
            dy = np.zeros(4); step = 1e-6 * max(1.0, abs(y[j])); dy[j] = step
            Jp[:, j] = (est_pos(y + dy, O[k]) - pos) / step
        pcov = Jp @ P @ Jp.T
        est.append(pos)
        psig.append(np.sqrt(max(pcov[0, 0] + pcov[1, 1], 0.0)))
    return np.array(est), np.array(psig)


# ================================================================== Module 5
# Maneuvering targets: adaptive detection and the Interacting Multiple Model
# (IMM) estimator. A target flies constant-velocity, turns, then straightens.
# A single CV filter lags the turn; a bank of models (CV + coordinated-turn),
# blended each step by their mode probabilities, tracks through it. State is the
# 2-D Cartesian x = [px, vx, py, vy], measured in position only.
# ------------------------------------------------------------------------------
def ct2d(dt, omega, q):
    """Coordinated-turn transition F and process-noise Q for state
    [px,vx,py,vy] at turn rate omega (rad/s).

    As omega -> 0 the matrix limits to the constant-velocity F, so a small turn
    rate is handled gracefully. Q is the same discrete white-noise-acceleration
    covariance used by the CV model (the turn is in F, not Q).
    """
    w = float(omega)
    if abs(w) < 1e-6:
        F, Q = cv_matrices_2d(dt, q)
        return F, Q
    s, c = np.sin(w * dt), np.cos(w * dt)
    F = np.array([[1.0, s / w,       0.0, -(1.0 - c) / w],
                  [0.0, c,           0.0, -s],
                  [0.0, (1.0 - c) / w, 1.0, s / w],
                  [0.0, s,           0.0, c]], dtype=float)
    qd = (q ** 2) * np.array([[dt ** 4 / 4, dt ** 3 / 2],
                              [dt ** 3 / 2, dt ** 2]])
    Q = np.zeros((4, 4))
    Q[0:2, 0:2] = qd
    Q[2:4, 2:4] = qd
    return F, Q


def simulate_maneuver_2d(steps, dt, r, speed=200.0, start=(0.0, 0.0),
                         heading_deg=0.0, turn_rate_deg=3.0,
                         turn_window=(0.375, 0.625), seed=0):
    """Truth path that flies straight, turns at a constant rate inside
    turn_window (fractions of the run), then flies straight again, plus noisy
    2-D position measurements.

    Returns (t, X, meas): X the (steps+1,4) truth [px,vx,py,vy]; meas the
    (steps+1,2) noisy [px,py] measurements. r is the position-measurement std
    (m). The finite turn window is what makes a single CV filter visibly lag
    while an IMM recovers.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(steps + 1) * dt
    h = np.deg2rad(heading_deg)
    w = np.deg2rad(turn_rate_deg)
    ks = int(turn_window[0] * steps)
    ke = int(turn_window[1] * steps)
    X = np.zeros((steps + 1, 4))
    X[0] = [start[0], speed * np.cos(h), start[1], speed * np.sin(h)]
    for k in range(1, steps + 1):
        turning = ks <= k <= ke
        F, _ = (ct2d(dt, w, 0.0) if turning else cv_matrices_2d(dt, 0.0))
        X[k] = F @ X[k - 1]
    meas = X[:, [0, 2]] + rng.normal(0.0, r, size=(steps + 1, 2))
    return t, X, meas


def _H_pos2d():
    """Position-only measurement matrix for state [px,vx,py,vy]."""
    return np.array([[1.0, 0.0, 0.0, 0.0],
                     [0.0, 0.0, 1.0, 0.0]])


def run_kf2d(meas, dt, r, model="cv", omega_deg=3.0, q=2.0):
    """Single linear Kalman filter on 2-D position measurements, using either
    the constant-velocity ("cv") or a fixed coordinated-turn ("ct") model.

    Returns (est_xy, nis): the (steps+1,2) position estimate and the normalized
    innovation squared (NIS) each step. Under a correctly modeled target NIS
    averages the measurement dimension (2 here); a sustained NIS well above 2 is
    the adaptive maneuver detector's trigger.
    """
    if model == "ct":
        F, Q = ct2d(dt, np.deg2rad(omega_deg), q)
    else:
        F, Q = cv_matrices_2d(dt, q)
    H = _H_pos2d()
    R = (r ** 2) * np.eye(2)
    # two-point initialization
    p0 = np.array([meas[0, 0], meas[0, 1]])
    p1 = np.array([meas[1, 0], meas[1, 1]]) if len(meas) > 1 else p0
    v0 = (p1 - p0) / dt
    x = np.array([p0[0], v0[0], p0[1], v0[1]], dtype=float)
    P = np.diag([r ** 2, (r / dt) ** 2 * 2, r ** 2, (r / dt) ** 2 * 2]).astype(float)
    I = np.eye(4)
    est = [[x[0], x[2]]]
    nis = [0.0]
    for z in meas[1:]:
        x = F @ x
        P = F @ P @ F.T + Q
        yk = z - H @ x
        S = H @ P @ H.T + R
        Sinv = np.linalg.inv(S)
        K = P @ H.T @ Sinv
        x = x + K @ yk
        P = (I - K @ H) @ P
        est.append([x[0], x[2]])
        nis.append(float(yk @ Sinv @ yk))
    return np.array(est), np.array(nis)


def run_imm(meas, dt, r, models=None, tpm=None, mu0=None):
    """Interacting Multiple Model (IMM) estimator on 2-D position measurements.

    models: list of dicts, each dict(kind="cv"|"ct", q=..., omega_deg=... ). The
    default bank is a constant-velocity model and a coordinated-turn model.
    tpm: mode transition-probability matrix (rows sum to 1); default sticky.
    mu0: initial mode probabilities.

    Each step runs the standard IMM cycle: mix each filter's state with the
    others in proportion to the transition probabilities, run every filter
    through predict/update, score each by its innovation likelihood, update the
    mode probabilities, and combine into one estimate. Returns
    (est_xy, mu_hist, nis_combined): the (steps+1,2) blended position estimate,
    the (steps+1, n_models) mode-probability history, and a combined NIS each
    step (the maneuver still shows as a transient bump while the bank re-mixes).
    """
    if models is None:
        models = [dict(kind="cv", q=2.0),
                  dict(kind="ct", q=2.0, omega_deg=3.0)]
    n = len(models)
    if tpm is None:
        # sticky by default: stay in the current mode most of the time
        tpm = np.full((n, n), 0.05 / max(n - 1, 1))
        np.fill_diagonal(tpm, 0.95)
    tpm = np.asarray(tpm, dtype=float)
    mu = np.full(n, 1.0 / n) if mu0 is None else np.asarray(mu0, dtype=float)

    H = _H_pos2d()
    R = (r ** 2) * np.eye(2)

    def matrices(m):
        if m["kind"] == "ct":
            return ct2d(dt, np.deg2rad(m.get("omega_deg", 3.0)), m.get("q", 2.0))
        return cv_matrices_2d(dt, m.get("q", 2.0))

    # shared two-point initialization for every model
    p0 = np.array([meas[0, 0], meas[0, 1]])
    p1 = np.array([meas[1, 0], meas[1, 1]]) if len(meas) > 1 else p0
    v0 = (p1 - p0) / dt
    x_init = np.array([p0[0], v0[0], p0[1], v0[1]], dtype=float)
    P_init = np.diag([r ** 2, (r / dt) ** 2 * 2, r ** 2, (r / dt) ** 2 * 2]).astype(float)
    xs = [x_init.copy() for _ in range(n)]
    Ps = [P_init.copy() for _ in range(n)]
    I = np.eye(4)

    est = [[x_init[0], x_init[2]]]
    mu_hist = [mu.copy()]
    nis_hist = [0.0]

    for z in meas[1:]:
        # ---- 1. mixing: predicted mode probs and mixing weights ----
        cbar = tpm.T @ mu                       # c_j = sum_i tpm_ij mu_i
        cbar = np.where(cbar > 0, cbar, 1e-12)
        # mixing weight mu_{i|j} = tpm_ij mu_i / c_j
        W = (tpm * mu[:, None]) / cbar[None, :]
        x_mix, P_mix = [], []
        for j in range(n):
            xj = sum(W[i, j] * xs[i] for i in range(n))
            Pj = np.zeros((4, 4))
            for i in range(n):
                d = xs[i] - xj
                Pj += W[i, j] * (Ps[i] + np.outer(d, d))
            x_mix.append(xj); P_mix.append(Pj)
        # ---- 2. model-matched filtering + likelihoods ----
        L = np.zeros(n)
        for j, m in enumerate(models):
            F, Q = matrices(m)
            xj = F @ x_mix[j]
            Pj = F @ P_mix[j] @ F.T + Q
            yk = z - H @ xj
            S = H @ Pj @ H.T + R
            Sinv = np.linalg.inv(S)
            K = Pj @ H.T @ Sinv
            xj = xj + K @ yk
            Pj = (I - K @ H) @ Pj
            xs[j], Ps[j] = xj, Pj
            md2 = float(yk @ Sinv @ yk)
            L[j] = np.exp(-0.5 * md2) / np.sqrt(np.linalg.det(2 * np.pi * S))
        # ---- 3. mode-probability update ----
        mu = cbar * L
        s = mu.sum()
        mu = mu / s if s > 0 else np.full(n, 1.0 / n)
        # ---- 4. combined estimate ----
        x_comb = sum(mu[j] * xs[j] for j in range(n))
        est.append([x_comb[0], x_comb[2]])
        mu_hist.append(mu.copy())
        # combined innovation (about the blended predicted measurement)
        ycomb = z - H @ x_comb
        nis_hist.append(float(ycomb @ np.linalg.inv(R + 1e-9 * np.eye(2)) @ ycomb))

    return np.array(est), np.array(mu_hist), np.array(nis_hist)


# ================================================================== Module 6
# Single-target correlation & association. A tracked target produces a predicted
# measurement zhat with innovation covariance S; among several returns (the true
# detection, plus clutter, with a chance of a missed detection) the tracker must
# decide which return to use. Gating discards implausible returns; the
# Mahalanobis distance ranks the survivors; nearest-neighbor picks the closest.
# The measurement here is 2-D position, so the innovation is 2-D (dof 2). Reuses
# the constant-velocity model cv_matrices_2d and the position matrix _H_pos2d.
# ------------------------------------------------------------------------------
def _gammainc_lower_reg(s, x):
    """Regularized lower incomplete gamma P(s, x) = gamma(s,x)/Gamma(s), no scipy.

    Series expansion for x < s+1, continued fraction otherwise (Numerical
    Recipes). Good to ~1e-10, enough for chi-square gate thresholds.
    """
    if x <= 0:
        return 0.0
    gln = math.lgamma(s)
    if x < s + 1.0:
        ap = s; total = 1.0 / s; delta = total
        for _ in range(200):
            ap += 1.0; delta *= x / ap; total += delta
            if abs(delta) < abs(total) * 1e-12:
                break
        return total * math.exp(-x + s * math.log(x) - gln)
    # continued fraction
    tiny = 1e-30
    b = x + 1.0 - s; c = 1.0 / tiny; d = 1.0 / b; h = d
    for i in range(1, 200):
        an = -i * (i - s)
        b += 2.0
        d = an * d + b
        if abs(d) < tiny: d = tiny
        c = b + an / c
        if abs(c) < tiny: c = tiny
        d = 1.0 / d; delt = d * c; h *= delt
        if abs(delt - 1.0) < 1e-12:
            break
    return 1.0 - math.exp(-x + s * math.log(x) - gln) * h


def chi2_cdf(x, dof):
    """Chi-square CDF via the regularized lower incomplete gamma."""
    return _gammainc_lower_reg(dof / 2.0, x / 2.0)


def chi2_gate(pg, dof=2):
    """Gate threshold gamma with gate probability pg for a chi-square with `dof`
    degrees of freedom: P(d^2 <= gamma) = pg, where d^2 is the normalized
    (Mahalanobis) innovation. For dof=2 this is the closed form gamma =
    -2 ln(1-pg); other dof are solved by bisection on chi2_cdf.
    """
    pg = float(np.clip(pg, 1e-6, 1 - 1e-9))
    if dof == 2:
        return -2.0 * math.log(1.0 - pg)
    lo, hi = 0.0, 1.0
    while chi2_cdf(hi, dof) < pg:
        hi *= 2.0
        if hi > 1e6:
            break
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        if chi2_cdf(mid, dof) < pg:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def mahalanobis2(nu, S):
    """Squared statistical (Mahalanobis) distance nu^T S^{-1} nu of an innovation
    nu under innovation covariance S. nu may be a single vector (returns a float)
    or an (m, d) array of m innovations (returns an (m,) array)."""
    nu = np.asarray(nu, dtype=float)
    Sinv = np.linalg.inv(S)
    if nu.ndim == 1:
        return float(nu @ Sinv @ nu)
    return np.einsum('ij,jk,ik->i', nu, Sinv, nu)


def gate_ellipse(zhat, S, gamma, n=120):
    """Boundary points of the ellipsoidal gate {z : (z-zhat)^T S^{-1}(z-zhat) =
    gamma} for plotting. Returns an (n, 2) array tracing the ellipse."""
    th = np.linspace(0, 2 * np.pi, n)
    circle = np.stack([np.cos(th), np.sin(th)])       # 2 x n unit circle
    L = np.linalg.cholesky(S)                          # S = L L^T
    pts = zhat[:, None] + np.sqrt(gamma) * (L @ circle)
    return pts.T


def ellipsoidal_gate(zs, zhat, S, gamma):
    """Boolean mask of returns zs (m,2) inside the ellipsoidal gate."""
    zs = np.atleast_2d(zs)
    return mahalanobis2(zs - zhat, S) <= gamma


def rectangular_gate(zs, zhat, S, gamma):
    """Boolean mask for the cheap axis-aligned (rectangular) gate: each coordinate
    of the innovation must satisfy |nu_i| <= sqrt(gamma * S_ii). Coarser than the
    ellipsoidal gate (it ignores the S off-diagonal), so it admits more clutter."""
    zs = np.atleast_2d(zs)
    nu = zs - zhat
    halfwidth = np.sqrt(gamma * np.diag(S))
    return np.all(np.abs(nu) <= halfwidth, axis=1)


def nn_associate(zs, zhat, S, gamma, gate='ellipse'):
    """Nearest-neighbor association: among the returns zs (m,2) that fall inside
    the gate, return the index of the one with the smallest Mahalanobis distance,
    or None if the gate is empty. gate='ellipse' or 'rect'."""
    zs = np.atleast_2d(zs)
    mask = (ellipsoidal_gate(zs, zhat, S, gamma) if gate == 'ellipse'
            else rectangular_gate(zs, zhat, S, gamma))
    if not np.any(mask):
        return None
    d2 = mahalanobis2(zs - zhat, S)
    d2 = np.where(mask, d2, np.inf)
    return int(np.argmin(d2))


def clutter_in_gate(zhat, S, gamma, beta, rng, gate='ellipse'):
    """Poisson clutter at spatial density beta (returns per unit area, m^-2)
    inside the gate. Draws Poisson(beta * bbox_area) uniform points over the
    ellipse's bounding box, then keeps those inside the chosen gate. Returns an
    (m, 2) array (possibly empty)."""
    ell = gate_ellipse(zhat, S, gamma)
    lo = ell.min(axis=0); hi = ell.max(axis=0)
    area = float((hi[0] - lo[0]) * (hi[1] - lo[1]))
    n = rng.poisson(beta * area)
    if n == 0:
        return np.empty((0, 2))
    pts = rng.uniform(lo, hi, size=(n, 2))
    mask = (ellipsoidal_gate(pts, zhat, S, gamma) if gate == 'ellipse'
            else rectangular_gate(pts, zhat, S, gamma))
    return pts[mask]


def cov_ellipse_R(sr, sc, angle_deg):
    """A 2x2 measurement-noise covariance R with along-track std sr and
    cross-track std sc, rotated by angle_deg. Handy for an elongated/correlated
    gate (e.g. a sensor with good range but poor cross-range)."""
    a = np.deg2rad(angle_deg)
    Rot = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    D = np.diag([sr ** 2, sc ** 2])
    return Rot @ D @ Rot.T


def simulate_clutter_track(steps, dt, R, beta, pd, speed=200.0, start=(-3000.0, 400.0),
                           heading_deg=0.0, q=1.0, pg=0.99, seed=0, gate='ellipse'):
    """Run a constant-velocity Kalman filter with nearest-neighbor gating on a
    single target embedded in Poisson clutter, and score the association.

    R: 2x2 position measurement covariance. beta: clutter spatial density
    (returns m^-2). pd: probability the target is detected each step. pg: gate
    probability (sets the chi-square threshold). Returns a dict with the truth,
    estimates, per-step return sets, chosen indices, and the mis-association
    rate = fraction of detected steps whose NN pick was NOT the true target
    return (a clutter return won, or the target was missed and clutter was
    picked)."""
    rng = np.random.default_rng(seed)
    F, Q = cv_matrices_2d(dt, q)
    H = _H_pos2d()
    gamma = chi2_gate(pg, dof=2)
    h = np.deg2rad(heading_deg)
    truth = np.zeros((steps + 1, 4))
    truth[0] = [start[0], speed * np.cos(h), start[1], speed * np.sin(h)]
    for k in range(1, steps + 1):
        truth[k] = F @ truth[k - 1]
    tpos = truth[:, [0, 2]]
    # initialize filter near the first true position
    x = np.array([tpos[0, 0], truth[0, 1], tpos[0, 1], truth[0, 3]], dtype=float)
    P = np.diag([R[0, 0] * 4, (speed) ** 2, R[1, 1] * 4, (speed) ** 2]).astype(float)
    est = [[x[0], x[2]]]
    detected = 0; wrong = 0
    frames = []
    for k in range(1, steps + 1):
        x = F @ x
        P = F @ P @ F.T + Q
        zhat = H @ x
        S = H @ P @ H.T + R
        # returns: true detection (maybe) + clutter
        returns = []
        target_idx = None
        is_det = rng.random() < pd
        if is_det:
            zt = tpos[k] + rng.multivariate_normal(np.zeros(2), R)
            returns.append(zt); target_idx = 0
        clut = clutter_in_gate(zhat, S, gamma, beta, rng, gate)
        for c in clut:
            returns.append(c)
        returns = np.array(returns) if returns else np.empty((0, 2))
        pick = nn_associate(returns, zhat, S, gamma, gate) if len(returns) else None
        # score association on detected steps where the target return is in the gate
        if is_det and ellipsoidal_gate(tpos[k][None, :], zhat, S, gamma)[0]:
            detected += 1
            if pick is None or pick != target_idx:
                wrong += 1
        # update with the picked return (NN); coast if none
        if pick is not None:
            z = returns[pick]
            y = z - zhat
            K = P @ H.T @ np.linalg.inv(S)
            x = x + K @ y
            P = (np.eye(4) - K @ H) @ P
        est.append([x[0], x[2]])
        frames.append(dict(zhat=zhat.copy(), S=S.copy(), returns=returns,
                           pick=pick, target_idx=target_idx, truth=tpos[k].copy()))
    rate = wrong / detected if detected else 0.0
    return dict(truth=tpos, est=np.array(est), frames=frames,
                misassoc_rate=rate, gamma=gamma)


def misassoc_vs_density(betas, dt=1.0, R=None, pd=0.95, pg=0.99, steps=60,
                        n_mc=40, gate='ellipse', seed0=100):
    """Monte-Carlo mean nearest-neighbor mis-association rate versus clutter
    density, over betas (returns m^-2). Returns an array of mean rates."""
    if R is None:
        R = (60.0 ** 2) * np.eye(2)
    out = []
    for bi, beta in enumerate(betas):
        rates = []
        for m in range(n_mc):
            r = simulate_clutter_track(steps, dt, R, beta, pd, pg=pg,
                                       seed=seed0 + 1000 * bi + m, gate=gate)
            rates.append(r['misassoc_rate'])
        out.append(float(np.mean(rates)))
    return np.array(out)


# =====================================================================
# Module 7 - Track management (M-of-N, SPRT) and range-rate (Doppler)
# =====================================================================
# Shared compute for the figures (figures/mod07.py) and the exercise app
# (exercises/ex07_track_mgmt.py), so the printed figures and the interactive
# app agree. State convention is [px, vx, py, vy] as in the Day-1 helpers.

# ---- 7.2 M-of-N sliding-window confirmation ----
def m_of_n(M, N, p):
    """P(>= M detections in a window of N Bernoulli(p) scans).

    p = single-scan detection probability. Use p = Pd for a true target and
    p = Pfa-in-gate for a clutter-seeded tentative track.
    """
    return float(sum(math.comb(N, k) * p ** k * (1 - p) ** (N - k)
                     for k in range(M, N + 1)))


def mofn_curve(rules, pd, pfa):
    """For a list of (M, N) rules, return (labels, p_confirm, p_falsetrack).

    p_confirm  = m_of_n(M, N, pd)   (a real target confirms)
    p_falsetrack = m_of_n(M, N, pfa) (a clutter sequence confirms)
    """
    labels = ["%d/%d" % (M, N) for (M, N) in rules]
    pc = np.array([m_of_n(M, N, pd) for (M, N) in rules])
    pf = np.array([m_of_n(M, N, pfa) for (M, N) in rules])
    return labels, pc, pf


# ---- 7.3 Sequential probability ratio test (SPRT) ----
def sprt_thresholds(alpha, beta):
    """Wald's log thresholds. Confirm at L >= A, delete at L <= B.

    alpha = P(confirm | no target); beta = P(delete | target present).
    """
    A = math.log((1 - beta) / alpha)
    B = math.log(beta / (1 - alpha))
    return A, B


def sprt_increments(pd, pfa):
    """Per-scan log-likelihood-ratio increments (detect, miss) for a
    detection model H1: Pd (target) vs H0: Pfa (clutter in the gate)."""
    l_det = math.log(pd / pfa)
    l_miss = math.log((1 - pd) / (1 - pfa))
    return l_det, l_miss


def sprt_run(hits, pd, pfa, alpha, beta):
    """Walk a 0/1 detection sequence, accumulating the SPRT score.

    hits: iterable of 0/1 (1 = gated detection that scan). Returns a dict with
    the running score `L` (array, one per scan incl. the start at 0), the
    `decision` ('confirm' | 'delete' | 'pending'), the scan index `k` at which
    it was reached (or None), and the thresholds `A`, `B`.
    """
    A, B = sprt_thresholds(alpha, beta)
    l_det, l_miss = sprt_increments(pd, pfa)
    L = [0.0]
    decision, kdec = "pending", None
    for i, h in enumerate(hits):
        L.append(L[-1] + (l_det if h else l_miss))
        if decision == "pending":
            if L[-1] >= A:
                decision, kdec = "confirm", i + 1
            elif L[-1] <= B:
                decision, kdec = "delete", i + 1
    return {"L": np.array(L), "decision": decision, "k": kdec, "A": A, "B": B}


def sprt_asn(pd, pfa, alpha, beta, hypothesis="target"):
    """Wald average sample number (expected scans to a decision).

    hypothesis='target' -> under H1 (Pd), 'false' -> under H0 (Pfa). This is
    the classic approximation (ignores boundary overshoot, so it slightly
    under-counts; real decisions take about one extra scan)."""
    A, B = sprt_thresholds(alpha, beta)
    l_det, l_miss = sprt_increments(pd, pfa)
    if hypothesis == "target":
        p = pd
        num = (1 - beta) * A + beta * B
    else:
        p = pfa
        num = alpha * A + (1 - alpha) * B
    einc = p * l_det + (1 - p) * l_miss
    return float(num / einc)


def sim_detection_stream(n_scans, p, rng):
    """A 0/1 detection stream of length n_scans, each scan Bernoulli(p)."""
    return (rng.random(n_scans) < p).astype(int)


# ---- 7.5 / 7.6 Measured range-rate (Doppler) ----
def h_rangerate(x):
    """Radial-velocity (range-rate) measurement of state [px,vx,py,vy] from a
    sensor at the origin:  rdot = (px*vx + py*vy) / sqrt(px^2 + py^2)."""
    px, vx, py, vy = x
    r = math.hypot(px, py)
    return (px * vx + py * vy) / r


def H_rangerate(x):
    """1x4 Jacobian of h_rangerate at state x (row vector)."""
    px, vx, py, vy = x
    r = math.hypot(px, py)
    r3 = r ** 3
    dot = px * vx + py * vy
    return np.array([[(vx * r * r - dot * px) / r3, px / r,
                      (vy * r * r - dot * py) / r3, py / r]])


def simulate_rr_track(steps=40, dt=1.0, q=0.5, sig_pos=40.0, sig_rr=2.0,
                      start=(4000.0, 800.0), vel=(-60.0, -8.0), seed=0):
    """Constant-velocity truth past a sensor at the origin, with noisy
    position and range-rate measurements. Default geometry is a closing
    target (moving toward the origin). Returns (t, X, Zpos, Zrr)."""
    rng = np.random.default_rng(seed)
    F, Q = cv_matrices_2d(dt, q)
    x = np.array([start[0], vel[0], start[1], vel[1]], dtype=float)
    t = np.arange(steps + 1) * dt
    X = np.zeros((steps + 1, 4))
    X[0] = x
    for k in range(1, steps + 1):
        X[k] = F @ X[k - 1] + rng.multivariate_normal(np.zeros(4), Q)
    Zpos = X[:, [0, 2]] + rng.normal(0.0, sig_pos, size=(steps + 1, 2))
    Zrr = np.array([h_rangerate(X[k]) for k in range(steps + 1)]) \
        + rng.normal(0.0, sig_rr, size=steps + 1)
    return t, X, Zpos, Zrr


def run_ekf_rr(t, Zpos, Zrr, dt=1.0, q=0.5, sig_pos=40.0, sig_rr=2.0,
               use_rr=True, x0=None, P0=None):
    """EKF over the position (and optionally range-rate) measurements.

    Position update is linear; the range-rate update is the EKF step using
    H_rangerate. Returns (Xhat, Perr) with per-scan position and velocity
    error magnitudes are NOT computed here (caller compares to truth)."""
    F, Q = cv_matrices_2d(dt, q)
    Hp = _H_pos2d()
    Rp = np.diag([sig_pos ** 2, sig_pos ** 2])
    n = len(Zpos)
    if x0 is None:
        x0 = np.array([Zpos[0, 0], 0.0, Zpos[0, 1], 0.0])
    xhat = np.asarray(x0, float).copy()
    P = P0 if P0 is not None else np.diag([sig_pos ** 2, 400.0,
                                           sig_pos ** 2, 400.0])
    Xhat = np.zeros((n, 4))
    Xhat[0] = xhat
    for k in range(1, n):
        # predict
        xhat = F @ xhat
        P = F @ P @ F.T + Q
        # linear position update
        S = Hp @ P @ Hp.T + Rp
        K = P @ Hp.T @ np.linalg.inv(S)
        xhat = xhat + K @ (Zpos[k] - Hp @ xhat)
        P = (np.eye(4) - K @ Hp) @ P
        # range-rate (EKF) update
        if use_rr:
            Hr = H_rangerate(xhat)
            Sr = float((Hr @ P @ Hr.T)[0, 0]) + sig_rr ** 2
            Kr = (P @ Hr.T) / Sr
            xhat = xhat + (Kr * (Zrr[k] - h_rangerate(xhat))).ravel()
            P = (np.eye(4) - Kr @ Hr) @ P
        Xhat[k] = xhat
    return Xhat


def track_errors(X, Xhat):
    """Per-scan (position-error, velocity-error) magnitudes between truth X
    and estimate Xhat (both (n,4), state [px,vx,py,vy])."""
    pe = np.hypot(Xhat[:, 0] - X[:, 0], Xhat[:, 2] - X[:, 2])
    ve = np.hypot(Xhat[:, 1] - X[:, 1], Xhat[:, 3] - X[:, 3])
    return pe, ve


def doppler_benefit(n_mc=200, warmup=10, seed0=0, **kw):
    """Monte-Carlo mean steady-state position/velocity RMSE with and without
    the range-rate measurement. Returns a dict of the four RMSE numbers and
    the percent improvements."""
    pN, vN, pR, vR = [], [], [], []
    for m in range(n_mc):
        t, X, Zp, Zr = simulate_rr_track(seed=seed0 + m, **kw)
        e0p, e0v = track_errors(X, run_ekf_rr(t, Zp, Zr, use_rr=False, **_ekf_kw(kw)))
        e1p, e1v = track_errors(X, run_ekf_rr(t, Zp, Zr, use_rr=True, **_ekf_kw(kw)))
        pN.append(e0p[warmup:].mean()); vN.append(e0v[warmup:].mean())
        pR.append(e1p[warmup:].mean()); vR.append(e1v[warmup:].mean())
    pN, vN, pR, vR = map(lambda a: float(np.mean(a)), (pN, vN, pR, vR))
    return {"pos_norr": pN, "pos_rr": pR, "vel_norr": vN, "vel_rr": vR,
            "pos_pct": 100 * (1 - pR / pN), "vel_pct": 100 * (1 - vR / vN)}


def _ekf_kw(kw):
    """Pass-through of the EKF-relevant kwargs from a simulate_rr_track kw set."""
    return {k: kw[k] for k in ("dt", "q", "sig_pos", "sig_rr") if k in kw}


def radtan_velocity_error(X, Xhat):
    """Split the velocity error into components along and across the line of
    sight to a sensor at the origin. Returns (radial_err, tangential_err),
    each (n,). Range-rate observes the RADIAL component only, so it sharpens
    radial velocity and leaves tangential velocity untouched - the key
    teaching point of section 7.6."""
    px, py = X[:, 0], X[:, 2]
    r = np.hypot(px, py)
    ux, uy = px / r, py / r                     # line-of-sight unit vector
    dvx, dvy = Xhat[:, 1] - X[:, 1], Xhat[:, 3] - X[:, 3]
    er = np.abs(dvx * ux + dvy * uy)            # along LOS
    et = np.abs(-dvx * uy + dvy * ux)           # across LOS
    return er, et


# =====================================================================
# Module 8 - Multi-target association: GNN/assignment, PDA, JPDA
# =====================================================================
# Shared compute for figures/mod08.py and exercises/ex08_multitarget.py.
# The optimal assignment uses scipy (available in the radar env); the greedy
# baseline and the PDA/JPDA updates are hand-rolled. State is [px,vx,py,vy].
from scipy.optimize import linear_sum_assignment as _lsa


# ---- 8.2 assignment ----
def optimal_assign(cost):
    """Optimal (minimum-total-cost) assignment via the Hungarian/Munkres
    algorithm. Returns a row->col array. cost need not be square (scipy
    handles rectangular)."""
    r, c = _lsa(np.asarray(cost, float))
    out = -np.ones(np.asarray(cost).shape[0], dtype=int)
    out[r] = c
    return out


def greedy_assign(cost):
    """Greedy nearest assignment: repeatedly take the cheapest still-available
    (row, col) cell. Fast but suboptimal - it can grab a cheap cell that forces
    an expensive swap later. Returns a row->col array (-1 if a row is unfilled)."""
    C = np.asarray(cost, float)
    nr, nc = C.shape
    row = -np.ones(nr, dtype=int)
    used_c = np.zeros(nc, dtype=bool)
    order = np.dstack(np.unravel_index(np.argsort(C, axis=None), C.shape))[0]
    for r, c in order:
        if row[r] == -1 and not used_c[c]:
            row[r] = c
            used_c[c] = True
    return row


def assign_cost(cost, rowcol):
    """Total cost of a row->col assignment (ignores rows assigned to -1)."""
    C = np.asarray(cost, float)
    return float(sum(C[i, j] for i, j in enumerate(rowcol) if j >= 0))


def assoc_cost_matrix(preds, S_list, meas, pd=0.9, lam=1e-4):
    """Negative-log-likelihood association cost c_ij for measurement j from
    track i: 0.5 * Mahalanobis^2 + 0.5*ln|2*pi*S_i| - ln(pd) (the M6 metric
    plus the Gaussian normalizer). preds: list of predicted measurements
    (2-vectors); S_list: their innovation covariances; meas: list of returns.
    Returns an (n_tracks, n_meas) cost matrix."""
    n, m = len(preds), len(meas)
    C = np.zeros((n, m))
    for i in range(n):
        Si = S_list[i]
        Sinv = np.linalg.inv(Si)
        norm = 0.5 * np.log(np.linalg.det(2 * np.pi * Si)) - np.log(pd)
        for j in range(m):
            nu = np.asarray(meas[j]) - np.asarray(preds[i])
            C[i, j] = 0.5 * float(nu @ Sinv @ nu) + norm
    return C


# ---- 8.3 PDA / GNN single-target update in clutter ----
def _gated(zs, zhat, Sinv, gamma):
    return [z for z in zs if float((z - zhat) @ Sinv @ (z - zhat)) <= gamma]


def gnn_update(xhat, P, zs, H, R, gamma):
    """Hard nearest-neighbor update: pick the gated return of smallest
    Mahalanobis distance and do a standard Kalman update. Coasts if the gate
    is empty. Returns (xhat, P)."""
    zhat = H @ xhat
    S = H @ P @ H.T + R
    Sinv = np.linalg.inv(S)
    K = P @ H.T @ Sinv
    g = _gated(zs, zhat, Sinv, gamma)
    if g:
        d = [float((z - zhat) @ Sinv @ (z - zhat)) for z in g]
        z = g[int(np.argmin(d))]
        xhat = xhat + K @ (z - zhat)
        P = (np.eye(len(xhat)) - K @ H) @ P
    return xhat, P


def pda_update(xhat, P, zs, H, R, pd, lam, gamma):
    """Probabilistic Data Association update. Weight every gated return by its
    association probability beta_i (with beta_0 for 'none detected'), update
    with the combined innovation sum_i beta_i nu_i, and inflate the covariance
    by the spread-of-innovations term. Returns (xhat, P)."""
    zhat = H @ xhat
    S = H @ P @ H.T + R
    Sinv = np.linalg.inv(S)
    K = P @ H.T @ Sinv
    g = _gated(zs, zhat, Sinv, gamma)
    n = len(xhat)
    if not g:
        return xhat, P
    e = np.array([np.exp(-0.5 * float((z - zhat) @ Sinv @ (z - zhat))) for z in g])
    b0 = lam * np.sqrt(np.linalg.det(2 * np.pi * S)) * (1 - pd)
    w = np.concatenate([[b0], pd * e])
    w = w / w.sum()
    beta = w[1:]
    nu = sum(bi * (z - zhat) for bi, z in zip(beta, g))
    xnew = xhat + K @ nu
    Pc = (np.eye(n) - K @ H) @ P
    spread = sum(bi * np.outer(z - zhat, z - zhat) for bi, z in zip(beta, g)) \
        - np.outer(nu, nu)
    Pnew = w[0] * P + (1 - w[0]) * Pc + K @ spread @ K.T
    return xnew, Pnew


def simulate_assoc_track(kind, seed=0, dt=1.0, q=1.0, r=30.0, steps=40,
                         pd=0.9, n_clutter=2.0, gamma=16.0,
                         start=(0.0, 18.0, 0.0, 12.0)):
    """One constant-velocity target in Poisson clutter, tracked by 'gnn' or
    'pda'. n_clutter is the expected number of false returns inside the gate.
    Returns dict with per-scan position error, the truth/estimate tracks, and
    a 'lost' flag (final error > 300 m)."""
    rng = np.random.default_rng(seed)
    F, Q = cv_matrices_2d(dt, q)
    H = _H_pos2d()
    R = r * r * np.eye(2)
    x = np.array(start, float)
    xhat = x + rng.normal(0, [60, 10, 60, 10])
    P = np.diag([60. ** 2, 100., 60. ** 2, 100.])
    lam = n_clutter / (np.pi * gamma * r * r)      # density -> ~n_clutter in gate
    errs = []
    Xt = np.zeros((steps, 4)); Xe = np.zeros((steps, 4))
    for k in range(steps):
        x = F @ x + rng.multivariate_normal(np.zeros(4), Q)
        xhat = F @ xhat; P = F @ P @ F.T + Q
        zhat = H @ xhat
        zs = []
        if rng.random() < pd:
            zs.append(H @ x + rng.normal(0, r, 2))
        for _ in range(rng.poisson(n_clutter)):
            zs.append(zhat + rng.normal(0, np.sqrt(gamma) * r * 0.6, 2))
        if kind == "gnn":
            xhat, P = gnn_update(xhat, P, zs, H, R, gamma)
        else:
            xhat, P = pda_update(xhat, P, zs, H, R, pd, lam, gamma)
        Xt[k] = x; Xe[k] = xhat
        errs.append(np.hypot(xhat[0] - x[0], xhat[2] - x[2]))
    errs = np.array(errs)
    return {"err": errs, "truth": Xt, "est": Xe,
            "rmse": float(errs[steps // 3:].mean()),
            "lost": bool(errs[-1] > 300)}


def assoc_loss_rate(kind, n_mc=300, **kw):
    """Monte-Carlo mean steady RMSE and track-loss fraction for 'gnn'/'pda'."""
    rms, lost = [], 0
    for s in range(n_mc):
        out = simulate_assoc_track(kind, seed=s, **kw)
        rms.append(out["rmse"]); lost += out["lost"]
    return {"rmse": float(np.mean(rms)), "loss": lost / n_mc}


# ---- 8.4 JPDA feasible-event weights (small scenes) ----
def jpda_weights(preds, S_list, meas, pd=0.9, lam=1e-4, gamma=16.0):
    """Joint PDA marginal association weights for a small scene. Enumerate all
    feasible joint events (each measurement to at most one target, each target
    to at most one gated measurement), score each by its probability, and
    marginalize to beta[i, j] = P(measurement j <- target i) and beta0[i] =
    P(target i detected by none). preds/S_list per target; meas the returns."""
    from itertools import product
    n, m = len(preds), len(meas)
    Sinv = [np.linalg.inv(S) for S in S_list]
    # gating: which measurements are valid for each target
    valid = [[j for j in range(m)
              if float((np.asarray(meas[j]) - preds[i]) @ Sinv[i]
                       @ (np.asarray(meas[j]) - preds[i])) <= gamma]
             for i in range(n)]
    gauss = np.zeros((n, m))
    for i in range(n):
        for j in valid[i]:
            nu = np.asarray(meas[j]) - preds[i]
            gauss[i, j] = np.exp(-0.5 * float(nu @ Sinv[i] @ nu)) \
                / np.sqrt(np.linalg.det(2 * np.pi * S_list[i]))
    beta = np.zeros((n, m)); beta0 = np.ones(n); Z = 0.0
    # each target picks a valid measurement or 0 (=missed); measurements unique
    for combo in product(*[[0] + [j + 1 for j in valid[i]] for i in range(n)]):
        picks = [c - 1 for c in combo if c > 0]
        if len(picks) != len(set(picks)):
            continue                                   # a measurement reused
        p = 1.0
        for i, c in enumerate(combo):
            if c == 0:
                p *= (1 - pd)
            else:
                p *= pd * gauss[i, c - 1] / lam
        Z += p
        for i, c in enumerate(combo):
            if c > 0:
                beta[i, c - 1] += p
            else:
                pass
    if Z > 0:
        beta /= Z
    beta0 = 1 - beta.sum(axis=1)
    return beta, beta0


def simulate_two_target(assoc="jpda", seed=0, dt=1.0, q=0.6, r=25.0, steps=40,
                        pd=0.9, n_clutter=1.0, gamma=20.0, sep=900.0, speed=22.0,
                        geometry="crossing"):
    """Two constant-velocity targets on CROSSING paths, in Poisson clutter,
    each tracked by its own Kalman filter with association mode:
      'gnn'  - each target independently takes its nearest gated return
      'pda'  - each target independently PDA-weights its gated returns
      'jpda' - joint weights (jpda_weights) so a shared return is split
    Returns dict: truth [2,steps,4], est [2,steps,4], per-target rmse, and
    'coalesce' = the minimum estimate-to-estimate separation minus the minimum
    truth separation (near 0 = estimates stay apart; large = they merged)."""
    rng = np.random.default_rng(seed)
    F, Q = cv_matrices_2d(dt, q)
    H = _H_pos2d(); R = r * r * np.eye(2)
    tc = steps * dt / 2
    if geometry == "parallel":
        # two targets running parallel a constant distance `sep` apart - the
        # classic coalescence test (independent PDA merges the estimates onto
        # the centreline; JPDA keeps them apart).
        x = [np.array([-speed * tc, speed, +sep / 2, 0.0]),
             np.array([-speed * tc, speed, -sep / 2, 0.0])]
    else:
        # symmetric crossing about the origin at mid-run
        x = [np.array([-speed * tc, speed, -sep / 2, sep / (2 * tc)]),
             np.array([-speed * tc, speed, sep / 2, -sep / (2 * tc)])]
    xh = [xi + rng.normal(0, [40, 5, 40, 5]) for xi in x]
    P = [np.diag([40. ** 2, 50., 40. ** 2, 50.]) for _ in range(2)]
    lam = n_clutter / (np.pi * gamma * r * r)
    Xt = np.zeros((2, steps, 4)); Xe = np.zeros((2, steps, 4))
    for k in range(steps):
        for i in range(2):
            x[i] = F @ x[i] + rng.multivariate_normal(np.zeros(4), Q)
        # predict
        for i in range(2):
            xh[i] = F @ xh[i]; P[i] = F @ P[i] @ F.T + Q
        preds = [H @ xh[i] for i in range(2)]
        S = [H @ P[i] @ H.T + R for i in range(2)]
        # measurements: each target (w.p. pd) + clutter around the pair centroid
        zs = []
        for i in range(2):
            if rng.random() < pd:
                zs.append(H @ x[i] + rng.normal(0, r, 2))
        cen = 0.5 * (preds[0] + preds[1])
        for _ in range(rng.poisson(n_clutter * 2)):
            zs.append(cen + rng.normal(0, np.sqrt(gamma) * r * 0.8, 2))
        # update
        if assoc == "jpda":
            beta, beta0 = jpda_weights(preds, S, zs, pd, lam, gamma)
            for i in range(2):
                Sinv = np.linalg.inv(S[i]); K = P[i] @ H.T @ Sinv
                g = [(j, z) for j, z in enumerate(zs)
                     if float((z - preds[i]) @ Sinv @ (z - preds[i])) <= gamma]
                if g:
                    nu = sum(beta[i, j] * (z - preds[i]) for j, z in g)
                    xh[i] = xh[i] + K @ nu
                    Pc = (np.eye(4) - K @ H) @ P[i]
                    spread = sum(beta[i, j] * np.outer(z - preds[i], z - preds[i])
                                 for j, z in g) - np.outer(nu, nu)
                    P[i] = beta0[i] * P[i] + (1 - beta0[i]) * Pc + K @ spread @ K.T
        else:
            for i in range(2):
                if assoc == "gnn":
                    xh[i], P[i] = gnn_update(xh[i], P[i], zs, H, R, gamma)
                else:
                    xh[i], P[i] = pda_update(xh[i], P[i], zs, H, R, pd, lam, gamma)
        for i in range(2):
            Xt[i, k] = x[i]; Xe[i, k] = xh[i]
    rmse = [float(np.hypot(Xe[i, :, 0] - Xt[i, :, 0],
                           Xe[i, :, 2] - Xt[i, :, 2]).mean()) for i in range(2)]
    est_sep = np.hypot(Xe[0, :, 0] - Xe[1, :, 0], Xe[0, :, 2] - Xe[1, :, 2])
    truth_sep = np.hypot(Xt[0, :, 0] - Xt[1, :, 0], Xt[0, :, 2] - Xt[1, :, 2])
    return {"truth": Xt, "est": Xe, "rmse": rmse,
            "min_est_sep": float(est_sep.min()),
            "min_truth_sep": float(truth_sep.min())}


# =====================================================================
# Module 9 - Coordinate conversions & sensor motion compensation
# =====================================================================
# Shared compute for figures/mod09.py and exercises/ex09_conversion.py. Pure
# numpy trig (no scipy). Angles in radians unless a name says _deg.

def polar_to_cart(r, b):
    """Naive polar-to-Cartesian conversion of a (range, bearing) measurement."""
    return np.array([r * np.cos(b), r * np.sin(b)])


def naive_cov(r, b, sr, sb):
    """Jacobian-linearized converted-measurement covariance (the WRONG,
    inconsistent one that the EKF/naive converter uses). sr, sb are the range
    and bearing measurement standard deviations."""
    J = np.array([[np.cos(b), -r * np.sin(b)],
                  [np.sin(b),  r * np.cos(b)]])
    return J @ np.diag([sr ** 2, sb ** 2]) @ J.T


def unbiased_convert(r, b, sb):
    """Multiplicative unbiased converted measurement (Lerro & Bar-Shalom 1993).
    Multiplies out the e^{-sb^2/2} bias that nonlinear conversion introduces."""
    k = np.exp(sb ** 2 / 2.0)
    return np.array([k * r * np.cos(b), k * r * np.sin(b)])


def unbiased_cov(r, b, sr, sb):
    """Exact covariance of the multiplicative unbiased converted measurement,
    about the true converted position. Verified to give NEES = 2.0 (a 2-D
    measurement) at all angular-noise levels, where naive_cov does not. In a
    filter, r and b are the measured values (the practical CMKF substitution)."""
    k2 = np.exp(sb ** 2)
    e2 = np.exp(-2.0 * sb ** 2)
    rr = r ** 2 + sr ** 2
    xt, yt = r * np.cos(b), r * np.sin(b)
    Exx = k2 * rr * 0.5 * (1 + np.cos(2 * b) * e2)
    Eyy = k2 * rr * 0.5 * (1 - np.cos(2 * b) * e2)
    Exy = k2 * rr * 0.5 * np.sin(2 * b) * e2
    return np.array([[Exx - xt ** 2, Exy - xt * yt],
                     [Exy - xt * yt, Eyy - yt ** 2]])


def nees(samples, mean, cov):
    """Mean normalized estimation-error squared of 2-D samples (shape (2, N))
    about `mean` under `cov`. Consistent = the state dimension (2 here)."""
    d = np.asarray(samples) - np.asarray(mean)[:, None]
    Cinv = np.linalg.inv(cov)
    return float(np.mean(np.einsum('ij,jk,ik->i', d.T, Cinv, d.T)))


# ---- 9.2 frame chain: polar/RUV -> ENU -> ECEF ----
_RE = 6_371_000.0   # spherical-earth radius (m); WGS-84 ellipsoid not modeled


def enu_from_polar(r, az, el):
    """Local East-North-Up from range r, azimuth az (clockwise from North) and
    elevation el (above horizontal). All angles in radians."""
    ce = np.cos(el)
    return np.array([r * ce * np.sin(az),   # East
                     r * ce * np.cos(az),   # North
                     r * np.sin(el)])       # Up


def _enu_to_ecef_rot(lat, lon):
    sl, cl = np.sin(lat), np.cos(lat)
    so, co = np.sin(lon), np.cos(lon)
    return np.array([[-so, -sl * co, cl * co],
                     [ co, -sl * so, cl * so],
                     [0.0,  cl,      sl]])


def ecef_from_enu(enu, lat, lon, h=0.0):
    """ENU (about a sensor at geodetic lat/lon/height, spherical earth) -> ECEF.
    A rotation set by the sensor's lat/lon plus a translation to its ECEF
    position."""
    p0 = (_RE + h) * np.array([np.cos(lat) * np.cos(lon),
                               np.cos(lat) * np.sin(lon),
                               np.sin(lat)])
    return p0 + _enu_to_ecef_rot(lat, lon) @ np.asarray(enu)


def frame_chain_example(r=50_000.0, az_deg=40.0, el_deg=8.0,
                        lat_deg=29.5, lon_deg=-98.5, h=250.0):
    """Carry one target measurement through RUV -> ENU -> ECEF and return the
    intermediate vectors (for the worked-example figure)."""
    az, el = np.deg2rad(az_deg), np.deg2rad(el_deg)
    lat, lon = np.deg2rad(lat_deg), np.deg2rad(lon_deg)
    enu = enu_from_polar(r, az, el)
    ecef = ecef_from_enu(enu, lat, lon, h)
    return {"polar": (r, az_deg, el_deg), "enu": enu, "ecef": ecef,
            "sensor_ecef": ecef_from_enu(np.zeros(3), lat, lon, h)}


# ---- 9.5 sensor motion compensation ----
def simulate_platform_track(compensate=True, steps=40, dt=1.0, sr=40.0,
                            sb_deg=0.5, seed=0, plat_speed=120.0,
                            tgt=(6000.0, 3000.0)):
    """A sensor platform moving along +x measures a (nearly) stationary target
    in its OWN moving frame. 'compensate' adds the platform position back before
    plotting the global track; without it, the reconstructed track drifts by the
    platform's motion. Returns truth, and the reconstructed track."""
    rng = np.random.default_rng(seed)
    sb = np.deg2rad(sb_deg)
    tgt = np.array(tgt, float)
    Xt = np.zeros((steps, 2)); Xr = np.zeros((steps, 2))
    for k in range(steps):
        plat = np.array([-plat_speed * (steps * dt) / 2 + plat_speed * k * dt, 0.0])
        rel = tgt - plat                       # target in the platform frame
        r = np.hypot(*rel) + rng.normal(0, sr)
        b = np.arctan2(rel[1], rel[0]) + rng.normal(0, sb)
        meas_local = polar_to_cart(r, b)       # converted in the platform frame
        Xr[k] = (plat + meas_local) if compensate else meas_local
        Xt[k] = tgt
    return {"truth": Xt, "recon": Xr,
            "rmse": float(np.hypot(Xr[:, 0] - Xt[:, 0],
                                   Xr[:, 1] - Xt[:, 1]).mean())}


# =====================================================================
# Module 10 - Multi-sensor fusion: JDL model & architectures
# =====================================================================
# Shared compute for figures/mod10.py and exercises/ex10_fusion.py. Pure numpy
# (covariance intersection's weight is a 1-D line search; no scipy). All
# covariances are 2x2 position covariances of two local tracks of one target.

def naive_fuse(P1, P2):
    """Bayesian fusion of two estimates ASSUMING INDEPENDENCE (information form,
    cross-covariance taken to be zero): P = (P1^-1 + P2^-1)^-1. This is the
    over-confident fusion when the tracks are actually correlated - it reports
    the SAME covariance regardless of the true correlation."""
    I1 = np.linalg.inv(P1)
    I2 = np.linalg.inv(P2)
    return np.linalg.inv(I1 + I2)


def crosscov(P1, P2, rho):
    """A simple shared-common-error model for the cross-covariance P12 between
    two local tracks of the same target: a fraction rho (0..1) of the geometric-
    mean per-axis standard deviations is common. rho = 0 is independent; rho -> 1
    is a fully shared error. Diagonal here for a clean teaching example."""
    return rho * np.array([[np.sqrt(P1[0, 0] * P2[0, 0]), 0.0],
                           [0.0, np.sqrt(P1[1, 1] * P2[1, 1])]])


def bc_fuse(P1, P2, P12):
    """Bar-Shalom / Campo cross-covariance-AWARE fused covariance for two
    correlated estimates with known cross-covariance P12:
        P = P1 - (P1 - P12) (P1 + P2 - P12 - P12^T)^-1 (P1 - P12)^T .
    This is the true/optimal fused covariance when P12 is known; with P12 = 0 it
    reduces to the naive information-form result."""
    D = P1 + P2 - P12 - P12.T
    A = P1 - P12
    return P1 - A @ np.linalg.inv(D) @ A.T


def ci_fuse(P1, P2, grid=99):
    """Covariance intersection: P^-1 = w P1^-1 + (1-w) P2^-1, with w in [0,1]
    chosen to minimize trace(P). CI is CONSISTENT for ANY unknown correlation
    between the two estimates (never over-confident), at the price of a looser
    (conservative) covariance. Returns (P, w)."""
    I1 = np.linalg.inv(P1)
    I2 = np.linalg.inv(P2)
    best = None
    for w in np.linspace(0.01, 0.99, grid):
        P = np.linalg.inv(w * I1 + (1.0 - w) * I2)
        t = np.trace(P)
        if best is None or t < best[0]:
            best = (t, w, P)
    return best[2], float(best[1])


def sim_fused_nees(P1, P2, rho, fused_cov, weights=(1.0, 1.0),
                   n=200000, seed=3):
    """Monte-Carlo NEES of a fused estimate when the two local tracks are
    actually correlated by `rho`. Draws correlated errors (e1, e2) from the
    joint covariance [[P1, P12],[P12^T, P2]], forms the fused estimate error
        e_f = fused_cov (w1 P1^-1 e1 + w2 P2^-1 e2),
    and scores it against `fused_cov` (the covariance the fuser CLAIMS).
    Consistent = 2.0; > 2 means over-confident (the fuser lies optimistically);
    < 2 means conservative. For NAIVE fusion pass weights = (1, 1); for
    covariance intersection pass weights = (w, 1 - w) with `fused_cov` the CI
    covariance."""
    rng = np.random.default_rng(seed)
    w1, w2 = weights
    P12 = crosscov(P1, P2, rho)
    J = np.block([[P1, P12], [P12.T, P2]])
    L = np.linalg.cholesky(J)
    z = L @ rng.standard_normal((4, n))
    e1, e2 = z[:2], z[2:]
    I1, I2 = np.linalg.inv(P1), np.linalg.inv(P2)
    ef = fused_cov @ (w1 * (I1 @ e1) + w2 * (I2 @ e2))
    Cinv = np.linalg.inv(fused_cov)
    return float(np.mean(np.einsum('ij,jk,ik->i', ef.T, Cinv, ef.T)))


def fusion_example(P1=None, P2=None, rho=0.6):
    """Assemble the worked track-fusion example: two local covariances, the
    naive fused covariance, the CI fused covariance, and the TRUE fused
    covariance at correlation `rho`, plus the naive and CI NEES at that rho.
    Defaults: P1 = diag(4, 9), P2 = diag(9, 4) (m^2)."""
    if P1 is None:
        P1 = np.array([[4.0, 0.0], [0.0, 9.0]])
    if P2 is None:
        P2 = np.array([[9.0, 0.0], [0.0, 4.0]])
    Pn = naive_fuse(P1, P2)
    Pci, w = ci_fuse(P1, P2)
    Ptrue = bc_fuse(P1, P2, crosscov(P1, P2, rho))
    return {"P1": P1, "P2": P2, "naive": Pn, "ci": Pci, "ci_w": w,
            "true": Ptrue, "rho": rho,
            "nees_naive": sim_fused_nees(P1, P2, rho, Pn, weights=(1.0, 1.0)),
            "nees_ci": sim_fused_nees(P1, P2, rho, Pci, weights=(w, 1.0 - w))}


# =====================================================================
# Module 11 - Fusion of multiple radars & multiple angle-only sensors
# =====================================================================
# Shared compute for figures/mod11.py and exercises/ex11_fusion.py. Pure numpy.
# Reuses naive_fuse / ci_fuse from the Module 10 block. Bearings are measured
# clockwise from North (b = atan2(East, North)); positions are (x=East, y=North)
# in metres. Angular arguments named *_deg are in degrees, else radians.

def radar_cov(r, bearing_deg, sr, sth_deg):
    """Converted-measurement position covariance of ONE radar viewing a target
    at range r along `bearing_deg` (from North). Thin in range (std sr), fat in
    cross-range (std r*sth), rotated into world (East, North). This is the
    long-range shape that makes multi-radar geometry pay off: two radars viewing
    from different bearings cross their good range axes."""
    b = np.deg2rad(bearing_deg)
    sth = np.deg2rad(sth_deg)
    u = np.array([np.sin(b), np.cos(b)])      # line-of-sight (range) direction
    v = np.array([np.cos(b), -np.sin(b)])     # cross-range direction
    return (sr ** 2) * np.outer(u, u) + ((r * sth) ** 2) * np.outer(v, v)


def fuse_radars(covs, method="ci"):
    """Fuse a list of position covariances. method='naive' assumes independence
    (information-form sum); method='ci' folds them pairwise with covariance
    intersection (consistent under the unknown track-to-track correlation of
    M10). Returns the fused covariance."""
    P = covs[0]
    for Q in covs[1:]:
        P = naive_fuse(P, Q) if method == "naive" else ci_fuse(P, Q)[0]
    return P


def bearing(sensor, tgt):
    """Bearing (rad, clockwise from North) from `sensor` to `tgt`, both (E, N)."""
    d = np.asarray(tgt, float) - np.asarray(sensor, float)
    return np.arctan2(d[0], d[1])


def bearing_line_intersect(s1, b1, s2, b2):
    """Intersection of two bearing lines: sensor s_i plus t * [sin b_i, cos b_i].
    Returns the (E, N) crossing, or None if the lines are near-parallel."""
    d1 = np.array([np.sin(b1), np.cos(b1)])
    d2 = np.array([np.sin(b2), np.cos(b2)])
    A = np.array([d1, -d2]).T
    if abs(np.linalg.det(A)) < 1e-9:
        return None
    t = np.linalg.solve(A, np.asarray(s2, float) - np.asarray(s1, float))
    return np.asarray(s1, float) + t[0] * d1


def triangulate_cov(sensors, tgt, sth_deg):
    """Cross-fix (triangulation) position covariance from >=2 angle-only sensors
    measuring bearing to `tgt`, each with bearing std sth_deg. Bearing-Jacobian
    information fusion: P = (sum_i H_i^T R_i^-1 H_i)^-1. Its trace/eigenvalues are
    the GDOP: the fix stretches along the bearing lines as the cut angle shrinks."""
    sth = np.deg2rad(sth_deg)
    info = np.zeros((2, 2))
    for s in sensors:
        d = np.asarray(tgt, float) - np.asarray(s, float)
        rng2 = d[0] ** 2 + d[1] ** 2
        # d(bearing)/d(E,N) for bearing = atan2(E, N)
        H = np.array([d[1] / rng2, -d[0] / rng2])
        info += np.outer(H, H) / (sth ** 2)
    return np.linalg.inv(info)


def cut_angle_deg(s1, s2, tgt):
    """Acute angle (deg) at which the two sensors' bearing lines cross at `tgt`.
    Near 90 deg is a strong fix; near 0/180 deg is a weak, stretched fix."""
    a1 = np.asarray(tgt, float) - np.asarray(s1, float)
    a2 = np.asarray(tgt, float) - np.asarray(s2, float)
    c = np.dot(a1, a2) / (np.hypot(*a1) * np.hypot(*a2))
    return float(np.degrees(np.arccos(np.clip(abs(c), 0.0, 1.0))))


def enumerate_intersections(s1, s2, targets):
    """All crossings of sensor-1 bearings with sensor-2 bearings for a list of
    `targets`. Returns a list of dicts {pos, real, i, j}: real = (i == j), i.e.
    the bearing of target i from s1 crossed with the bearing of the SAME target
    from s2. The off-diagonal (i != j) crossings are GHOST targets."""
    b1 = [bearing(s1, t) for t in targets]
    b2 = [bearing(s2, t) for t in targets]
    out = []
    for i in range(len(targets)):
        for j in range(len(targets)):
            p = bearing_line_intersect(s1, b1[i], s2, b2[j])
            if p is not None and p[1] > min(s1[1], s2[1]):  # in front of the pair
                out.append({"pos": p, "real": (i == j), "i": i, "j": j})
    return out


def resolve_with_third(s3, candidates, targets, tol=1500.0):
    """Mark which candidate fixes a THIRD sensor confirms: a candidate is kept if
    s3's bearing to some true target passes within `tol` metres of it. Ghost
    fixes are not on any true bearing from s3, so they are rejected."""
    tb = [bearing(s3, t) for t in targets]
    for c in candidates:
        d = np.asarray(c["pos"], float) - np.asarray(s3, float)
        cb = np.arctan2(d[0], d[1])
        rng = np.hypot(*d)
        # perpendicular miss distance from the nearest true s3-bearing
        miss = min(abs(np.sin(cb - b)) * rng for b in tb)
        c["confirmed"] = bool(miss < tol)
    return candidates


def gdop_field(s1, s2, sth_deg, xs, ys):
    """Fused-position rms (sqrt-trace of the two-sensor triangulation covariance,
    in metres) over a grid of target locations xs x ys, for a fixed sensor pair.
    The 'well' of good accuracy sits where the cut angle is near 90 deg; the field
    blows up where the geometry goes collinear."""
    Z = np.empty((len(ys), len(xs)))
    for a, y in enumerate(ys):
        for b_, x in enumerate(xs):
            P = triangulate_cov([s1, s2], (x, y), sth_deg)
            Z[a, b_] = np.sqrt(np.trace(P))
    return Z


# =====================================================================
# Module 12 - Radar/angle-only fusion & sensor alignment (registration)
# =====================================================================
# Shared compute for figures/mod12.py and exercises/ex12_registration.py.
# Pure numpy. Reuses the Module 11 radar_cov / bearing. Positions (E, N) in
# metres; bearings clockwise from North; angular args *_deg in degrees.

def bearing_info(sensor, tgt, sth_deg):
    """Fisher information (2x2) a single angle-only bearing to `tgt` contributes
    about the target position, for bearing std sth_deg. H = d(bearing)/d(E,N)."""
    sth = np.deg2rad(sth_deg)
    d = np.asarray(tgt, float) - np.asarray(sensor, float)
    rng2 = d[0] ** 2 + d[1] ** 2
    H = np.array([d[1] / rng2, -d[0] / rng2])
    return np.outer(H, H) / (sth ** 2)


def radar_ao_fuse(P_radar, ao_sensors, tgt, sth_deg):
    """Fuse a full radar position covariance P_radar (at `tgt`) with one or more
    non-colocated angle-only sensors' bearings. The passive bearings pin the
    cross-range the radar measures poorly. Returns the fused covariance."""
    info = np.linalg.inv(P_radar)
    for s in np.atleast_2d(ao_sensors):
        info = info + bearing_info(s, tgt, sth_deg)
    return np.linalg.inv(info)


def apply_azimuth_bias(sensor, tgt, bias_deg):
    """Where a sensor with an azimuth (boresight) bias `bias_deg` REPORTS a target
    that truly sits at `tgt`: rotate the line of sight by the bias about the
    sensor, so the reported position is displaced cross-range by ~ R*bias. Returns
    the reported (E, N)."""
    s = np.asarray(sensor, float)
    d = np.asarray(tgt, float) - s
    rng = np.hypot(*d)
    b = np.arctan2(d[0], d[1]) + np.deg2rad(bias_deg)   # biased bearing
    return s + rng * np.array([np.sin(b), np.cos(b)])


def assoc_gate(P1, P2, conf=0.99):
    """Association-gate extent (metres) between two tracks with covariances P1, P2:
    the chi-square (2 dof) gate radius on the RELATIVE-position covariance P1 + P2.
    Two reports gate together as one target when their separation is below this."""
    chi2 = {0.90: 4.605, 0.95: 5.991, 0.99: 9.210}.get(conf, 9.210)
    return float(np.sqrt(chi2 * np.max(np.linalg.eigvalsh(P1 + P2))))


def estimate_bias_ls(true_bearings_deg, measured_bearings_deg):
    """Least-squares azimuth-bias estimate from N common-target observations: the
    mean offset between a biased sensor's measured bearings and the true bearings
    (from an unbiased reference). Returns (estimate_deg, std_of_estimate_deg)."""
    r = np.asarray(measured_bearings_deg, float) - np.asarray(true_bearings_deg, float)
    n = len(r)
    return float(np.mean(r)), float(np.std(r, ddof=1) / np.sqrt(n)) if n > 1 else (float(np.mean(r)), float("nan"))


def simulate_registration(bias_deg=1.5, N=10, r=50000.0, sr=30.0, sth_deg=0.3,
                          look1=-30.0, look2=30.0, seed=0):
    """Full registration workflow for the exercise/figure. Two radars view one
    target at range r; sensor 2 carries an unknown azimuth bias. Returns the true
    target, each sensor's reported position (sensor 2 displaced by the bias), the
    fused split, the association gate, the least-squares bias estimate from N
    common observations, and the corrected (re-merged) sensor-2 report + residual."""
    rng = np.random.default_rng(seed)
    tgt = np.array([0.0, r])                      # true target due north at range r
    s1 = np.array([-r * np.sin(np.deg2rad(-look1)), 0.0]) * 0  # keep sensors at origin-ish
    # place the two sensors so their look angles to the target are look1, look2
    s1 = tgt - r * np.array([np.sin(np.deg2rad(look1)), np.cos(np.deg2rad(look1))])
    s2 = tgt - r * np.array([np.sin(np.deg2rad(look2)), np.cos(np.deg2rad(look2))])
    P1 = radar_cov(r, look1, sr, sth_deg)
    P2 = radar_cov(r, look2, sr, sth_deg)
    rep1 = tgt.copy()
    rep2_biased = apply_azimuth_bias(s2, tgt, bias_deg)
    gate = assoc_gate(P1, P2)
    split = float(np.hypot(*(rep2_biased - rep1))) > gate

    # LS bias estimate from N common targets spread in bearing (seen by both)
    tb = np.linspace(-25, 25, N)                  # true bearings of N common targets
    meas = tb + bias_deg + rng.normal(0, sth_deg, size=N)
    est, est_std = estimate_bias_ls(tb, meas)
    rep2_corr = apply_azimuth_bias(s2, tgt, bias_deg - est)   # remove the estimate
    residual = float(np.hypot(*(rep2_corr - rep1)))
    return {"tgt": tgt, "s1": s1, "s2": s2, "P1": P1, "P2": P2,
            "rep1": rep1, "rep2_biased": rep2_biased, "rep2_corr": rep2_corr,
            "gate": gate, "split": split, "shift": float(np.hypot(*(rep2_biased - rep1))),
            "bias_true": bias_deg, "bias_est": est, "bias_est_std": est_std,
            "residual": residual}


# =====================================================================
# Module 13 - Attribute fusion & performance metrics (capstone)
# =====================================================================
# Shared compute for figures/mod13.py and exercises/ex13_attribute_metrics.py.
# scipy is allowed (M8 precedent) for the OSPA optimal assignment.

# ---- 13.2 Bayesian sequential class probability ----
def bayes_class_update(prior, confusion, declaration):
    """One sequential Bayesian class-probability update. `confusion[i, j]` is
    P(declare class j | true class i); `declaration` is the observed class index.
    Returns the posterior over the true class."""
    prior = np.asarray(prior, float)
    like = np.asarray(confusion, float)[:, declaration]
    post = prior * like
    return post / post.sum()


def bayes_run(true_class, confusion, n, prior=None, seed=0):
    """Simulate `n` noisy declarations from a sensor with the given confusion
    matrix observing `true_class`, and return the history of posteriors, shape
    (n+1, n_classes) including the flat/initial prior."""
    conf = np.asarray(confusion, float)
    k = conf.shape[0]
    rng = np.random.default_rng(seed)
    p = np.full(k, 1.0 / k) if prior is None else np.asarray(prior, float)
    hist = [p.copy()]
    for _ in range(n):
        d = rng.choice(k, p=conf[true_class])
        p = bayes_class_update(p, conf, d)
        hist.append(p.copy())
    return np.array(hist)


def mean_decls_to_confidence(true_class, confusion, thresh=0.95, n=25,
                             trials=500):
    """Average number of declarations for the true class posterior to reach
    `thresh`, over `trials` Monte-Carlo runs (a classifier-quality summary)."""
    out = []
    for s in range(trials):
        h = bayes_run(true_class, confusion, n, seed=s)[:, true_class]
        idx = np.argmax(h >= thresh)
        out.append(idx if h[idx] >= thresh else n)
    return float(np.mean(out))


# ---- 13.3 Dempster-Shafer evidential fusion ----
def dempster_combine(m1, m2):
    """Combine two mass functions by Dempster's rule. Each is a dict mapping a
    frozenset of class labels (a focal element) to a mass. Returns (combined
    masses, conflict K). High K (near 1) is the Zadeh high-conflict regime where
    the rule's normalization gives a counterintuitive result."""
    out = {}
    K = 0.0
    for a, va in m1.items():
        for b, vb in m2.items():
            inter = a & b
            if inter:
                out[inter] = out.get(inter, 0.0) + va * vb
            else:
                K += va * vb
    s = sum(out.values())
    if s > 0:
        out = {k: v / s for k, v in out.items()}
    return out, float(K)


def belief(mass, hyp):
    """Belief in `hyp` (a frozenset): total mass on focal elements that are
    subsets of hyp (the lower bound of the probability interval)."""
    hyp = frozenset(hyp)
    return float(sum(v for a, v in mass.items() if a <= hyp))


def plausibility(mass, hyp):
    """Plausibility of `hyp`: total mass on focal elements that intersect hyp
    (the upper bound of the probability interval)."""
    hyp = frozenset(hyp)
    return float(sum(v for a, v in mass.items() if a & hyp))


# ---- 13.5 / 13.6 performance metrics ----
def ospa(X, Y, c=100.0, p=2.0):
    """Optimal sub-pattern assignment distance between truth set X and estimate
    set Y (each an array of position rows). Returns (total, localization,
    cardinality). Cutoff c bounds any single miss; order p. Captures BOTH
    localization error and cardinality (missed / spurious tracks)."""
    from scipy.optimize import linear_sum_assignment
    X = np.atleast_2d(X) if len(X) else np.empty((0, 2))
    Y = np.atleast_2d(Y) if len(Y) else np.empty((0, 2))
    m, n = len(X), len(Y)
    if m == 0 and n == 0:
        return 0.0, 0.0, 0.0
    if m == 0 or n == 0:
        return c, 0.0, c
    if m > n:                       # ensure m <= n
        X, Y, m, n = Y, X, n, m
    D = np.minimum(c, np.linalg.norm(X[:, None, :] - Y[None, :, :], axis=2))
    r, col = linear_sum_assignment(D ** p)
    loc = (D[r, col] ** p).sum()
    card = (c ** p) * (n - m)
    total = ((loc + card) / n) ** (1.0 / p)
    return float(total), float((loc / n) ** (1.0 / p)), float((card / n) ** (1.0 / p))


def track_purity(labels):
    """Fraction of a track's assignments spent on its dominant truth label.
    1.0 = a clean track; lower = swaps/contamination. `labels` is the sequence
    of truth-target ids the track was associated with over its life."""
    labels = np.asarray(labels)
    if len(labels) == 0:
        return 1.0
    return float(np.max(np.bincount(labels)) / len(labels))


def fragmentation(labels):
    """Number of contiguous segments a single truth is broken into by a track
    set: 1 = ideal (one track covers it), >1 = fragmented. `labels` is the
    sequence of TRACK ids assigned to one truth over time (-1 = uncovered)."""
    labels = np.asarray(labels)
    seg = 0
    prev = None
    for x in labels:
        if x != -1 and x != prev:
            seg += 1
        prev = x
    return int(seg)


def default_confusion(k=3, correct=0.70):
    """A simple symmetric k-class confusion matrix: `correct` on the diagonal,
    the remainder split evenly off-diagonal. Larger `correct` = sharper sensor."""
    off = (1.0 - correct) / (k - 1)
    M = np.full((k, k), off)
    np.fill_diagonal(M, correct)
    return M
