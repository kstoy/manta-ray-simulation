import numpy as np

import ballstate as bs
import rodstate as rs

# ===============================================================
# Broadphase helper (uniform grid in XY to get neighbor pairs)
# ===============================================================

def grid_pairs_xy(positions, radii, cell_size, margin):
    """
    Simple uniform grid broadphase on the horizontal plane. Returns candidate (i,j) pairs.
    """
    N = positions.shape[0]
    cells = {}
    for i in range(N):
        x = positions[i, 0]; y = positions[i, 1]
        kx = int(np.floor(x/cell_size))
        ky = int(np.floor(y/cell_size))
        cells.setdefault((kx,ky), []).append(i)

    pairs = []
    neighbors = [(-1,-1),(-1,0),(-1,1),
                 ( 0,-1),( 0,0),( 0,1),
                 ( 1,-1),( 1,0),( 1,1)]
    for (kx,ky), idxs in cells.items():
        for dx,dy in neighbors:
            lst = cells.get((kx+dx, ky+dy))
            if lst is None: continue
            for i in idxs:
                for j in lst:
                    if j <= i: continue
                    sumR = radii[i] + radii[j] + margin
                    dx_ = positions[i, 0] - positions[j, 0]
                    dy_ = positions[i, 1] - positions[j, 1]
                    if dx_*dx_ + dy_*dy_ <= sumR*sumR:
                        pairs.append((i,j))
    return pairs


# ===============================================================
# XPBD step (position-level constraints) + friction (velocity-level)
# ===============================================================

def step(
    ballsstate: bs.BallsState,
    rodsstate: rs.RodsState,
    dt=1/240,
    gravity=9.81,
    mu_s=0.45,
    mu_k=0.45,
    compliance_n=0,  # 0 → hard constraints; >0 (e.g., 1e-8) allows tiny softness
    num_pos_iters=18,
    substeps=2,
    pair_margin=0.08,   # recruit near neighbors (reduces late/large penetrations)
    use_grid_broadphase=False,
    linear_damping=0.01,
):
    """
    Performs one *frame* worth of simulation by splitting into 'substeps'.
    Each substep:
      1) predict positions with semi-implicit Euler,
      2) XPBD-projection to remove penetration (terrain + ball-ball),
      3) recompute velocities from corrected positions,
      4) apply friction impulses at contact points (with spin).
    """
    assert dt > 0 and substeps >= 1
    substeps = int(substeps)
    h = dt / substeps

    gvec = np.zeros(3)
    gvec[2] = -abs(gravity)

    #worst_pen = 0.0  # debug metric

    for _ in range(substeps):
        # -------- Predict (semi-implicit) --------
        ballsstate.v += gvec * h
        if linear_damping > 0.0:
            damp = 1.0 / (1.0 + linear_damping)
            ballsstate.v *= damp
            ballsstate.w *= damp
        r0 = ballsstate.r.copy()
        ballsstate.r += h * ballsstate.v

        # -------- Build contact candidates (once per substep) --------
        surf_idx = []
        for i in range(ballsstate.N):
            x, y, z = ballsstate.r[i]
            z_s, dfx, dfy = rodsstate.surfacejet(x,y)
            ntil = np.array( [ -dfx, -dfy, 1.0 ] )
            nlen = float(np.linalg.norm(ntil))
            if nlen < 1e-12: continue
            C = (z - z_s)/nlen - ballsstate.R[i]   # >=0 ok; <0 penetration
            if C <= pair_margin:
                surf_idx.append(i)

        if use_grid_broadphase:
            cell = 2.0 * float(np.max(ballsstate.R)) + pair_margin
            pair_idx = grid_pairs_xy(ballsstate.r, ballsstate.R, cell_size=cell, margin=pair_margin)
        else:
            pair_idx = []
            for i in range(ballsstate.N):
                for j in range(i+1, ballsstate.N):
                    d = ballsstate.r[i] - ballsstate.r[j]
                    if np.dot(d,d) <= (ballsstate.R[i] + ballsstate.R[j] + pair_margin)**2:
                        pair_idx.append((i,j))

        # -------- XPBD normal constraints (position-level) --------
        lamN_surface = np.zeros(len(surf_idx))
        lamN_pairs   = np.zeros(len(pair_idx))
        alpha = compliance_n / (h*h)  # XPBD scaled compliance

        for _it in range(num_pos_iters):
            # Surface contacts
            for si, i in enumerate(surf_idx):
                x, y, z = ballsstate.r[i]
                [z_s, dfx, dfy] = rodsstate.surfacejet(x,y)
                n = np.array( [-dfx, -dfy, 1.0 ] )
                nlen = float(np.linalg.norm(n))
                if nlen < 1e-12: continue
                n /= nlen

                # Use unscaled C for simple math: C = (z - z_s) - R*nlen
                C_unscaled = (z - z_s) - ballsstate.R[i]*nlen
                if C_unscaled < 0.0:
                    w = ballsstate.inv_m[i]
                    if w <= 0.0: continue
                    # Δλ = -(C/nlen + α λ)/(w + α)
                    C_norm = C_unscaled / nlen
                    denom = w + alpha
                    dLam = -(C_norm + alpha * lamN_surface[si]) / max(denom, 1e-12)
                    lam_new = max(0.0, lamN_surface[si] + dLam)
                    dLam = lam_new - lamN_surface[si]
                    lamN_surface[si] = lam_new
                    ballsstate.r[i] += (dLam * w) * n
                    #worst_pen = max(worst_pen, -C_norm)

            # Ball–ball contacts
            for pi, (i, j) in enumerate(pair_idx):
                d = ballsstate.r[i] - ballsstate.r[j]
                dist = float(np.linalg.norm(d))
                sumR = ballsstate.R[i] + ballsstate.R[j]
                if dist < 1e-12:
                    n = np.array([1.0, 0.0, 0.0])  # arbitrary
                    C = -sumR
                else:
                    n = d / dist
                    C = dist - sumR   # >=0 ok; <0 penetration

                if C < 0.0:
                    wi = ballsstate.inv_m[i]; wj = ballsstate.inv_m[j]
                    w = wi + wj
                    if w <= 0.0: continue
                    denom = w + alpha
                    dLam = -(C + alpha * lamN_pairs[pi]) / max(denom, 1e-12)
                    lam_new = max(0.0, lamN_pairs[pi] + dLam)
                    dLam = lam_new - lamN_pairs[pi]
                    lamN_pairs[pi] = lam_new
                    ballsstate.r[i] += (dLam * wi) * n
                    ballsstate.r[j] -= (dLam * wj) * n
                    #worst_pen = max(worst_pen, -C)

        # -------- Update velocities from corrected positions --------
        ballsstate.v = (ballsstate.r - r0) / h

"""         # -------- Friction (velocity-level) using λ_n/h as normal impulse --------
        # Terrain friction with spin (rolling)
        for si, i in enumerate(surf_idx):
            x, y, z = ballsstate.r[i]
            z_s, dfx, dfy = rodsstate.surfacejet(x, y)
            n = np.array( [ -dfx, -dfy ,1.0 ] )
            n_len = float(np.linalg.norm(n))
            if n_len < 1e-12: continue
            n /= n_len

            ri = -ballsstate.R[i] * n
            vc = ballsstate.v[i] + np.cross(ballsstate.w[i], ri)
            vt_vec = vc - np.dot(vc, n) * n
            vt = float(np.linalg.norm(vt_vec))

            Jn = lamN_surface[si] / h  # normal impulse this substep
            if vt > 0.0 and Jn > 0.0:
                kt = ballsstate.inv_m[i] + (ballsstate.R[i]*ballsstate.R[i]) * ballsstate.inv_I[i]  # tangent effective mass
                jt_stick = -(vt_vec / max(kt, 1e-12))
                if np.linalg.norm(jt_stick) <= mu_s * Jn:
                    jt = jt_stick
                else:
                    jt = -mu_k * Jn * (vt_vec / vt)
                ballsstate.v[i] += jt * ballsstate.inv_m[i]
                ballsstate.w[i] += np.cross(ri, jt) * ballsstate.inv_I[i]

        # Ball-ball friction (with spin)
        for pi, (i, j) in enumerate(pair_idx):
            d = ballsstate.r[i] - ballsstate.r[j]
            dist = float(np.linalg.norm(d))
            if dist < 1e-12:
                n = np.array([1.0, 0.0, 0.0])
            else:
                n = d / dist
            ri = -ballsstate.R[i] * n
            rj =  ballsstate.R[j] * n

            vc_i = ballsstate.v[i] + np.cross(ballsstate.w[i], ri)
            vc_j = ballsstate.v[j] + np.cross(ballsstate.w[j], rj)
            vc = vc_i - vc_j
            vt_vec = vc - np.dot(vc, n) * n
            vt = float(np.linalg.norm(vt_vec))

            Jn = lamN_pairs[pi] / h
            if vt > 0.0 and Jn > 0.0:
                kt = (ballsstate.inv_m[i] + ballsstate.inv_m[j]) \
                   + (ballsstate.R[i]*ballsstate.R[i]) * ballsstate.inv_I[i] \
                   + (ballsstate.R[j]*ballsstate.R[j]) * ballsstate.inv_I[j]
                jt_stick = -(vt_vec / max(kt, 1e-12))
                if np.linalg.norm(jt_stick) <= mu_s * Jn:
                    jt = jt_stick
                else:
                    jt = -mu_k * Jn * (vt_vec / vt)

                ballsstate.v[i] +=  jt * ballsstate.inv_m[i]
                ballsstate.v[j] += -jt * ballsstate.inv_m[j]
                ballsstate.w[i] += np.cross(ri,  jt) * ballsstate.inv_I[i]
                ballsstate.w[j] += np.cross(rj, -jt) * ballsstate.inv_I[j] """


