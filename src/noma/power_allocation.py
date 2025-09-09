# src/noma/power_allocation.py
import numpy as np

def feasible_for_weak(p_w, P, gain_w, N0, T):
    """Check weak user's SINR condition for given p_w."""
    p_s = 1.0 - p_w
    num = P * p_w * gain_w
    den = P * p_s * gain_w + N0
    return (num / den) >= T

def allocate_power_gridsearch(gain_w, gain_s, cfg, grid_points=101):
    """
    Grid search allocator (existing). Returns p_w, p_s, feasible_flag, info.
    """
    P = cfg.TX_POWER
    N0 = cfg.NOISE_POWER
    zeta = cfg.SIC_IMPERFECTION
    T = 2 ** (cfg.TARGET_RATE_BPS) - 1

    eps = 1e-4
    p_ws = np.linspace(eps, 1.0 - eps, grid_points)

    best = None
    best_score = -1
    best_info = {}

    for p_w in p_ws:
        p_s = 1.0 - p_w
        # weak feasibility
        num_w = P * p_w * gain_w
        den_w = P * p_s * gain_w + N0
        sinr_w = num_w / den_w
        ok_weak = sinr_w >= T

        # strong decode weak
        num_sdec = P * p_w * gain_s
        den_sdec = P * p_s * gain_s + N0
        sinr_sdec = num_sdec / den_sdec
        ok_s_decode = sinr_sdec >= T

        # residual for strong after SIC
        if ok_s_decode:
            residual = zeta * (P * p_w * gain_s)
        else:
            residual = P * p_w * gain_s
        sinr_s_after = (P * p_s * gain_s) / (N0 + residual)
        ok_strong = sinr_s_after >= T

        score = int(ok_weak) + int(ok_strong)
        sum_sinr = sinr_w + sinr_s_after

        if (score > best_score) or (score == best_score and (best_info.get("sum_sinr", -1) < sum_sinr)):
            best_score = score
            best = (p_w, p_s)
            best_info = {
                "ok_weak": ok_weak,
                "ok_s_decode": ok_s_decode,
                "ok_strong": ok_strong,
                "sinr_w": sinr_w,
                "sinr_sdec": sinr_sdec,
                "sinr_s_after": sinr_s_after,
                "sum_sinr": sum_sinr,
            }
            if score == 2:
                break

    feasible = best_info.get("ok_weak", False) and best_info.get("ok_strong", False)
    return best[0], best[1], feasible, best_info

def allocate_power_closedform(gain_w, gain_s, cfg):
    """
    Closed-form feasibility check and allocation for a NOMA pair.
    Derivation (carefully):
      - Weak user requirement:
        (P*p_w*g_w) / (P*(1-p_w)*g_w + N0) >= T
        => p_w >= T/(1+T) * (1 + N0/(P*g_w))

      - Strong user decoding weak requirement (to enable SIC):
        (P*p_w*g_s) / (P*(1-p_w)*g_s + N0) >= T
        => p_w >= T/(1+T) * (1 + N0/(P*g_s))

      - Strong user after SIC requirement (assuming successful decode):
        (P*(1-p_w)*g_s) / (N0 + zeta * P * p_w * g_s) >= T
        => 1 - p_w >= T*(N0 + zeta*P*p_w*g_s)/(P*g_s)
        => p_w <= [1 - T*N0/(P*g_s)] / (1 + T*zeta)

      Combine lower bounds and upper bound for feasible p_w.
    Returns: p_w, p_s, feasible_flag, info_dict
    """
    P = cfg.TX_POWER
    N0 = cfg.NOISE_POWER
    zeta = cfg.SIC_IMPERFECTION
    T = 2 ** (cfg.TARGET_RATE_BPS) - 1

    # avoid division by zero / tiny gains
    eps = 1e-12
    g_w = max(gain_w, eps)
    g_s = max(gain_s, eps)

    # lower bounds from weak and strong decoding weak
    lower_w = (T / (1.0 + T)) * (1.0 + (N0 / (P * g_w)))
    lower_sdecode = (T / (1.0 + T)) * (1.0 + (N0 / (P * g_s)))
    lower_bound = max(lower_w, lower_sdecode)

    # upper bound from strong user's own SINR after (imperfect) SIC
    # derived: p_w <= [1 - T*N0/(P*g_s)] / (1 + T*zeta)
    numerator = 1.0 - (T * N0) / (P * g_s)
    denom = 1.0 + T * zeta
    upper_bound = numerator / denom

    # clamp into [0,1]
    lower_bound_clamped = max(lower_bound, 0.0)
    upper_bound_clamped = min(upper_bound, 1.0 - 1e-6)

    info = {
        "lower_w": lower_w,
        "lower_sdecode": lower_sdecode,
        "lower_bound": lower_bound_clamped,
        "upper_bound": upper_bound_clamped,
        "T": T,
        "g_w": g_w,
        "g_s": g_s,
        "zeta": zeta,
    }

    feasible = (lower_bound_clamped <= upper_bound_clamped) and (lower_bound_clamped < 1.0)
    if feasible:
        # choose a p_w inside feasible interval; pick the lower_bound (gives weaker user just enough)
        p_w = min(max(lower_bound_clamped, 0.001), 0.999)
        # ensure also below upper bound
        if p_w > upper_bound_clamped:
            p_w = (lower_bound_clamped + upper_bound_clamped) / 2.0
        p_s = 1.0 - p_w
    else:
        # fallback: choose a heuristic (give more power to weak user)
        p_w = min(max(lower_bound_clamped, 0.5), 0.99)
        p_s = 1.0 - p_w

    # final feasibility check (for reporting)
    # compute SINRs with chosen p_w
    num_w = P * p_w * g_w
    den_w = P * p_s * g_w + N0
    sinr_w = num_w / den_w

    # strong decode weak
    num_sdec = P * p_w * g_s
    den_sdec = P * p_s * g_s + N0
    sinr_sdec = num_sdec / den_sdec
    ok_s_decode = sinr_sdec >= T

    # residual after SIC
    residual = zeta * (P * p_w * g_s) if ok_s_decode else P * p_w * g_s
    sinr_s_after = (P * p_s * g_s) / (N0 + residual)

    ok_weak = sinr_w >= T
    ok_strong = sinr_s_after >= T

    info.update({
        "p_w": p_w,
        "p_s": p_s,
        "sinr_w": sinr_w,
        "sinr_sdec": sinr_sdec,
        "sinr_s_after": sinr_s_after,
        "ok_weak": ok_weak,
        "ok_s_decode": ok_s_decode,
        "ok_strong": ok_strong,
    })

    feasible_final = ok_weak and ok_strong
    return p_w, p_s, feasible_final, info
