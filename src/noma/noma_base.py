# src/noma/noma_base.py
import numpy as np
from .sic import sinr_weak_user, sinr_strong_decode_weak, sinr_strong_after_sic

def sinr_threshold_from_rate(rate_bps):
    return 2 ** (rate_bps) - 1

def simulate_noma_pair(gain_weak, gain_strong, cfg, p_w=None, p_s=None):
    """
    Simulate NOMA transmission for a two-user pair.
    If p_w/p_s provided, use them; otherwise fall back to cfg POWER_COEFFs.
    Returns:
        (weak_success, strong_success, sinr_w, sinr_s_decode_w, sinr_s_after)
    """
    P = cfg.TX_POWER
    N0 = cfg.NOISE_POWER
    if p_w is None or p_s is None:
        p_w = cfg.POWER_COEFF_WEAK
        p_s = cfg.POWER_COEFF_STRONG
    zeta = cfg.SIC_IMPERFECTION
    sinr_th = sinr_threshold_from_rate(cfg.TARGET_RATE_BPS)

    # Weak user SINR
    sinr_w = sinr_weak_user(P, p_w, gain_weak, p_s, N0)
    weak_success = sinr_w >= sinr_th

    # Strong user's ability to decode weak user's signal (before SIC)
    sinr_s_decode_w = sinr_strong_decode_weak(P, p_w, gain_strong, p_s, N0)
    can_decode_weak = sinr_s_decode_w >= sinr_th

    # residual interference depending on decoding success
    if can_decode_weak:
        residual = zeta * (P * p_w * gain_strong)
    else:
        residual = P * p_w * gain_strong

    sinr_s_after = sinr_strong_after_sic(P, p_s, gain_strong, N0, residual)
    strong_success = sinr_s_after >= sinr_th

    return weak_success, strong_success, sinr_w, sinr_s_decode_w, sinr_s_after
