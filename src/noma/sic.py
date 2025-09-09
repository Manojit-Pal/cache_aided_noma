# src/noma/sic.py
def sinr_weak_user(P_tx, p_weak, gain_w, p_strong, noise):
    """
    SINR for weak user when decoding own signal (treats strong user's signal as interference).
    """
    num = P_tx * p_weak * gain_w
    den = P_tx * p_strong * gain_w + noise
    return num / den

def sinr_strong_decode_weak(P_tx, p_weak, gain_s, p_strong, noise):
    """
    SINR at strong user when decoding weak user's signal (before SIC).
    """
    num = P_tx * p_weak * gain_s
    den = P_tx * p_strong * gain_s + noise
    return num / den

def sinr_strong_after_sic(P_tx, p_strong, gain_s, noise, residual_from_weak=0.0):
    """
    SINR at strong user to decode its own signal after (imperfect) SIC.
    residual_from_weak is the residual interference power (linear).
    """
    num = P_tx * p_strong * gain_s
    den = noise + residual_from_weak
    return num / den
