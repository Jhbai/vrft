import numpy as np
from fopdt import referenceModel
def optimize(y: np.ndarray, u: np.ndarray, tau: float = 5.0, delay_time: float = 3.0, ts: float = 1.0) -> dict:
    # 建立虛擬參考訊號 r_bar
    ref_model = referenceModel(tau=tau, delay_time=delay_time, ts=ts)
    r_bar = ref_model.inv_predict(y)
    
    # 截斷有效長度並計算誤差 e_bar
    n_valid = len(r_bar)
    e_bar = r_bar - y[:n_valid]
    u_valid = u[:n_valid]
    
    # 建立 PID 特徵矩陣 Phi = [P, I, D]
    P = e_bar
    I = np.cumsum(e_bar) * ts
    D = np.r_[0.0, np.diff(e_bar)] / ts
    Phi = np.column_stack([P, I, D])
    
    # 求解最佳參數
    theta, _, _, _ = np.linalg.lstsq(Phi, u_valid, rcond=None)
    
    return theta # {"Kp": theta[0], "Ki": theta[1], "Kd": theta[2]}
