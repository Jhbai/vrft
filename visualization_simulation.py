import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import lfilter
from fodpt import referenceModel

class PlantSimulation:
    def __init__(self, K: float, tau: float, delay_time: float, ts: float = 1.0):
        self.K = K
        self.tau = tau
        self.ts = ts
        self.d = int(round(delay_time / ts))
        self.a = np.exp(-ts / tau)
        
        self.y_prev = 0.0
        self.u_buffer = np.zeros(self.d + 1) # delay queue for FIFO
        self.buffer_idx = 0

    def step(self, u: float, dist: float = 0.0) -> float:
        # 當前 u 推入 delay queue
        self.u_buffer[self.buffer_idx] = u
        
        # 取出 d 拍以前的控制量
        delayed_idx = (self.buffer_idx - self.d) % len(self.u_buffer)
        u_delayed = self.u_buffer[delayed_idx]
        
        # 更新環形緩衝區指標
        self.buffer_idx = (self.buffer_idx + 1) % len(self.u_buffer)
        
        # 一階濾波器差分方程更新狀態，並疊加擾動
        y_k = self.a * self.y_prev + (1 - self.a) * self.K * u_delayed + dist
        self.y_prev = y_k
        return y_k

# 閉迴路資料產生
TS = 1.0
PLANT_K = 1.2
PLANT_TAU = 8.0
PLANT_DELAY = 3.0
N_SAMPLES = 1000
np.random.seed(42)
u_open = np.repeat(np.random.randn(N_SAMPLES // 20), 20) * 10
u_open = np.clip(u_open, -15, 15)

plant_open = PlantSimulation(K=PLANT_K, tau=PLANT_TAU, delay_time=PLANT_DELAY, ts=TS)
y_open = np.zeros(N_SAMPLES)
for i in range(N_SAMPLES):
    sensor_noise = np.random.normal(0, 0.05)
    y_open[i] = plant_open.step(u_open[i], dist=sensor_noise)


# 設定期望的閉迴路模型 (要求比原系統快一點)
ref_model = referenceModel(tau=5.0, delay_time=PLANT_DELAY, ts=TS)

# 計算虛擬參考訊號與虛擬誤差 (利用時間平移非因果逆運算)
r_bar = ref_model.inv_predict(y_open)
n_valid = len(r_bar)
e_bar = r_bar - y_open[:n_valid]
u_valid = u_open[:n_valid]

# 定義離散位置型 PID 控制器特徵矩陣
P = e_bar
I = np.cumsum(e_bar) * TS
D = np.r_[0.0, np.diff(e_bar)] / TS
Phi = np.column_stack([P, I, D])

# 透過解析解 (線性最小平方法) 算出全局最佳 PID 參數
theta, residuals, rank, s = np.linalg.lstsq(Phi, u_valid, rcond=None)
Kp, Ki, Kd = theta
print(f"VRFT 擬合結果: Kp = {Kp:.3f}, Ki = {Ki:.3f}, Kd = {Kd:.3f}")
u_hat = Phi @ theta


# 持續干擾測試
N_CLOSED = 800
r_closed = np.ones(N_CLOSED) * 50.0  # Setpoint 設為 50

# 模擬「多股不同廢水持續進來」的複合型干擾
# 1. 基礎水質微小波動
dist_continuous = np.random.normal(0, 0.2, N_CLOSED)
# 2. 緩慢的環境/濃度飄移
dist_continuous += np.sin(np.linspace(0, 4*np.pi, N_CLOSED)) * 2.0
# 3. 突發的不同廢水注入 (多個 Step 干擾)
dist_continuous[200:] += 8.0   # 第200秒注入鹼性廢水
dist_continuous[450:] -= 15.0  # 第450秒注入強酸廢水
dist_continuous[650:] += 5.0   # 第650秒注入另一股廢水

plant_closed = PlantSimulation(K=PLANT_K, tau=PLANT_TAU, delay_time=PLANT_DELAY, ts=TS)
y_closed = np.zeros(N_CLOSED)
u_closed_history = np.zeros(N_CLOSED)

# PID 控制器狀態
err_sum = 0.0
err_prev = 0.0

for k in range(N_CLOSED):
    if k == 0:
        y_k = 0
    else:
        y_k = y_closed[k-1]
        
    # 計算誤差
    e_k = r_closed[k] - y_k
    err_sum += e_k * TS
    e_diff = (e_k - err_prev) / TS
    err_prev = e_k
    
    # PID 計算控制輸出
    u_k = Kp * e_k + Ki * err_sum + Kd * e_diff
    u_k = np.clip(u_k, 0, 100)
    u_closed_history[k] = u_k
    
    y_closed[k] = plant_closed.step(u_k, dist=dist_continuous[k])


fig = plt.figure(figsize=(14, 10))
ax1 = plt.subplot(2, 1, 1)
time_open = np.arange(n_valid) * TS
ax1.plot(time_open, u_valid, label="Actual Input $u$ (PRBS)", alpha=0.7, color='gray')
ax1.plot(time_open, u_hat, label="VRFT Predicted $\hat{u}$", linestyle='--', color='blue')
ax1.set_title("Step 4: VRFT Open-Loop Sanity Check (MSE Loss Fitting)")
ax1.set_ylabel("Control Signal")
ax1.legend()
ax1.grid(True)

ax2 = plt.subplot(2, 1, 2)
time_closed = np.arange(N_CLOSED) * TS
ax2.plot(time_closed, r_closed, 'k--', label="Setpoint (Target pH/Level)")
ax2.plot(time_closed, y_closed, 'r-', label="System Output $y$ (Actual)", linewidth=2)

ax2_dist = ax2.twinx()
ax2_dist.fill_between(time_closed, dist_continuous, 0, color='purple', alpha=0.15, label="Wastewater Disturbance")
ax2_dist.set_ylabel("Disturbance Magnitude", color='purple')

ax2.set_title("Step 5: Closed-Loop Control Under Continuous Wastewater Disturbances")
ax2.set_xlabel("Time (s)")
ax2.set_ylabel("Process Variable")
ax2.legend(loc='upper left')
ax2_dist.legend(loc='lower right')
ax2.grid(True)

plt.tight_layout()
plt.show()
