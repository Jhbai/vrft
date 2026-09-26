from scipy.optimize import Bounds, LinearConstraint, lsq_linear, minimize
def loss(theta):
    res = Phi @ theta - u_valid
    return 0.5 * np.dot(res, res)

def grad(theta):
    return Phi.T @ (Phi @ theta - u_valid)
  
def optimize(y: np.ndarray, u: np.ndarray, tau: float = 5.0, delay_time: float = 3.0, ts: float = 1.0, u_max: float = 100, lb: list = [0.0, 0.0, 0.0] ,ub: list = [np.inf, np.inf, np.inf]) -> dict:
    ref_model = referenceModel(tau=tau, delay_time=delay_time, ts=ts)
    r_bar = ref_model.inv_predict(y)
    n_valid = len(r_bar)
    e_bar = r_bar - y[:n_valid]
    u_valid = u[:n_valid]

    P = e_bar
    I = np.cumsum(e_bar) * ts
    D = np.r_[0.0, np.diff(e_bar)] / ts
    Phi = np.column_stack([P, I, D])

    param_bounds = Bounds(lb = lb, ub = ub)
    output_constraint = LinearConstraint(Phi, lb=0.0, ub=np.full(n_valid, u_max))

    res = minimize(loss, x0=[1.0, 0.1, 0.01], jac=grad, bounds=param_bounds, constraints=output_constraint, method="trust-constr")
    theta = res.x
    mse = np.mean((Phi @ theta - u_valid) ** 2)

    return {"Kp": theta[0], "Ki": theta[1], "Kd": theta[2], "MSE": mse}
