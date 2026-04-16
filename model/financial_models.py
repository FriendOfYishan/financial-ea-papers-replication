import numpy as np
import pandas as pd
from scipy import optimize
from scipy.stats import norm
import warnings
warnings.filterwarnings('ignore')

class CAPM:
    """
    资本资产定价模型 (Capital Asset Pricing Model)
    
    模型形式:
    E[R_i] = R_f + beta_i * (E[R_m] - R_f)
    
    其中:
    - R_i: 资产收益率
    - R_f: 无风险收益率
    - R_m: 市场收益率
    - beta_i: 资产的系统性风险系数
    """
    
    def __init__(self):
        self.beta = None
        self.alpha = None
        self.r_squared = None
    
    def fit(self, returns, market_returns, risk_free_rate=0):
        """
        拟合CAPM模型
        
        参数:
        returns: 资产收益率序列
        market_returns: 市场收益率序列
        risk_free_rate: 无风险收益率（默认0）
        """
        # 超额收益率
        excess_returns = returns - risk_free_rate
        excess_market = market_returns - risk_free_rate
        
        # OLS回归
        n = len(excess_returns.dropna())
        x = excess_market.dropna().values.reshape(-1, 1)
        y = excess_returns.dropna().values
        
        # 添加常数项
        x_with_const = np.column_stack([np.ones(n), x])
        
        # 计算beta和alpha
        beta_hat = np.linalg.lstsq(x_with_const, y, rcond=None)[0]
        self.alpha = beta_hat[0]
        self.beta = beta_hat[1]
        
        # 计算R-squared
        y_pred = x_with_const @ beta_hat
        ss_tot = np.sum((y - np.mean(y))**2)
        ss_res = np.sum((y - y_pred)**2)
        self.r_squared = 1 - ss_res / ss_tot
        
        return self
    
    def predict(self, market_returns, risk_free_rate=0):
        """预测资产收益率"""
        return risk_free_rate + self.alpha + self.beta * (market_returns - risk_free_rate)
    
    def summary(self):
        """输出模型摘要"""
        print("=" * 60)
        print("CAPM 模型摘要")
        print("=" * 60)
        print(f"Alpha (α): {self.alpha:.6f}")
        print(f"Beta (β): {self.beta:.6f}")
        print(f"R-squared: {self.r_squared:.4f}")
        print("=" * 60)


class GARCH:
    """
    GARCH(p, q) 模型 - 广义自回归条件异方差模型
    
    模型形式:
    σ_t² = ω + α₁ε_{t-1}² + ... + α_qε_{t-q}² + β₁σ_{t-1}² + ... + β_pσ_{t-p}²
    
    其中:
    - σ_t²: t时刻的条件方差
    - ε_t: 残差项
    - ω, α_i, β_i: 模型参数
    """
    
    def __init__(self, p=1, q=1):
        self.p = p
        self.q = q
        self.omega = None
        self.alpha = None
        self.beta = None
        self.sigma2 = None
    
    def fit(self, data):
        """拟合GARCH模型"""
        self.data = np.array(data, dtype=np.float64)
        self.n = len(self.data)
        
        # 初始化参数
        initial_params = np.array([0.01] + [0.1]*self.q + [0.8]*self.p)
        
        # 参数约束：所有参数非负，且 α + β < 1
        bounds = [(1e-10, None)] + [(1e-10, 0.99)] * (self.q + self.p)
        
        result = optimize.minimize(
            self._negative_log_likelihood,
            initial_params,
            bounds=bounds,
            method='L-BFGS-B'
        )
        
        self.omega = result.x[0]
        self.alpha = result.x[1:self.q+1]
        self.beta = result.x[self.q+1:self.q+1+self.p]
        
        # 计算条件方差
        self._compute_sigma2()
        
        return self
    
    def _negative_log_likelihood(self, params):
        """负对数似然函数"""
        omega = params[0]
        alpha = params[1:self.q+1]
        beta = params[self.q+1:self.q+1+self.p]
        
        # 检查参数约束
        if omega <= 0 or np.any(alpha <= 0) or np.any(beta <= 0):
            return np.inf
        if np.sum(alpha) + np.sum(beta) >= 1:
            return np.inf
        
        # 计算条件方差
        sigma2 = np.zeros(self.n)
        sigma2[0] = np.var(self.data)
        
        for t in range(1, self.n):
            sigma2[t] = omega
            for i in range(min(self.q, t)):
                sigma2[t] += alpha[i] * self.data[t-i-1]**2
            for i in range(min(self.p, t)):
                sigma2[t] += beta[i] * sigma2[t-i-1]
        
        # 对数似然
        log_likelihood = -0.5 * np.sum(np.log(2 * np.pi * sigma2) + self.data**2 / sigma2)
        
        return -log_likelihood
    
    def _compute_sigma2(self):
        """计算条件方差"""
        self.sigma2 = np.zeros(self.n)
        self.sigma2[0] = np.var(self.data)
        
        for t in range(1, self.n):
            self.sigma2[t] = self.omega
            for i in range(min(self.q, t)):
                self.sigma2[t] += self.alpha[i] * self.data[t-i-1]**2
            for i in range(min(self.p, t)):
                self.sigma2[t] += self.beta[i] * self.sigma2[t-i-1]
    
    def predict_volatility(self, steps=1):
        """预测波动率"""
        predictions = np.zeros(steps)
        last_sigma2 = self.sigma2[-1]
        
        for i in range(steps):
            predictions[i] = self.omega + np.sum(self.alpha) * self.data[-1]**2 + np.sum(self.beta) * last_sigma2
            last_sigma2 = predictions[i]
        
        return np.sqrt(predictions)
    
    def summary(self):
        """输出模型摘要"""
        print("=" * 60)
        print(f"GARCH({self.p}, {self.q}) 模型摘要")
        print("=" * 60)
        print(f"ω (omega): {self.omega:.6f}")
        print(f"α (alpha): {self.alpha}")
        print(f"β (beta): {self.beta}")
        print(f"持久性: {np.sum(self.alpha) + np.sum(self.beta):.4f}")
        print("=" * 60)


class ARMA:
    """
    ARMA(p, q) 模型 - 自回归移动平均模型
    
    模型形式:
    y_t = c + φ₁y_{t-1} + ... + φ_py_{t-p} + ε_t + θ₁ε_{t-1} + ... + θ_qε_{t-q}
    
    其中:
    - φ: AR系数
    - θ: MA系数
    - ε_t: 白噪声
    """
    
    def __init__(self, p=1, q=1):
        self.p = p
        self.q = q
        self.phi = None
        self.theta = None
        self.mu = None
        self.sigma2 = None
    
    def fit(self, data):
        """拟合ARMA模型"""
        self.data = np.array(data, dtype=np.float64)
        self.n = len(self.data)
        
        # 初始化参数
        initial_params = np.zeros(self.p + self.q + 2)
        initial_params[-1] = np.var(self.data)
        
        result = optimize.minimize(
            self._negative_log_likelihood,
            initial_params,
            method='L-BFGS-B'
        )
        
        self.phi = result.x[:self.p]
        self.theta = result.x[self.p:self.p+self.q]
        self.mu = result.x[self.p+self.q]
        self.sigma2 = result.x[-1]
        
        return self
    
    def _negative_log_likelihood(self, params):
        """负对数似然函数"""
        phi = params[:self.p]
        theta = params[self.p:self.p+self.q]
        mu = params[self.p+self.q]
        sigma2 = params[-1]
        
        if sigma2 <= 0:
            return np.inf
        
        # 计算残差
        residuals = self._compute_residuals(phi, theta, mu)
        
        # 对数似然
        n_eff = len(residuals)
        log_likelihood = -0.5 * n_eff * np.log(2 * np.pi * sigma2) - \
                       0.5 * np.sum(residuals**2) / sigma2
        
        return -log_likelihood
    
    def _compute_residuals(self, phi, theta, mu):
        """计算残差"""
        n = len(self.data)
        max_lag = max(self.p, self.q)
        residuals = np.zeros(n - max_lag)
        
        for i in range(max_lag, n):
            ar_part = np.sum([phi[j] * self.data[i-j-1] for j in range(self.p)])
            ma_part = np.sum([theta[j] * residuals[i-j-1-max_lag] for j in range(self.q) if i-j-1 >= max_lag])
            residuals[i - max_lag] = self.data[i] - ar_part - ma_part - mu
        
        return residuals
    
    def predict(self, steps=1):
        """预测"""
        predictions = np.zeros(steps)
        last_data = self.data[-max(self.p, self.q):]
        
        for i in range(steps):
            ar_part = np.sum([self.phi[j] * last_data[-(j+1)] for j in range(self.p)])
            predictions[i] = ar_part + self.mu
            last_data = np.append(last_data[1:], predictions[i])
        
        return predictions
    
    def summary(self):
        """输出模型摘要"""
        print("=" * 60)
        print(f"ARMA({self.p}, {self.q}) 模型摘要")
        print("=" * 60)
        print(f"常数项 (μ): {self.mu:.6f}")
        print(f"AR系数 (φ): {self.phi}")
        print(f"MA系数 (θ): {self.theta}")
        print(f"误差方差 (σ²): {self.sigma2:.6f}")
        print("=" * 60)


class Cointegration:
    """
    协整检验与模型
    
    Engle-Granger 两步法:
    1. 用OLS估计长期均衡关系
    2. 对残差进行单位根检验
    """
    
    def __init__(self):
        self.coeffs = None
        self.residuals = None
        self.adf_stat = None
        self.adf_pval = None
    
    def fit(self, y, x):
        """
        拟合协整模型
        
        参数:
        y: 因变量
        x: 自变量（可以是多变量）
        """
        y = np.array(y).reshape(-1, 1)
        x = np.array(x)
        
        if x.ndim == 1:
            x = x.reshape(-1, 1)
        
        # 添加常数项
        X = np.column_stack([np.ones(len(y)), x])
        
        # OLS回归
        self.coeffs = np.linalg.lstsq(X, y, rcond=None)[0]
        
        # 计算残差
        self.residuals = y - X @ self.coeffs
        
        # ADF检验残差
        from scipy.stats import adfuller
        result = adfuller(self.residuals)
        self.adf_stat = result[0]
        self.adf_pval = result[1]
        
        return self
    
    def is_cointegrated(self, alpha=0.05):
        """判断是否存在协整关系"""
        return self.adf_pval < alpha
    
    def summary(self):
        """输出模型摘要"""
        print("=" * 60)
        print("协整检验摘要 (Engle-Granger 两步法)")
        print("=" * 60)
        print(f"回归系数: {self.coeffs.flatten()}")
        print(f"ADF统计量: {self.adf_stat:.4f}")
        print(f"ADF p值: {self.adf_pval:.4f}")
        print(f"协整关系: {'存在' if self.is_cointegrated() else '不存在'}")
        print("=" * 60)
