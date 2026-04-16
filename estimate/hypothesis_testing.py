import numpy as np
import pandas as pd
import scipy.stats as stats
from scipy import optimize

class HypothesisTesting:
    """
    假设检验工具类
    """
    
    @staticmethod
    def t_test(sample1, sample2=None, alternative='two-sided'):
        """
        t检验
        
        参数:
        sample1: 样本1
        sample2: 样本2（独立样本t检验时使用）
        alternative: 备择假设 ('two-sided', 'less', 'greater')
        
        返回:
        t_stat: t统计量
        p_value: p值
        """
        sample1 = np.array(sample1).flatten()
        sample1 = sample1[~np.isnan(sample1)]
        
        if sample2 is None:
            # 单样本t检验
            t_stat, p_value = stats.ttest_1samp(sample1, 0)
        else:
            # 独立样本t检验
            sample2 = np.array(sample2).flatten()
            sample2 = sample2[~np.isnan(sample2)]
            t_stat, p_value = stats.ttest_ind(sample1, sample2)
        
        # 根据备择假设调整p值
        if alternative == 'less':
            p_value = p_value / 2 if t_stat < 0 else 1 - p_value / 2
        elif alternative == 'greater':
            p_value = p_value / 2 if t_stat > 0 else 1 - p_value / 2
        
        return {'t_stat': t_stat, 'p_value': p_value}
    
    @staticmethod
    def f_test(sample1, sample2):
        """
        F检验（方差齐性检验）
        
        参数:
        sample1: 样本1
        sample2: 样本2
        
        返回:
        f_stat: F统计量
        p_value: p值
        """
        sample1 = np.array(sample1).flatten()
        sample2 = np.array(sample2).flatten()
        
        var1 = np.var(sample1, ddof=1)
        var2 = np.var(sample2, ddof=1)
        
        # F统计量 = 较大方差 / 较小方差
        if var1 >= var2:
            f_stat = var1 / var2
            df1 = len(sample1) - 1
            df2 = len(sample2) - 1
        else:
            f_stat = var2 / var1
            df1 = len(sample2) - 1
            df2 = len(sample1) - 1
        
        # 双尾检验
        p_value = 2 * min(stats.f.cdf(f_stat, df1, df2), 1 - stats.f.cdf(f_stat, df1, df2))
        
        return {'f_stat': f_stat, 'p_value': p_value}
    
    @staticmethod
    def chisquare_test(observed, expected):
        """
        卡方检验
        
        参数:
        observed: 观测频数
        expected: 期望频数
        
        返回:
        chi_stat: 卡方统计量
        p_value: p值
        """
        observed = np.array(observed)
        expected = np.array(expected)
        
        chi_stat, p_value = stats.chisquare(observed, expected)
        
        return {'chi_stat': chi_stat, 'p_value': p_value}
    
    @staticmethod
    def adf_test(data):
        """
        Augmented Dickey-Fuller 检验（单位根检验）
        
        参数:
        data: 时间序列数据
        
        返回:
        adf_stat: ADF统计量
        p_value: p值
        critical_values: 临界值
        """
        data = np.array(data).flatten()
        data = data[~np.isnan(data)]
        
        result = stats.adfuller(data)
        
        return {
            'adf_stat': result[0],
            'p_value': result[1],
            'critical_values': result[4]
        }
    
    @staticmethod
    def granger_causality_test(data, max_lag=5):
        """
        Granger因果检验
        
        参数:
        data: 包含两列的DataFrame
        max_lag: 最大滞后阶数
        
        返回:
        检验结果字典
        """
        results = {}
        
        for lag in range(1, max_lag + 1):
            # 从data中提取两列
            y = data.iloc[:, 0].values
            x = data.iloc[:, 1].values
            
            # 构建滞后变量
            n = len(y)
            X = np.ones((n - lag, 2 * lag + 1))
            
            for i in range(lag):
                X[:, i + 1] = y[n - lag - i - 1 : n - i - 1]  # y的滞后
                X[:, i + lag + 1] = x[n - lag - i - 1 : n - i - 1]  # x的滞后
            
            y_endog = y[lag:]
            
            # 无约束回归（包含x的滞后项）
            beta_ur = np.linalg.lstsq(X, y_endog, rcond=None)[0]
            ssr_ur = np.sum((y_endog - X @ beta_ur) ** 2)
            
            # 约束回归（不包含x的滞后项）
            X_r = X[:, :lag + 1]
            beta_r = np.linalg.lstsq(X_r, y_endog, rcond=None)[0]
            ssr_r = np.sum((y_endog - X_r @ beta_r) ** 2)
            
            # F检验
            df1 = lag
            df2 = n - 2 * lag - 1
            f_stat = ((ssr_r - ssr_ur) / df1) / (ssr_ur / df2)
            p_value = 1 - stats.f.cdf(f_stat, df1, df2)
            
            results[lag] = {'f_stat': f_stat, 'p_value': p_value}
        
        return results
    
    @staticmethod
    def variance_ratio_test(data, q=2):
        """
        方差比检验（随机游走检验）
        
        参数:
        data: 时间序列数据
        q: 时间间隔
        
        返回:
        vr_stat: 方差比统计量
        p_value: p值
        """
        data = np.array(data).flatten()
        data = data[~np.isnan(data)]
        
        n = len(data)
        returns = np.diff(data)
        
        # 计算方差比
        sigma1_sq = np.var(returns, ddof=1)
        sigmaq_sq = np.var(np.diff(data, q), ddof=1) / q
        
        vr_stat = sigmaq_sq / sigma1_sq
        
        # 渐近正态分布
        se = np.sqrt(2 * (2 * q - 1) * (q - 1) / (3 * q * n))
        z_stat = (vr_stat - 1) / se
        p_value = 2 * (1 - stats.norm.cdf(np.abs(z_stat)))
        
        return {'vr_stat': vr_stat, 'z_stat': z_stat, 'p_value': p_value}


class RegressionDiagnostics:
    """
    回归诊断工具类
    """
    
    @staticmethod
    def breusch_pagan_test(residuals, X):
        """
        Breusch-Pagan 异方差检验
        
        参数:
        residuals: 回归残差
        X: 自变量矩阵
        
        返回:
        bp_stat: BP统计量
        p_value: p值
        """
        residuals = np.array(residuals).flatten()
        X = np.array(X)
        
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        # 添加常数项
        X = np.column_stack([np.ones(len(residuals)), X])
        
        # 对残差平方进行回归
        res_sq = residuals ** 2
        beta = np.linalg.lstsq(X, res_sq, rcond=None)[0]
        r_squared = 1 - np.sum((res_sq - X @ beta) ** 2) / np.sum((res_sq - np.mean(res_sq)) ** 2)
        
        n = len(residuals)
        bp_stat = n * r_squared
        p_value = 1 - stats.chi2.cdf(bp_stat, X.shape[1] - 1)
        
        return {'bp_stat': bp_stat, 'p_value': p_value}
    
    @staticmethod
    def durbin_watson_test(residuals):
        """
        Durbin-Watson 自相关检验
        
        参数:
        residuals: 回归残差
        
        返回:
        dw_stat: DW统计量
        """
        residuals = np.array(residuals).flatten()
        
        diff_resid = np.diff(residuals)
        dw_stat = np.sum(diff_resid ** 2) / np.sum(residuals ** 2)
        
        return {'dw_stat': dw_stat}
    
    @staticmethod
    def jarque_bera_test(residuals):
        """
        Jarque-Bera 正态性检验
        
        参数:
        residuals: 回归残差
        
        返回:
        jb_stat: JB统计量
        p_value: p值
        """
        residuals = np.array(residuals).flatten()
        residuals = residuals[~np.isnan(residuals)]
        
        jb_stat, p_value = stats.jarque_bera(residuals)
        
        return {'jb_stat': jb_stat, 'p_value': p_value}
    
    @staticmethod
    def variance_inflation_factor(X):
        """
        计算方差膨胀因子（VIF）
        
        参数:
        X: 自变量矩阵
        
        返回:
        vif_values: VIF值数组
        """
        X = np.array(X)
        
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        n_features = X.shape[1]
        vif_values = np.zeros(n_features)
        
        for i in range(n_features):
            # 将第i列作为因变量，其他列作为自变量
            y = X[:, i]
            X_excl = np.delete(X, i, axis=1)
            
            # 添加常数项
            X_excl = np.column_stack([np.ones(len(y)), X_excl])
            
            # 回归
            beta = np.linalg.lstsq(X_excl, y, rcond=None)[0]
            r_squared = 1 - np.sum((y - X_excl @ beta) ** 2) / np.sum((y - np.mean(y)) ** 2)
            
            vif_values[i] = 1 / (1 - r_squared)
        
        return {'vif': vif_values}


class ModelSelection:
    """
    模型选择工具类
    """
    
    @staticmethod
    def aic(n, log_likelihood, k):
        """
        计算AIC
        
        参数:
        n: 样本数量
        log_likelihood: 对数似然值
        k: 参数数量
        
        返回:
        aic: AIC值
        """
        return 2 * k - 2 * log_likelihood
    
    @staticmethod
    def bic(n, log_likelihood, k):
        """
        计算BIC
        
        参数:
        n: 样本数量
        log_likelihood: 对数似然值
        k: 参数数量
        
        返回:
        bic: BIC值
        """
        return k * np.log(n) - 2 * log_likelihood
    
    @staticmethod
    def aicc(n, log_likelihood, k):
        """
        计算修正的AIC（AICc）
        
        参数:
        n: 样本数量
        log_likelihood: 对数似然值
        k: 参数数量
        
        返回:
        aicc: AICc值
        """
        aic = 2 * k - 2 * log_likelihood
        return aic + 2 * k * (k + 1) / (n - k - 1)
    
    @staticmethod
    def information_criteria(n, log_likelihood, k):
        """
        计算所有信息准则
        
        参数:
        n: 样本数量
        log_likelihood: 对数似然值
        k: 参数数量
        
        返回:
        包含AIC、BIC、AICc的字典
        """
        return {
            'aic': ModelSelection.aic(n, log_likelihood, k),
            'bic': ModelSelection.bic(n, log_likelihood, k),
            'aicc': ModelSelection.aicc(n, log_likelihood, k)
        }
