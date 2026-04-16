import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns

class DescriptiveStats:
    """
    描述性统计分析类
    """
    
    def __init__(self, data):
        self.data = data
        self.stats_summary = None
    
    def compute_summary_stats(self):
        """计算描述性统计量"""
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        stats_dict = {}
        
        for col in numeric_cols:
            values = self.data[col].dropna().values
            
            stats_dict[col] = {
                'count': len(values),
                'mean': np.mean(values),
                'std': np.std(values, ddof=1),
                'min': np.min(values),
                '25%': np.percentile(values, 25),
                '50%': np.median(values),
                '75%': np.percentile(values, 75),
                'max': np.max(values),
                'skewness': stats.skew(values),
                'kurtosis': stats.kurtosis(values, fisher=True),
                'jarque_bera_stat': stats.jarque_bera(values)[0],
                'jarque_bera_pval': stats.jarque_bera(values)[1]
            }
        
        self.stats_summary = pd.DataFrame(stats_dict).T
        return self.stats_summary
    
    def compute_correlation_matrix(self):
        """计算相关系数矩阵"""
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        return self.data[numeric_cols].corr()
    
    def test_normality(self, column):
        """正态性检验"""
        values = self.data[column].dropna().values
        
        # Shapiro-Wilk 检验
        shapiro_stat, shapiro_pval = stats.shapiro(values)
        
        # Kolmogorov-Smirnov 检验
        ks_stat, ks_pval = stats.kstest(values, 'norm', 
                                       args=(np.mean(values), np.std(values)))
        
        # Jarque-Bera 检验
        jb_stat, jb_pval = stats.jarque_bera(values)
        
        return {
            'shapiro_stat': shapiro_stat,
            'shapiro_pval': shapiro_pval,
            'ks_stat': ks_stat,
            'ks_pval': ks_pval,
            'jb_stat': jb_stat,
            'jb_pval': jb_pval
        }
    
    def test_stationarity(self, column):
        """平稳性检验（ADF检验）"""
        values = self.data[column].dropna().values
        
        # Augmented Dickey-Fuller 检验
        adf_stat, adf_pval, used_lag, nobs, critical_values, icbest = stats.adfuller(values)
        
        return {
            'adf_stat': adf_stat,
            'adf_pval': adf_pval,
            'critical_values': critical_values,
            'stationary': adf_pval < 0.05
        }
    
    def plot_histogram(self, column, bins=50, figsize=(10, 6)):
        """绘制直方图"""
        plt.figure(figsize=figsize)
        sns.histplot(self.data[column].dropna(), bins=bins, kde=True)
        plt.title(f'{column} 直方图')
        plt.xlabel(column)
        plt.ylabel('频率')
        plt.grid(True, alpha=0.3)
        plt.show()
    
    def plot_time_series(self, columns=None, figsize=(15, 10)):
        """绘制时间序列图"""
        if columns is None:
            columns = self.data.select_dtypes(include=[np.number]).columns
        
        n_cols = len(columns)
        fig, axes = plt.subplots(n_cols, 1, figsize=figsize)
        
        if n_cols == 1:
            axes = [axes]
        
        for i, col in enumerate(columns):
            axes[i].plot(self.data.index, self.data[col])
            axes[i].set_title(f'{col} 时间序列')
            axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def plot_correlation_heatmap(self, figsize=(12, 10)):
        """绘制相关系数热力图"""
        corr_matrix = self.compute_correlation_matrix()
        
        plt.figure(figsize=figsize)
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
        plt.title('相关系数矩阵')
        plt.show()
    
    def plot_qq(self, column, figsize=(10, 6)):
        """绘制Q-Q图"""
        plt.figure(figsize=figsize)
        stats.probplot(self.data[column].dropna(), plot=plt)
        plt.title(f'{column} Q-Q图')
        plt.show()
    
    def get_summary_report(self):
        """生成完整的统计摘要报告"""
        report = []
        report.append("=" * 60)
        report.append("描述性统计分析报告")
        report.append("=" * 60)
        report.append("")
        
        # 数据基本信息
        report.append("1. 数据基本信息")
        report.append("-" * 60)
        report.append(f"样本数量: {len(self.data)}")
        report.append(f"变量数量: {len(self.data.columns)}")
        report.append(f"时间范围: {self.data.index.min()} 至 {self.data.index.max()}")
        report.append("")
        
        # 描述性统计
        report.append("2. 描述性统计量")
        report.append("-" * 60)
        summary = self.compute_summary_stats()
        report.append(summary.to_string())
        report.append("")
        
        # 相关系数矩阵
        report.append("3. 相关系数矩阵")
        report.append("-" * 60)
        corr_matrix = self.compute_correlation_matrix()
        report.append(corr_matrix.to_string())
        report.append("")
        
        # 正态性检验
        report.append("4. 正态性检验")
        report.append("-" * 60)
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            normality = self.test_normality(col)
            report.append(f"\n变量: {col}")
            report.append(f"  Shapiro-Wilk: 统计量={normality['shapiro_stat']:.4f}, p值={normality['shapiro_pval']:.4f}")
            report.append(f"  Jarque-Bera: 统计量={normality['jb_stat']:.4f}, p值={normality['jb_pval']:.4f}")
            is_normal = normality['jb_pval'] > 0.05
            report.append(f"  结论: {'服从正态分布' if is_normal else '不服从正态分布'}")
        
        # 平稳性检验
        report.append("\n5. 平稳性检验（ADF）")
        report.append("-" * 60)
        for col in numeric_cols:
            stationarity = self.test_stationarity(col)
            report.append(f"\n变量: {col}")
            report.append(f"  ADF统计量: {stationarity['adf_stat']:.4f}")
            report.append(f"  p值: {stationarity['adf_pval']:.4f}")
            report.append(f"  临界值: {stationarity['critical_values']}")
            report.append(f"  结论: {'平稳序列' if stationarity['stationary'] else '非平稳序列'}")
        
        report.append("\n" + "=" * 60)
        report.append("报告结束")
        report.append("=" * 60)
        
        return "\n".join(report)
