import pandas as pd
import numpy as np
import os
from datetime import datetime

class DataLoader:
    """
    金融数据加载与处理类
    """
    
    def __init__(self, data_dir='data'):
        self.data_dir = data_dir
        self.raw_data = None
        self.processed_data = None
    
    def load_csv(self, filename):
        """加载CSV数据文件"""
        filepath = os.path.join(self.data_dir, filename)
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"文件 {filepath} 不存在")
        self.raw_data = pd.read_csv(filepath)
        return self.raw_data
    
    def load_multiple_csv(self, filenames):
        """加载多个CSV文件并合并"""
        dfs = []
        for filename in filenames:
            filepath = os.path.join(self.data_dir, filename)
            if os.path.exists(filepath):
                df = pd.read_csv(filepath)
                dfs.append(df)
        if dfs:
            self.raw_data = pd.concat(dfs, ignore_index=True)
        return self.raw_data
    
    def preprocess(self):
        """数据预处理"""
        if self.raw_data is None:
            raise ValueError("请先加载数据")
        
        df = self.raw_data.copy()
        
        # 1. 处理日期列
        date_cols = [col for col in df.columns if 'date' in col.lower() or 'Date' in col]
        for col in date_cols:
            df[col] = pd.to_datetime(df[col])
            df.set_index(col, inplace=True)
        
        # 2. 处理缺失值
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            # 使用中位数填充数值型缺失值
            df[col] = df[col].fillna(df[col].median())
        
        # 3. 计算收益率
        if 'close' in df.columns or 'Close' in df.columns:
            price_col = 'close' if 'close' in df.columns else 'Close'
            df['return'] = df[price_col].pct_change()
            df['log_return'] = np.log(df[price_col] / df[price_col].shift(1))
        
        # 4. 计算波动率（滚动窗口）
        if 'return' in df.columns:
            df['volatility'] = df['return'].rolling(window=252).std() * np.sqrt(252)
        
        self.processed_data = df
        return self.processed_data
    
    def save_processed_data(self, filename='processed_data.csv'):
        """保存处理后的数据"""
        if self.processed_data is None:
            raise ValueError("请先处理数据")
        filepath = os.path.join(self.data_dir, filename)
        self.processed_data.to_csv(filepath)
        print(f"处理后的数据已保存到 {filepath}")
    
    def generate_sample_data(self, n=1000):
        """生成模拟金融数据"""
        dates = pd.date_range(start='2015-01-01', periods=n, freq='D')
        
        # 生成模拟价格（几何布朗运动）
        np.random.seed(42)
        returns = np.random.normal(0.0005, 0.02, n)
        prices = 100 * np.exp(np.cumsum(returns))
        
        # 生成其他财务指标
        volume = np.random.poisson(1000000, n)
        pe_ratio = np.random.normal(15, 3, n)
        dividend_yield = np.random.normal(0.02, 0.01, n)
        
        self.raw_data = pd.DataFrame({
            'date': dates,
            'close': prices,
            'volume': volume,
            'pe_ratio': pe_ratio,
            'dividend_yield': dividend_yield,
            'market_return': np.random.normal(0.0005, 0.015, n)
        })
        
        self.raw_data['return'] = self.raw_data['close'].pct_change()
        self.processed_data = self.raw_data
        return self.raw_data
