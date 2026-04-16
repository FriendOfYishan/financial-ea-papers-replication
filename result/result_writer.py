import pandas as pd
import numpy as np
import os
import json
from datetime import datetime

class ResultWriter:
    """
    结果输出类
    """
    
    def __init__(self, output_dir='result'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def write_dataframe(self, df, filename, index=True):
        """保存DataFrame到CSV文件"""
        filepath = os.path.join(self.output_dir, filename)
        df.to_csv(filepath, index=index)
        print(f"数据已保存到 {filepath}")
    
    def write_json(self, data, filename):
        """保存数据到JSON文件"""
        filepath = os.path.join(self.output_dir, filename)
        
        # 处理numpy类型
        def convert(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, pd.DataFrame):
                return obj.to_dict()
            elif isinstance(obj, datetime):
                return obj.isoformat()
            else:
                return obj
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, default=convert, indent=2, ensure_ascii=False)
        
        print(f"数据已保存到 {filepath}")
    
    def write_text(self, content, filename):
        """保存文本内容到文件"""
        filepath = os.path.join(self.output_dir, filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"文本已保存到 {filepath}")
    
    def write_model_summary(self, model, filename='model_summary.txt'):
        """保存模型摘要"""
        import io
        import sys
        
        # 捕获print输出
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()
        
        model.summary()
        
        summary = sys.stdout.getvalue()
        sys.stdout = old_stdout
        
        self.write_text(summary, filename)
    
    def write_report(self, title, sections):
        """
        生成完整的分析报告
        
        参数:
        title: 报告标题
        sections: 报告章节列表，每个元素包含标题和内容
        """
        report = []
        report.append(f"# {title}")
        report.append("")
        report.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        for i, section in enumerate(sections, 1):
            report.append(f"## {i}. {section['title']}")
            report.append("")
            report.append(section['content'])
            report.append("")
        
        filepath = os.path.join(self.output_dir, 'analysis_report.md')
        self.write_text('\n'.join(report), 'analysis_report.md')
    
    def save_figure(self, fig, filename, dpi=300):
        """保存matplotlib图形"""
        filepath = os.path.join(self.output_dir, filename)
        fig.savefig(filepath, dpi=dpi, bbox_inches='tight')
        print(f"图形已保存到 {filepath}")


class ExperimentTracker:
    """
    实验追踪类
    """
    
    def __init__(self, tracking_dir='result/experiments'):
        self.tracking_dir = tracking_dir
        os.makedirs(tracking_dir, exist_ok=True)
        self.experiments = []
    
    def add_experiment(self, name, parameters, metrics):
        """
        添加实验记录
        
        参数:
        name: 实验名称
        parameters: 参数字典
        metrics: 指标字典
        """
        experiment = {
            'name': name,
            'timestamp': datetime.now().isoformat(),
            'parameters': parameters,
            'metrics': metrics
        }
        self.experiments.append(experiment)
        
        # 保存到文件
        filepath = os.path.join(self.tracking_dir, f'{name}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(experiment, f, indent=2, ensure_ascii=False)
    
    def get_best_experiment(self, metric, maximize=True):
        """
        获取最佳实验
        
        参数:
        metric: 指标名称
        maximize: 是否最大化（True为最大化，False为最小化）
        
        返回:
        最佳实验记录
        """
        if not self.experiments:
            return None
        
        if maximize:
            best = max(self.experiments, key=lambda x: x['metrics'].get(metric, -np.inf))
        else:
            best = min(self.experiments, key=lambda x: x['metrics'].get(metric, np.inf))
        
        return best
    
    def generate_summary_table(self):
        """生成实验汇总表格"""
        if not self.experiments:
            return pd.DataFrame()
        
        rows = []
        for exp in self.experiments:
            row = {
                'name': exp['name'],
                'timestamp': exp['timestamp'],
                **exp['parameters'],
                **exp['metrics']
            }
            rows.append(row)
        
        return pd.DataFrame(rows)
