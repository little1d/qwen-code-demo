import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os

def load_data():
    """
    加载任务1-4的数据和结果
    """
    print("加载任务1-4的数据和结果...")
    
    # 加载基线特征数据
    try:
        baseline_demo = pd.read_csv('task2/tables/descriptive_stats_age.csv')
        baseline_gender = pd.read_csv('task2/tables/descriptive_stats_gender.csv')
        baseline_region = pd.read_csv('task2/tables/descriptive_stats_region.csv')
        baseline_comorbidity = pd.read_csv('task2/tables/descriptive_stats_comorbidity_count.csv')
        print("基线特征数据加载完成")
    except FileNotFoundError as e:
        print(f"无法加载基线特征数据: {e}")
        baseline_demo = baseline_gender = baseline_region = baseline_comorbidity = None
    
    # 加载匹配前后平衡性评估
    try:
        balance_assessment = pd.read_csv('task3/tables/balance_assessment.csv')
        matched_pairs = pd.read_csv('task3/tables/matched_pairs.csv')
        print("匹配平衡性数据加载完成")
    except FileNotFoundError as e:
        print(f"无法加载匹配平衡性数据: {e}")
        balance_assessment = matched_pairs = None
    
    # 加载结局指标数据
    try:
        outcomes = pd.read_csv('task4/tables/outcomes.csv')
        resource_comparison = pd.read_csv('task4/tables/resource_utilization_comparison.csv')
        cost_comparison = pd.read_csv('task4/tables/cost_comparison.csv')
        pdc_results = pd.read_csv('task4/tables/pdc_results.csv')
        discontinuation_analysis = pd.read_csv('task4/tables/discontinuation_analysis.csv')
        main_outcomes_summary = pd.read_csv('task4/tables/main_outcomes_summary.csv')
        print("结局指标数据加载完成")
    except FileNotFoundError as e:
        print(f"无法加载结局指标数据: {e}")
        outcomes = resource_comparison = cost_comparison = pdc_results = discontinuation_analysis = main_outcomes_summary = None
    
    # 加载可视化图表
    try:
        figures_path = 'task4/figures/'
        print(f"可视化图表路径: {figures_path}")
    except:
        print("无法找到可视化图表")
    
    return {
        'baseline_demo': baseline_demo,
        'baseline_gender': baseline_gender,
        'baseline_region': baseline_region,
        'baseline_comorbidity': baseline_comorbidity,
        'balance_assessment': balance_assessment,
        'matched_pairs': matched_pairs,
        'outcomes': outcomes,
        'resource_comparison': resource_comparison,
        'cost_comparison': cost_comparison,
        'pdc_results': pdc_results,
        'discontinuation_analysis': discontinuation_analysis,
        'main_outcomes_summary': main_outcomes_summary,
        'figures_path': 'task4/figures/'
    }

if __name__ == "__main__":
    # 加载数据
    data = load_data()
    print("\n数据加载完成！")