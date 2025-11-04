import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

def load_data():
    """
    加载任务1和任务3的数据
    """
    print("加载任务1和任务3的数据...")
    
    # 加载任务1的研究队列数据
    study_population = pd.read_csv('task1/study_population.csv')
    print(f"研究队列数据: {study_population.shape}")
    
    # 加载原始的各个数据表
    patient_demo = pd.read_csv('task1/cleaned_patient_demo.csv')
    diagnoses = pd.read_csv('task1/cleaned_diagnoses.csv')
    medical_events = pd.read_csv('task1/cleaned_medical_events.csv')
    costs = pd.read_csv('task1/cleaned_costs.csv')
    prescriptions = pd.read_csv('task1/cleaned_prescriptions.csv')
    
    print(f"患者人口统计学数据: {patient_demo.shape}")
    print(f"诊断数据: {diagnoses.shape}")
    print(f"医疗事件数据: {medical_events.shape}")
    print(f"费用数据: {costs.shape}")
    print(f"处方数据: {prescriptions.shape}")
    
    # 加载匹配结果
    try:
        matched_pairs = pd.read_csv('task3/tables/matched_pairs.csv')
        print(f"匹配对数据: {matched_pairs.shape}")
        matched_analysis = True
    except FileNotFoundError:
        print("未找到匹配对数据，使用全部队列进行分析")
        matched_pairs = None
        matched_analysis = False
    
    # 将日期列转换为datetime类型
    study_population['followup_start'] = pd.to_datetime(study_population['followup_start'])
    study_population['followup_end'] = pd.to_datetime(study_population['followup_end'])
    medical_events['event_date'] = pd.to_datetime(medical_events['event_date'])
    costs['cost_date'] = pd.to_datetime(costs['cost_date'])
    prescriptions['prescription_date'] = pd.to_datetime(prescriptions['prescription_date'])
    
    return {
        'study_population': study_population,
        'patient_demo': patient_demo,
        'diagnoses': diagnoses,
        'medical_events': medical_events,
        'costs': costs,
        'prescriptions': prescriptions,
        'matched_pairs': matched_pairs,
        'matched_analysis': matched_analysis
    }

if __name__ == "__main__":
    # 加载数据
    data = load_data()
    print("\n数据加载完成！")