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

def calculate_outcomes(data):
    """
    计算所有结局指标
    """
    print("计算结局指标...")
    
    study_pop = data['study_population']
    medical_events = data['medical_events']
    costs = data['costs']
    prescriptions = data['prescriptions']
    
    # 准备结果DataFrame
    outcomes = study_pop[['patient_id', 'treatment_group']].copy()
    
    # 1. 计算随访期内的急诊就诊次数
    print("计算随访期内的急诊就诊次数...")
    er_events = medical_events[medical_events['event_type'] == 'ER'].copy()
    
    # 筛选在随访期内的急诊事件
    er_followup = er_events.merge(
        study_pop[['patient_id', 'followup_start', 'followup_end']], 
        on='patient_id', 
        how='inner'
    )
    er_followup = er_followup[
        (er_followup['event_date'] >= er_followup['followup_start']) &
        (er_followup['event_date'] <= er_followup['followup_end'])
    ]
    
    # 计算每个患者的急诊就诊次数
    er_counts = er_followup.groupby('patient_id').size().reset_index(name='er_visits')
    outcomes = outcomes.merge(er_counts, on='patient_id', how='left')
    outcomes['er_visits'] = outcomes['er_visits'].fillna(0)
    
    # 2. 计算随访期内的住院次数
    print("计算随访期内的住院次数...")
    hosp_events = medical_events[medical_events['event_type'] == 'Hospitalization'].copy()
    
    # 筛选在随访期内的住院事件
    hosp_followup = hosp_events.merge(
        study_pop[['patient_id', 'followup_start', 'followup_end']], 
        on='patient_id', 
        how='inner'
    )
    hosp_followup = hosp_followup[
        (hosp_followup['event_date'] >= hosp_followup['followup_start']) &
        (hosp_followup['event_date'] <= hosp_followup['followup_end'])
    ]
    
    # 计算每个患者的住院次数
    hosp_counts = hosp_followup.groupby('patient_id').size().reset_index(name='hospitalizations')
    outcomes = outcomes.merge(hosp_counts, on='patient_id', how='left')
    outcomes['hospitalizations'] = outcomes['hospitalizations'].fillna(0)
    
    # 3. 计算随访期内的住院天数总和
    print("计算随访期内的住院天数总和...")
    hosp_days = hosp_followup.groupby('patient_id')['length_of_stay'].sum().reset_index(name='total_hospital_days')
    outcomes = outcomes.merge(hosp_days, on='patient_id', how='left')
    outcomes['total_hospital_days'] = outcomes['total_hospital_days'].fillna(0)
    
    # 4. 计算随访期内的门诊就诊次数
    print("计算随访期内的门诊就诊次数...")
    op_events = medical_events[medical_events['event_type'] == 'Outpatient'].copy()
    
    # 筛选在随访期内的门诊事件
    op_followup = op_events.merge(
        study_pop[['patient_id', 'followup_start', 'followup_end']], 
        on='patient_id', 
        how='inner'
    )
    op_followup = op_followup[
        (op_followup['event_date'] >= op_followup['followup_start']) &
        (op_followup['event_date'] <= op_followup['followup_end'])
    ]
    
    # 计算每个患者的门诊就诊次数
    op_counts = op_followup.groupby('patient_id').size().reset_index(name='outpatient_visits')
    outcomes = outcomes.merge(op_counts, on='patient_id', how='left')
    outcomes['outpatient_visits'] = outcomes['outpatient_visits'].fillna(0)
    
    # 5. 计算随访期内的总医疗费用
    print("计算随访期内的总医疗费用...")
    costs_followup = costs.merge(
        study_pop[['patient_id', 'followup_start', 'followup_end']], 
        on='patient_id', 
        how='inner'
    )
    costs_followup = costs_followup[
        (costs_followup['cost_date'] >= costs_followup['followup_start']) &
        (costs_followup['cost_date'] <= costs_followup['followup_end'])
    ]
    
    # 计算每个患者的总医疗费用
    total_costs = costs_followup.groupby('patient_id')['cost_amount'].sum().reset_index(name='total_cost')
    outcomes = outcomes.merge(total_costs, on='patient_id', how='left')
    outcomes['total_cost'] = outcomes['total_cost'].fillna(0)
    
    # 6. 计算随访期内的处方费用
    print("计算随访期内的处方费用...")
    prescription_costs = costs_followup[costs_followup['cost_type'] == 'Prescription'].copy()
    rx_costs = prescription_costs.groupby('patient_id')['cost_amount'].sum().reset_index(name='prescription_cost')
    outcomes = outcomes.merge(rx_costs, on='patient_id', how='left')
    outcomes['prescription_cost'] = outcomes['prescription_cost'].fillna(0)
    
    print(f"结局指标计算完成，共处理 {len(outcomes)} 名患者")
    
    return outcomes

if __name__ == "__main__":
    # 加载数据
    data = load_data()
    
    # 计算结局指标
    outcomes = calculate_outcomes(data)
    
    print("\n结局指标计算完成！")
    
    # 保存中间结果
    os.makedirs('task4/tables', exist_ok=True)
    outcomes.to_csv('task4/tables/outcomes.csv', index=False)
    print("结局指标已保存至 task4/tables/outcomes.csv")