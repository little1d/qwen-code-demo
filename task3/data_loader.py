import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from scipy.spatial.distance import cdist
import os

def load_data():
    """
    加载任务1和任务2生成的数据
    """
    print("加载任务1和任务2生成的数据...")
    
    # 加载任务1的研究队列数据
    study_population = pd.read_csv('task1/study_population.csv')
    print(f"研究队列数据: {study_population.shape}")
    
    # 加载患者人口统计学数据
    patient_demo = pd.read_csv('task1/cleaned_patient_demo.csv')
    print(f"患者人口统计学数据: {patient_demo.shape}")
    
    # 加载诊断数据
    diagnoses = pd.read_csv('task1/cleaned_diagnoses.csv')
    print(f"诊断数据: {diagnoses.shape}")
    
    # 加载任务2的合并症数据
    comorbidity_data = pd.read_csv('task2/tables/descriptive_stats_comorbidity_count.csv')
    print(f"合并症数据: {comorbidity_data.shape}")
    
    # 整合所有数据
    # 先合并患者人口统计学数据
    baseline_data = study_population[['patient_id', 'treatment_group']].merge(
        patient_demo, 
        on='patient_id', 
        how='left'
    )
    
    # 获取合并症数据
    comorbidities = diagnoses[diagnoses['is_target_disease'] == False].copy()
    comorbidities['diagnosis_date'] = pd.to_datetime(comorbidities['diagnosis_date'])
    
    # 获取研究队列的基线期信息
    baseline_info = study_population[['patient_id', 'baseline_start', 'baseline_end']].copy()
    baseline_info['baseline_start'] = pd.to_datetime(baseline_info['baseline_start'])
    baseline_info['baseline_end'] = pd.to_datetime(baseline_info['baseline_end'])
    
    # 合并基线期信息
    comorbidities_with_baseline = comorbidities.merge(baseline_info, on='patient_id', how='inner')
    
    # 筛选在基线期内的诊断记录
    baseline_comorbidities = comorbidities_with_baseline[
        (comorbidities_with_baseline['diagnosis_date'] >= comorbidities_with_baseline['baseline_start']) &
        (comorbidities_with_baseline['diagnosis_date'] <= comorbidities_with_baseline['baseline_end'])
    ]
    
    # 计算每个患者的合并症数量
    patient_comorbidity_counts = baseline_comorbidities.groupby('patient_id').size().reset_index(name='comorbidity_count')
    
    # 将合并症信息添加到基线数据
    baseline_data = baseline_data.merge(patient_comorbidity_counts, on='patient_id', how='left')
    baseline_data['comorbidity_count'] = baseline_data['comorbidity_count'].fillna(0)
    
    # 编码性别和地区变量
    le_gender = LabelEncoder()
    baseline_data['gender_encoded'] = le_gender.fit_transform(baseline_data['gender'])
    
    le_region = LabelEncoder()
    baseline_data['region_encoded'] = le_region.fit_transform(baseline_data['region'])
    
    print("数据加载完成")
    print(f"最终基线数据形状: {baseline_data.shape}")
    print(f"治疗组分布:\n{baseline_data['treatment_group'].value_counts()}")
    
    return baseline_data, le_gender, le_region

if __name__ == "__main__":
    # 加载数据
    baseline_data, le_gender, le_region = load_data()
    
    print("\n任务3数据准备完成！")