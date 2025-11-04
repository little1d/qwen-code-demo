import pandas as pd
from datetime import datetime
import os

# 定义数据文件路径
data_dir = 'data'
patient_demo_file = os.path.join(data_dir, 'patient_demographics.csv')
diagnoses_file = os.path.join(data_dir, 'diagnoses.csv')
prescriptions_file = os.path.join(data_dir, 'prescriptions.csv')
medical_events_file = os.path.join(data_dir, 'medical_events.csv')
costs_file = os.path.join(data_dir, 'costs.csv')

def load_dataset():
    """
    加载所有CSV数据文件到pandas DataFrame
    """
    print("开始加载数据集...")
    
    # 加载患者人口统计学信息
    patient_demo_df = pd.read_csv(patient_demo_file)
    print(f"患者人口统计学数据加载完成: {patient_demo_df.shape}")
    
    # 加载诊断记录
    diagnoses_df = pd.read_csv(diagnoses_file)
    print(f"诊断记录数据加载完成: {diagnoses_df.shape}")
    
    # 加载处方记录
    prescriptions_df = pd.read_csv(prescriptions_file)
    print(f"处方记录数据加载完成: {prescriptions_df.shape}")
    
    # 加载医疗事件记录
    medical_events_df = pd.read_csv(medical_events_file)
    print(f"医疗事件记录数据加载完成: {medical_events_df.shape}")
    
    # 加载费用记录
    costs_df = pd.read_csv(costs_file)
    print(f"费用记录数据加载完成: {costs_df.shape}")
    
    print("所有数据集加载完成!")
    
    return {
        'patient_demo': patient_demo_df,
        'diagnoses': diagnoses_df,
        'prescriptions': prescriptions_df,
        'medical_events': medical_events_df,
        'costs': costs_df
    }

def get_target_patients(prescriptions_df):
    """
    识别有目标药物处方记录的患者
    """
    target_patients = prescriptions_df[prescriptions_df['is_target_drug'] == True]['patient_id'].unique()
    print(f"有目标药物处方记录的患者数: {len(target_patients)}")
    return target_patients

def get_target_diagnosis_patients(diagnoses_df, start_year=2018, end_year=2023):
    """
    识别在指定年份范围内首次诊断目标疾病的患者
    """
    # 将诊断日期转换为datetime类型
    diagnoses_df['diagnosis_date'] = pd.to_datetime(diagnoses_df['diagnosis_date'])
    
    # 筛选目标疾病诊断
    target_diagnoses = diagnoses_df[diagnoses_df['is_target_disease'] == True].copy()
    
    # 筛选在指定年份范围内的诊断
    target_diagnoses = target_diagnoses[
        (target_diagnoses['diagnosis_date'].dt.year >= start_year) & 
        (target_diagnoses['diagnosis_date'].dt.year <= end_year)
    ]
    
    # 获取每个患者的最早诊断日期
    earliest_diagnosis = target_diagnoses.groupby('patient_id')['diagnosis_date'].min().reset_index()
    earliest_diagnosis.columns = ['patient_id', 'first_diagnosis_date']
    
    print(f"在{start_year}-{end_year}年间首次诊断目标疾病的患者数: {len(earliest_diagnosis)}")
    return earliest_diagnosis

if __name__ == "__main__":
    # 加载数据集
    datasets = load_dataset()
    
    # 获取有目标药物处方记录的患者
    target_patients = get_target_patients(datasets['prescriptions'])
    
    # 获取在指定年份范围内首次诊断目标疾病的患者
    earliest_diagnosis = get_target_diagnosis_patients(datasets['diagnoses'])
    
    # 找到同时满足两个条件的患者
    eligible_patients = set(target_patients).intersection(set(earliest_diagnosis['patient_id']))
    print(f"同时满足条件的患者数: {len(eligible_patients)}")