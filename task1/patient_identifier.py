import pandas as pd
from datetime import datetime, timedelta
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

def identify_eligible_patients(diagnoses_df, prescriptions_df, start_year=2018, end_year=2023):
    """
    识别符合研究条件的患者：
    - 在指定年份范围内首次诊断目标疾病（is_target_disease=True）
    - 有目标药物处方记录（is_target_drug=True）
    """
    print("开始识别符合条件的患者...")
    
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
    
    # 获取有目标药物处方记录的患者
    target_patients = prescriptions_df[prescriptions_df['is_target_drug'] == True]['patient_id'].unique()
    print(f"有目标药物处方记录的患者数: {len(target_patients)}")
    
    # 找到同时满足两个条件的患者
    eligible_patients = set(earliest_diagnosis['patient_id']).intersection(set(target_patients))
    print(f"同时满足条件的患者数: {len(eligible_patients)}")
    
    # 筛选符合条件的患者数据
    eligible_diagnoses = earliest_diagnosis[earliest_diagnosis['patient_id'].isin(eligible_patients)]
    
    return eligible_diagnoses, prescriptions_df[prescriptions_df['patient_id'].isin(eligible_patients)]

def define_study_periods(prescriptions_df):
    """
    根据首次用药日期定义基线期（前90天）和随访期（后365天）
    """
    print("定义基线期和随访期...")
    
    # 获取每个患者的首次用药日期
    first_rx = prescriptions_df.groupby('patient_id')['first_rx_date'].min()
    first_rx = pd.to_datetime(first_rx)
    
    # 创建包含患者ID和首次用药日期的DataFrame
    study_patients = pd.DataFrame({
        'patient_id': first_rx.index,
        'first_rx_date': first_rx.values
    })
    
    # 定义基线期（首次用药前90天）
    study_patients['baseline_start'] = study_patients['first_rx_date'] - timedelta(days=90)
    study_patients['baseline_end'] = study_patients['first_rx_date'] - timedelta(days=1)  # 基线期结束为用药前一天
    
    # 定义随访期（首次用药后365天）
    study_patients['followup_start'] = study_patients['first_rx_date']
    study_patients['followup_end'] = study_patients['first_rx_date'] + timedelta(days=365)
    
    print(f"研究期定义完成，共 {len(study_patients)} 名患者")
    
    return study_patients

if __name__ == "__main__":
    # 加载数据集
    datasets = load_dataset()
    
    # 识别符合条件的患者
    eligible_diagnoses, eligible_prescriptions = identify_eligible_patients(
        datasets['diagnoses'], 
        datasets['prescriptions']
    )
    
    # 定义研究期
    study_patients = define_study_periods(eligible_prescriptions)
    
    # 检查是否有完整12个月随访数据的患者
    # 这里我们假设所有患者都有足够的随访时间，实际研究中需要根据数据截止日期判断
    print(f"研究患者队列构建完成，共 {len(study_patients)} 名患者")