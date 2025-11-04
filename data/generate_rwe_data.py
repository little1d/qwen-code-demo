"""
RWE数据生成脚本
生成模拟的真实世界证据研究数据，用于测试分析能力
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

# 设置随机种子以确保可重现性
np.random.seed(42)
random.seed(42)

# 参数设置
N_PATIENTS = 2000  # 总患者数
STUDY_START = datetime(2018, 1, 1)
STUDY_END = datetime(2023, 12, 31)
FOLLOW_UP_DAYS = 365

# 定义药物编码
TREATMENT_A = "DRUG_A"  # 新治疗方案
TREATMENT_B = "DRUG_B"  # 标准治疗方案

# 生成患者ID列表
patient_ids = [f"P{i:06d}" for i in range(1, N_PATIENTS + 1)]

# 1. 生成患者人口统计学数据
def generate_demographics():
    """生成患者人口统计学信息"""
    data = {
        'patient_id': patient_ids,
        'age': np.random.normal(60, 12, N_PATIENTS).astype(int).clip(18, 100),
        'gender': np.random.choice(['M', 'F'], N_PATIENTS, p=[0.52, 0.48]),
        'region': np.random.choice(['North', 'South', 'East', 'West', 'Central'], N_PATIENTS),
    }
    return pd.DataFrame(data)

# 2. 生成诊断记录
def generate_diagnoses(demographics):
    """生成诊断记录"""
    diagnoses_list = []
    
    # 常见合并症的ICD-10编码
    comorbidities = {
        'Hypertension': 'I10',
        'Diabetes': 'E11',
        'Hyperlipidemia': 'E78.5',
        'Coronary Heart Disease': 'I25.9',
        'Chronic Kidney Disease': 'N18.9',
        'Obesity': 'E66.9',
        'Depression': 'F32.9',
        'COPD': 'J44.9',
    }
    
    # 目标疾病（所有患者都有）
    target_disease = 'Type2_Diabetes'  # 假设为2型糖尿病
    target_icd = 'E11.9'
    
    for patient_id in patient_ids:
        # 目标疾病诊断（在研究期间内随机日期）
        dx_date = pd.Timestamp(np.random.choice(
            pd.date_range(STUDY_START, STUDY_END - timedelta(days=FOLLOW_UP_DAYS), freq='D')
        ))
        diagnoses_list.append({
            'patient_id': patient_id,
            'diagnosis_date': dx_date,
            'icd10_code': target_icd,
            'diagnosis_name': target_disease,
            'is_target_disease': True
        })
        
        # 合并症诊断（在目标疾病诊断前后随机分布）
        n_comorbidities = np.random.poisson(2)  # 平均2个合并症
        selected_comorbidities = np.random.choice(
            list(comorbidities.keys()), 
            size=min(n_comorbidities, len(comorbidities)), 
            replace=False
        )
        
        for comorbidity in selected_comorbidities:
            comorbidity_date = dx_date + timedelta(days=np.random.randint(-365, 0))
            diagnoses_list.append({
                'patient_id': patient_id,
                'diagnosis_date': comorbidity_date,
                'icd10_code': comorbidities[comorbidity],
                'diagnosis_name': comorbidity,
                'is_target_disease': False
            })
    
    return pd.DataFrame(diagnoses_list)

# 3. 生成处方记录
def generate_prescriptions(diagnoses):
    """生成处方记录"""
    prescriptions_list = []
    
    # 按患者分组，找到目标疾病诊断日期
    patient_dx_dates = diagnoses[diagnoses['is_target_disease']].groupby('patient_id')['diagnosis_date'].min()
    
    # 其他常见药物的编码
    other_drugs = ['ACE_INHIBITOR', 'STATIN', 'METFORMIN', 'ANTIHYPERTENSIVE']
    
    for patient_id in patient_ids:
        if patient_id not in patient_dx_dates.index:
            continue
            
        dx_date = patient_dx_dates[patient_id]
        
        # 确定患者使用的治疗方案（随机分配，但Treatment A效果更好）
        # 使用倾向性评分：年龄、性别影响分配
        demographics_subset = demographics[demographics['patient_id'] == patient_id].iloc[0]
        age_norm = (demographics_subset['age'] - 60) / 12  # 标准化
        gender_effect = 0.2 if demographics_subset['gender'] == 'M' else -0.2
        
        # Treatment A倾向性（新治疗方案通常用于更复杂的患者）
        p_treatment_a = 1 / (1 + np.exp(-(0.1 * age_norm + gender_effect + np.random.normal(0, 0.5))))
        treatment_group = TREATMENT_A if np.random.random() < p_treatment_a else TREATMENT_B
        
        # 首次用药日期（诊断后30-90天内）
        first_rx_date = dx_date + timedelta(days=np.random.randint(30, 90))
        
        # 生成目标药物的处方记录（12个月随访期内）
        current_date = first_rx_date
        days_covered = 0
        adherence = np.random.beta(8, 2)  # 依从性：beta分布，均值约80%
        
        while current_date <= first_rx_date + timedelta(days=FOLLOW_UP_DAYS):
            # 计算实际用药天数（考虑依从性）
            days_supply = 30  # 每次处方30天用量
            actual_days = int(days_supply * adherence + np.random.normal(0, 3))
            actual_days = max(1, min(actual_days, 60))  # 限制在1-60天之间
            
            prescriptions_list.append({
                'patient_id': patient_id,
                'prescription_date': current_date,
                'drug_code': treatment_group,
                'drug_name': treatment_group,
                'days_supply': actual_days,
                'quantity': 30,
                'is_target_drug': True,
                'treatment_group': treatment_group,
                'first_rx_date': first_rx_date
            })
            
            days_covered += actual_days
            
            # 下一次处方（考虑依从性和停药可能性）
            gap_days = np.random.exponential(5)  # 处方间隔的随机性
            # Treatment A依从性更好，停药率更低
            discontinuation_rate = 0.15 if treatment_group == TREATMENT_A else 0.25
            if np.random.random() < discontinuation_rate and days_covered > 180:
                # 停药（之后可能重新开始）
                restart_prob = 0.3 if treatment_group == TREATMENT_A else 0.2
                if np.random.random() > restart_prob:
                    break  # 永久停药
                else:
                    # 重新开始用药（60-120天后）
                    gap_days += np.random.randint(60, 120)
            
            current_date += timedelta(days=int(actual_days + gap_days))
        
        # 生成伴随用药
        n_concomitant = np.random.poisson(2)
        for _ in range(n_concomitant):
            other_drug = np.random.choice(other_drugs)
            rx_date = first_rx_date + timedelta(days=np.random.randint(0, FOLLOW_UP_DAYS))
            prescriptions_list.append({
                'patient_id': patient_id,
                'prescription_date': rx_date,
                'drug_code': other_drug,
                'drug_name': other_drug,
                'days_supply': 30,
                'quantity': 30,
                'is_target_drug': False,
                'treatment_group': treatment_group,
                'first_rx_date': first_rx_date
            })
    
    return pd.DataFrame(prescriptions_list)

# 4. 生成医疗事件（急诊、住院、门诊）
def generate_medical_events(demographics, prescriptions):
    """生成医疗事件记录"""
    events_list = []
    
    # 获取每个患者的首次用药日期和分组
    patient_info = prescriptions[prescriptions['is_target_drug']].groupby('patient_id').agg({
        'first_rx_date': 'first',
        'treatment_group': 'first'
    }).reset_index()
    
    for _, row in patient_info.iterrows():
        patient_id = row['patient_id']
        first_rx_date = row['first_rx_date']
        treatment_group = row['treatment_group']
        
        baseline_end = first_rx_date
        follow_up_end = first_rx_date + timedelta(days=FOLLOW_UP_DAYS)
        
        # Treatment A效果更好，医疗资源利用更低
        base_er_rate = 0.8 if treatment_group == TREATMENT_A else 1.2  # 急诊率
        base_hosp_rate = 0.3 if treatment_group == TREATMENT_A else 0.5  # 住院率
        base_op_rate = 4.0 if treatment_group == TREATMENT_A else 5.5  # 门诊率
        
        # 生成急诊事件
        n_er = np.random.poisson(base_er_rate)
        for _ in range(n_er):
            event_date = pd.Timestamp(np.random.choice(
                pd.date_range(first_rx_date, follow_up_end, freq='D')
            ))
            events_list.append({
                'patient_id': patient_id,
                'event_date': event_date,
                'event_type': 'ER',
                'treatment_group': treatment_group
            })
        
        # 生成住院事件
        n_hosp = np.random.poisson(base_hosp_rate)
        for _ in range(n_hosp):
            event_date = pd.Timestamp(np.random.choice(
                pd.date_range(first_rx_date, follow_up_end, freq='D')
            ))
            length_of_stay = np.random.lognormal(1.5, 0.8)  # 住院天数
            length_of_stay = max(1, min(int(length_of_stay), 30))
            
            events_list.append({
                'patient_id': patient_id,
                'event_date': event_date,
                'event_type': 'Hospitalization',
                'length_of_stay': length_of_stay,
                'treatment_group': treatment_group
            })
        
        # 生成门诊事件
        n_op = np.random.poisson(base_op_rate)
        for _ in range(n_op):
            event_date = pd.Timestamp(np.random.choice(
                pd.date_range(first_rx_date, follow_up_end, freq='D')
            ))
            events_list.append({
                'patient_id': patient_id,
                'event_date': event_date,
                'event_type': 'Outpatient',
                'treatment_group': treatment_group
            })
    
    return pd.DataFrame(events_list)

# 5. 生成费用记录
def generate_costs(demographics, prescriptions, medical_events):
    """生成费用记录"""
    costs_list = []
    
    # 获取每个患者的首次用药日期和分组
    patient_info = prescriptions[prescriptions['is_target_drug']].groupby('patient_id').agg({
        'first_rx_date': 'first',
        'treatment_group': 'first'
    }).reset_index()
    
    for _, row in patient_info.iterrows():
        patient_id = row['patient_id']
        first_rx_date = row['first_rx_date']
        treatment_group = row['treatment_group']
        follow_up_end = first_rx_date + timedelta(days=FOLLOW_UP_DAYS)
        
        # 处方费用（目标药物）
        target_rx = prescriptions[
            (prescriptions['patient_id'] == patient_id) & 
            (prescriptions['is_target_drug']) &
            (prescriptions['prescription_date'] >= first_rx_date) &
            (prescriptions['prescription_date'] <= follow_up_end)
        ]
        
        # Treatment A价格更高，但总成本可能更低（因为减少住院）
        rx_unit_cost = 150 if treatment_group == TREATMENT_A else 100
        for _, rx in target_rx.iterrows():
            costs_list.append({
                'patient_id': patient_id,
                'cost_date': rx['prescription_date'],
                'cost_type': 'Prescription',
                'cost_amount': rx_unit_cost * rx['quantity'],
                'treatment_group': treatment_group
            })
        
        # 伴随用药费用
        concomitant_rx = prescriptions[
            (prescriptions['patient_id'] == patient_id) & 
            (~prescriptions['is_target_drug']) &
            (prescriptions['prescription_date'] >= first_rx_date) &
            (prescriptions['prescription_date'] <= follow_up_end)
        ]
        for _, rx in concomitant_rx.iterrows():
            costs_list.append({
                'patient_id': patient_id,
                'cost_date': rx['prescription_date'],
                'cost_type': 'Prescription_Concomitant',
                'cost_amount': np.random.uniform(20, 80),
                'treatment_group': treatment_group
            })
        
        # 医疗事件费用
        patient_events = medical_events[medical_events['patient_id'] == patient_id]
        for _, event in patient_events.iterrows():
            if event['event_type'] == 'ER':
                cost_amount = np.random.uniform(500, 2000)
            elif event['event_type'] == 'Hospitalization':
                cost_amount = event.get('length_of_stay', 5) * np.random.uniform(1000, 3000)
            else:  # Outpatient
                cost_amount = np.random.uniform(100, 500)
            
            costs_list.append({
                'patient_id': patient_id,
                'cost_date': event['event_date'],
                'cost_type': event['event_type'],
                'cost_amount': cost_amount,
                'treatment_group': treatment_group
            })
    
    return pd.DataFrame(costs_list)

# 主函数
if __name__ == "__main__":
    print("Generating RWE dataset...")
    
    # 生成各个数据表
    print("1. Generating demographics...")
    demographics = generate_demographics()
    
    print("2. Generating diagnoses...")
    diagnoses = generate_diagnoses(demographics)
    
    print("3. Generating prescriptions...")
    prescriptions = generate_prescriptions(diagnoses)
    
    print("4. Generating medical events...")
    medical_events = generate_medical_events(demographics, prescriptions)
    
    print("5. Generating costs...")
    costs = generate_costs(demographics, prescriptions, medical_events)
    
    # 保存数据
    print("6. Saving data files...")
    demographics.to_csv('patient_demographics.csv', index=False)
    diagnoses.to_csv('diagnoses.csv', index=False)
    prescriptions.to_csv('prescriptions.csv', index=False)
    medical_events.to_csv('medical_events.csv', index=False)
    costs.to_csv('costs.csv', index=False)
    
    # 生成数据字典
    data_dictionary = {
        'patient_demographics.csv': {
            'description': '患者人口统计学信息',
            'columns': {
                'patient_id': '患者唯一标识符',
                'age': '年龄',
                'gender': '性别 (M/F)',
                'region': '地区'
            }
        },
        'diagnoses.csv': {
            'description': '诊断记录',
            'columns': {
                'patient_id': '患者唯一标识符',
                'diagnosis_date': '诊断日期',
                'icd10_code': 'ICD-10编码',
                'diagnosis_name': '诊断名称',
                'is_target_disease': '是否为目标疾病 (True/False)'
            }
        },
        'prescriptions.csv': {
            'description': '处方记录',
            'columns': {
                'patient_id': '患者唯一标识符',
                'prescription_date': '处方日期',
                'drug_code': '药物编码',
                'drug_name': '药物名称',
                'days_supply': '供应天数',
                'quantity': '数量',
                'is_target_drug': '是否为目标药物 (True/False)',
                'treatment_group': '治疗组 (DRUG_A/DRUG_B)',
                'first_rx_date': '首次用药日期'
            }
        },
        'medical_events.csv': {
            'description': '医疗事件记录',
            'columns': {
                'patient_id': '患者唯一标识符',
                'event_date': '事件日期',
                'event_type': '事件类型 (ER/Hospitalization/Outpatient)',
                'length_of_stay': '住院天数 (仅住院事件)',
                'treatment_group': '治疗组'
            }
        },
        'costs.csv': {
            'description': '费用记录',
            'columns': {
                'patient_id': '患者唯一标识符',
                'cost_date': '费用发生日期',
                'cost_type': '费用类型 (Prescription/Prescription_Concomitant/ER/Hospitalization/Outpatient)',
                'cost_amount': '费用金额',
                'treatment_group': '治疗组'
            }
        }
    }
    
    import json
    with open('data_dictionary.json', 'w', encoding='utf-8') as f:
        json.dump(data_dictionary, f, ensure_ascii=False, indent=2)
    
    print("\nData generation complete!")
    print(f"Generated {len(demographics)} patients")
    print(f"Generated {len(diagnoses)} diagnosis records")
    print(f"Generated {len(prescriptions)} prescription records")
    print(f"Generated {len(medical_events)} medical events")
    print(f"Generated {len(costs)} cost records")
    print("\nFiles saved:")
    print("  - patient_demographics.csv")
    print("  - diagnoses.csv")
    print("  - diagnoses.csv")
    print("  - prescriptions.csv")
    print("  - medical_events.csv")
    print("  - costs.csv")
    print("  - data_dictionary.json")

