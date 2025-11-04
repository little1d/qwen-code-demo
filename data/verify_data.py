"""
验证生成的数据是否符合预期
"""

import pandas as pd
import os

def verify_data():
    """验证数据文件是否存在且格式正确"""
    
    files = [
        'patient_demographics.csv',
        'diagnoses.csv',
        'prescriptions.csv',
        'medical_events.csv',
        'costs.csv'
    ]
    
    print("验证数据文件...")
    
    for file in files:
        if not os.path.exists(file):
            print(f"❌ 文件不存在: {file}")
            return False
        else:
            df = pd.read_csv(file)
            print(f"✅ {file}: {len(df)} 行, {len(df.columns)} 列")
    
    # 检查关键字段
    print("\n检查关键字段...")
    
    demographics = pd.read_csv('patient_demographics.csv')
    if 'patient_id' not in demographics.columns:
        print("❌ patient_demographics.csv 缺少 patient_id 字段")
        return False
    
    prescriptions = pd.read_csv('prescriptions.csv')
    required_cols = ['patient_id', 'treatment_group', 'is_target_drug', 'first_rx_date']
    missing = [col for col in required_cols if col not in prescriptions.columns]
    if missing:
        print(f"❌ prescriptions.csv 缺少字段: {missing}")
        return False
    
    # 检查治疗组分布
    print("\n检查治疗组分布...")
    target_rx = prescriptions[prescriptions['is_target_drug'] == True]
    if len(target_rx) > 0:
        treatment_dist = target_rx['treatment_group'].value_counts()
        print(f"  Treatment A: {treatment_dist.get('DRUG_A', 0)} 患者")
        print(f"  Treatment B: {treatment_dist.get('DRUG_B', 0)} 患者")
    
    # 检查日期范围
    print("\n检查日期范围...")
    prescriptions['prescription_date'] = pd.to_datetime(prescriptions['prescription_date'])
    date_range = prescriptions['prescription_date'].agg(['min', 'max'])
    print(f"  处方日期范围: {date_range['min']} 至 {date_range['max']}")
    
    print("\n✅ 数据验证通过！")
    return True

if __name__ == "__main__":
    verify_data()

