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
    
    # 编码分类变量
    le_gender = LabelEncoder()
    baseline_data['gender_encoded'] = le_gender.fit_transform(baseline_data['gender'])
    
    le_region = LabelEncoder()
    baseline_data['region_encoded'] = le_region.fit_transform(baseline_data['region'])
    
    # 将治疗组也进行编码（DRUG_A=0, DRUG_B=1）
    le_treatment = LabelEncoder()
    baseline_data['treatment_encoded'] = le_treatment.fit_transform(baseline_data['treatment_group'])
    
    print("数据加载完成")
    print(f"最终基线数据形状: {baseline_data.shape}")
    print(f"治疗组分布:\n{baseline_data['treatment_group'].value_counts()}")
    
    return baseline_data, le_gender, le_region, le_treatment

def build_propensity_score_model(baseline_data):
    """
    使用逻辑回归建立倾向性评分模型
    """
    print("构建倾向性评分模型...")
    
    # 选择协变量（年龄、性别、地区、合并症数量）
    features = ['age', 'gender_encoded', 'region_encoded', 'comorbidity_count']
    X = baseline_data[features]
    y = baseline_data['treatment_encoded']  # DRUG_A=0, DRUG_B=1
    
    # 构建逻辑回归模型
    lr_model = LogisticRegression(random_state=42)
    lr_model.fit(X, y)
    
    # 预测倾向性评分
    propensity_scores = lr_model.predict_proba(X)[:, 1]  # 取属于DRUG_B组的概率
    baseline_data['propensity_score'] = propensity_scores
    
    # 输出模型系数
    print("倾向性评分模型系数:")
    for feature, coef in zip(features, lr_model.coef_[0]):
        print(f"  {feature}: {coef:.4f}")
    
    print(f"截距: {lr_model.intercept_[0]:.4f}")
    
    # 计算模型的AUC
    from sklearn.metrics import roc_auc_score
    auc = roc_auc_score(y, propensity_scores)
    print(f"模型AUC: {auc:.4f}")
    
    return lr_model, baseline_data

def perform_matching(baseline_data, caliper=0.1):
    """
    执行1:1最近邻匹配
    """
    print(f"执行1:1最近邻匹配 (caliper={caliper})...")
    
    # 分离处理组和对照组
    treated = baseline_data[baseline_data['treatment_group'] == 'DRUG_A'].copy()
    control = baseline_data[baseline_data['treatment_group'] == 'DRUG_B'].copy()
    
    print(f"处理组 (DRUG_A) 患者数: {len(treated)}")
    print(f"对照组 (DRUG_B) 患者数: {len(control)}")
    
    # 初始化匹配结果
    matched_pairs = []
    treated_matched = set()
    control_matched = set()
    
    # 为每个处理组患者找到最接近的对照组患者
    for _, treated_row in treated.iterrows():
        if treated_row['patient_id'] in treated_matched:
            continue
            
        treated_score = treated_row['propensity_score']
        
        # 计算与所有未匹配对照组患者的评分差异
        unmatched_control = control[~control['patient_id'].isin(control_matched)]
        score_diffs = np.abs(unmatched_control['propensity_score'] - treated_score)
        
        # 寻找差异最小且在caliper范围内的对照组患者
        valid_controls = unmatched_control[score_diffs <= caliper]
        
        if not valid_controls.empty:
            # 选择评分差异最小的患者
            closest_control_idx = valid_controls.iloc[score_diffs[valid_controls.index].idxmin()].name
            closest_control = control.loc[closest_control_idx]
            
            # 记录匹配对
            matched_pairs.append({
                'treated_patient_id': treated_row['patient_id'],
                'control_patient_id': closest_control['patient_id'],
                'treated_propensity_score': treated_row['propensity_score'],
                'control_propensity_score': closest_control['propensity_score'],
                'treated_age': treated_row['age'],
                'control_age': closest_control['age'],
                'treated_gender': treated_row['gender'],
                'control_gender': closest_control['gender'],
                'treated_region': treated_row['region'],
                'control_region': closest_control['region'],
                'treated_comorbidity_count': treated_row['comorbidity_count'],
                'control_comorbidity_count': closest_control['comorbidity_count']
            })
            
            # 标记为已匹配
            treated_matched.add(treated_row['patient_id'])
            control_matched.add(closest_control['patient_id'])
    
    matched_df = pd.DataFrame(matched_pairs)
    print(f"成功匹配的对数: {len(matched_df)}")
    
    return matched_df

if __name__ == "__main__":
    # 加载数据
    baseline_data, le_gender, le_region, le_treatment = load_data()
    
    # 构建倾向性评分模型
    propensity_model, baseline_data_with_scores = build_propensity_score_model(baseline_data)
    
    # 执行匹配
    matched_data = perform_matching(baseline_data_with_scores, caliper=0.1)
    
    print("\n倾向性评分匹配完成！")
    
    # 保存倾向性评分模型结果
    os.makedirs('task3/tables', exist_ok=True)
    baseline_data_with_scores[['patient_id', 'treatment_group', 'propensity_score']].to_csv(
        'task3/tables/propensity_scores.csv', 
        index=False
    )
    
    # 保存匹配结果
    matched_data.to_csv('task3/tables/matched_pairs.csv', index=False)
    print("匹配结果已保存至 task3/tables/matched_pairs.csv")