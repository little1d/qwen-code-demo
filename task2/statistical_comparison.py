import pandas as pd
import numpy as np
from scipy import stats
import os

def load_data():
    """
    加载任务1生成的数据集
    """
    print("加载任务1生成的数据集...")
    
    # 读取研究队列数据
    study_population = pd.read_csv('task1/study_population.csv')
    print(f"研究队列数据: {study_population.shape}")
    
    # 读取患者人口统计学数据
    patient_demo = pd.read_csv('task1/cleaned_patient_demo.csv')
    print(f"患者人口统计学数据: {patient_demo.shape}")
    
    # 读取诊断数据
    diagnoses = pd.read_csv('task1/cleaned_diagnoses.csv')
    print(f"诊断数据: {diagnoses.shape}")
    
    # 读取医疗事件数据
    medical_events = pd.read_csv('task1/cleaned_medical_events.csv')
    print(f"医疗事件数据: {medical_events.shape}")
    
    # 读取费用数据
    costs = pd.read_csv('task1/cleaned_costs.csv')
    print(f"费用数据: {costs.shape}")
    
    # 读取处方数据
    prescriptions = pd.read_csv('task1/cleaned_prescriptions.csv')
    print(f"处方数据: {prescriptions.shape}")
    
    return {
        'study_population': study_population,
        'patient_demo': patient_demo,
        'diagnoses': diagnoses,
        'medical_events': medical_events,
        'costs': costs,
        'prescriptions': prescriptions
    }

def calculate_baseline_demographics(datasets):
    """
    计算两组的基线特征：年龄、性别、地区分布
    """
    print("计算基线人口统计学特征...")
    
    # 合并研究队列和人口统计学数据
    study_with_demo = datasets['study_population'][['patient_id', 'treatment_group']].merge(
        datasets['patient_demo'], 
        on='patient_id', 
        how='left'
    )
    
    # 按治疗组计算年龄统计
    age_stats = study_with_demo.groupby('treatment_group')['age'].agg(['count', 'mean', 'std', 'min', 'max']).round(2)
    age_stats.columns = ['n', 'mean_age', 'std_age', 'min_age', 'max_age']
    
    # 性别分布
    gender_dist = pd.crosstab(study_with_demo['treatment_group'], study_with_demo['gender'], margins=True)
    
    # 地区分布
    region_dist = pd.crosstab(study_with_demo['treatment_group'], study_with_demo['region'], margins=True)
    
    print("年龄统计:")
    print(age_stats)
    print("\n性别分布:")
    print(gender_dist)
    print("\n地区分布:")
    print(region_dist)
    
    return {
        'age_stats': age_stats,
        'gender_dist': gender_dist,
        'region_dist': region_dist,
        'study_with_demo': study_with_demo
    }

def identify_comorbidities(datasets):
    """
    从diagnoses表中识别合并症数量和类型（is_target_disease=False的记录）
    """
    print("识别合并症...")
    
    # 筛选非目标疾病的诊断记录
    comorbidities = datasets['diagnoses'][datasets['diagnoses']['is_target_disease'] == False].copy()
    
    # 将诊断日期转换为datetime类型
    comorbidities['diagnosis_date'] = pd.to_datetime(comorbidities['diagnosis_date'])
    
    # 获取研究队列的基线期信息
    baseline_info = datasets['study_population'][['patient_id', 'baseline_start', 'baseline_end']].copy()
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
    
    # 获取合并症类型
    patient_comorbidity_types = baseline_comorbidities.groupby('patient_id')['diagnosis_name'].apply(lambda x: ', '.join(x.unique())).reset_index()
    patient_comorbidity_types.columns = ['patient_id', 'comorbidity_types']
    
    # 合并所有信息
    patient_comorbidities = patient_comorbidity_counts.merge(patient_comorbidity_types, on='patient_id', how='outer')
    patient_comorbidities = patient_comorbidities.fillna(0)  # 没有合并症的患者计数为0
    
    # 将合并症信息添加到研究队列中
    study_with_comorbidities = datasets['study_population'][['patient_id', 'treatment_group']].merge(
        patient_comorbidities, 
        on='patient_id', 
        how='left'
    )
    study_with_comorbidities['comorbidity_count'] = study_with_comorbidities['comorbidity_count'].fillna(0)
    study_with_comorbidities['comorbidity_types'] = study_with_comorbidities['comorbidity_types'].fillna('None')
    
    # 按治疗组统计合并症
    comorbidity_stats = study_with_comorbidities.groupby('treatment_group')['comorbidity_count'].agg(['count', 'mean', 'std', 'min', 'max']).round(2)
    comorbidity_stats.columns = ['n', 'mean_comorbidities', 'std_comorbidities', 'min_comorbidities', 'max_comorbidities']
    
    print("合并症统计:")
    print(comorbidity_stats)
    
    # 获取最常见的合并症类型
    all_comorbidities = baseline_comorbidities['diagnosis_name'].value_counts()
    print(f"\n最常见的10种合并症:")
    print(all_comorbidities.head(10))
    
    return {
        'comorbidity_stats': comorbidity_stats,
        'study_with_comorbidities': study_with_comorbidities,
        'all_comorbidities': all_comorbidities
    }

def calculate_baseline_resource_utilization(datasets):
    """
    计算基线期内的医疗资源利用情况
    """
    print("计算基线期内的医疗资源利用情况...")
    
    # 获取研究队列的基线期信息
    baseline_info = datasets['study_population'][['patient_id', 'baseline_start', 'baseline_end']].copy()
    baseline_info['baseline_start'] = pd.to_datetime(baseline_info['baseline_start'])
    baseline_info['baseline_end'] = pd.to_datetime(baseline_info['baseline_end'])
    
    # 处理医疗事件数据
    medical_events = datasets['medical_events'].copy()
    medical_events['event_date'] = pd.to_datetime(medical_events['event_date'])
    
    # 筛选基线期内的医疗事件
    baseline_events = medical_events.merge(baseline_info, on='patient_id', how='inner')
    baseline_events = baseline_events[
        (baseline_events['event_date'] >= baseline_events['baseline_start']) &
        (baseline_events['event_date'] <= baseline_events['baseline_end'])
    ]
    
    # 按事件类型统计
    event_counts = pd.crosstab(baseline_events['patient_id'], baseline_events['event_type'], margins=False)
    
    # 计算每位患者的各类事件次数
    patient_event_counts = baseline_events.groupby(['patient_id', 'event_type']).size().unstack(fill_value=0)
    
    # 添加到研究队列中
    study_with_events = datasets['study_population'][['patient_id', 'treatment_group']].merge(
        patient_event_counts, 
        left_on='patient_id', 
        right_index=True, 
        how='left'
    ).fillna(0)
    
    # 按治疗组统计各类事件的平均值
    event_stats = study_with_events.groupby('treatment_group')[event_counts.columns].mean().round(2) if not event_counts.empty else pd.DataFrame()
    
    print("基线期医疗事件统计（平均次数）:")
    print(event_stats)
    
    # 处理费用数据
    costs = datasets['costs'].copy()
    costs['cost_date'] = pd.to_datetime(costs['cost_date'])
    
    # 筛选基线期内的费用记录
    baseline_costs = costs.merge(baseline_info, on='patient_id', how='inner')
    baseline_costs = baseline_costs[
        (baseline_costs['cost_date'] >= baseline_costs['baseline_start']) &
        (baseline_costs['cost_date'] <= baseline_costs['baseline_end'])
    ]
    
    # 按费用类型统计
    cost_by_type = baseline_costs.groupby(['patient_id', 'cost_type'])['cost_amount'].sum().unstack(fill_value=0)
    
    # 总费用
    total_costs = baseline_costs.groupby('patient_id')['cost_amount'].sum()
    
    # 添加到研究队列中
    study_with_costs = datasets['study_population'][['patient_id', 'treatment_group']].merge(
        cost_by_type, 
        left_on='patient_id', 
        right_index=True, 
        how='left'
    ).merge(
        total_costs.to_frame('total_baseline_cost'),
        left_on='patient_id',
        right_index=True,
        how='left'
    ).fillna(0)
    
    # 按治疗组统计费用
    cost_stats = study_with_costs.groupby('treatment_group').agg({
        col: ['count', 'mean', 'std', 'min', 'max'] for col in ['total_baseline_cost'] + list(cost_by_type.columns) if col in study_with_costs.columns
    }).round(2)
    
    print("基线期费用统计（平均值）:")
    print(cost_stats)
    
    return {
        'event_stats': event_stats,
        'cost_stats': cost_stats,
        'study_with_events': study_with_events,
        'study_with_costs': study_with_costs
    }

def generate_descriptive_stats_tables(baseline_demo, comorbidities, resource_utilization, datasets):
    """
    生成基线特征描述性统计表
    """
    print("生成基线特征描述性统计表...")
    
    # 合并所有基线特征数据
    baseline_data = datasets['study_population'][['patient_id', 'treatment_group']].copy()
    
    # 添加人口统计学特征
    demo_data = datasets['study_population'][['patient_id', 'treatment_group']].merge(
        datasets['patient_demo'], 
        on='patient_id', 
        how='left'
    )
    baseline_data = baseline_data.merge(demo_data[['patient_id', 'age', 'gender', 'region']], on='patient_id', how='left')
    
    # 添加合并症信息
    baseline_data = baseline_data.merge(
        comorbidities['study_with_comorbidities'][['patient_id', 'comorbidity_count', 'comorbidity_types']], 
        on='patient_id', 
        how='left'
    )
    
    # 添加基线期资源利用信息
    baseline_with_events = datasets['study_population'][['patient_id', 'treatment_group']].merge(
        resource_utilization['study_with_events'], 
        on='patient_id', 
        how='left'
    )
    
    baseline_with_costs = datasets['study_population'][['patient_id', 'treatment_group']].merge(
        resource_utilization['study_with_costs'][['patient_id', 'total_baseline_cost']], 
        on='patient_id', 
        how='left'
    )
    
    # 合并所有数据
    baseline_data = baseline_data.merge(
        baseline_with_events.drop(columns=['treatment_group']), 
        on='patient_id', 
        how='left'
    ).merge(
        baseline_with_costs.drop(columns=['treatment_group']), 
        on='patient_id', 
        how='left'
    )
    
    # 初始化所有事件类型的列（如果不存在）
    for event_type in ['ER', 'Hospitalization', 'Outpatient']:
        if event_type not in baseline_data.columns:
            baseline_data[event_type] = 0
    
    baseline_data = baseline_data.fillna(0)
    
    # 为每个特征创建描述性统计表
    descriptive_stats = {}
    
    # 年龄描述性统计
    age_summary = baseline_data.groupby('treatment_group')['age'].agg(['count', 'mean', 'std', 'min', 'max']).round(2)
    age_summary.columns = ['n', 'mean', 'std', 'min', 'max']
    descriptive_stats['age'] = age_summary
    
    # 性别分布
    gender_summary = pd.crosstab(baseline_data['treatment_group'], baseline_data['gender'], margins=True)
    descriptive_stats['gender'] = gender_summary
    
    # 地区分布
    region_summary = pd.crosstab(baseline_data['treatment_group'], baseline_data['region'], margins=True)
    descriptive_stats['region'] = region_summary
    
    # 合并症数量
    comorbidity_summary = baseline_data.groupby('treatment_group')['comorbidity_count'].agg(['count', 'mean', 'std', 'min', 'max']).round(2)
    comorbidity_summary.columns = ['n', 'mean', 'std', 'min', 'max']
    descriptive_stats['comorbidity_count'] = comorbidity_summary
    
    # 基线期事件统计
    for event_type in ['ER', 'Hospitalization', 'Outpatient']:
        if event_type in baseline_data.columns:
            event_summary = baseline_data.groupby('treatment_group')[event_type].agg(['count', 'mean', 'std', 'min', 'max']).round(2)
            event_summary.columns = ['n', 'mean', 'std', 'min', 'max']
            descriptive_stats[f'baseline_{event_type}_events'] = event_summary
    
    # 基线期总费用
    cost_summary = baseline_data.groupby('treatment_group')['total_baseline_cost'].agg(['count', 'mean', 'std', 'min', 'max']).round(2)
    cost_summary.columns = ['n', 'mean', 'std', 'min', 'max']
    descriptive_stats['total_baseline_cost'] = cost_summary
    
    print("描述性统计表生成完成:")
    for name, table in descriptive_stats.items():
        print(f"\n{name}:")
        print(table)
    
    # 保存描述性统计表为CSV文件
    os.makedirs('task2/tables', exist_ok=True)
    for name, table in descriptive_stats.items():
        table.to_csv(f'task2/tables/descriptive_stats_{name}.csv')
    
    print(f"\n所有描述性统计表已保存至 task2/tables/ 目录")
    
    return descriptive_stats, baseline_data

def perform_statistical_comparison(baseline_summary):
    """
    进行两组基线特征比较（统计检验）
    """
    print("进行两组基线特征比较（统计检验）...")
    
    comparison_results = {}
    
    # 年龄比较 - t检验
    drug_a_age = baseline_summary[baseline_summary['treatment_group'] == 'DRUG_A']['age']
    drug_b_age = baseline_summary[baseline_summary['treatment_group'] == 'DRUG_B']['age']
    t_stat, p_value = stats.ttest_ind(drug_a_age, drug_b_age)
    comparison_results['age'] = {
        'test': 't-test',
        'statistic': t_stat,
        'p_value': p_value,
        'significant': p_value < 0.05
    }
    
    # 性别分布比较 - 卡方检验
    gender_crosstab = pd.crosstab(baseline_summary['treatment_group'], baseline_summary['gender'])
    chi2, p_value, dof, expected = stats.chi2_contingency(gender_crosstab)
    comparison_results['gender'] = {
        'test': 'Chi-squared test',
        'statistic': chi2,
        'p_value': p_value,
        'significant': p_value < 0.05
    }
    
    # 地区分布比较 - 卡方检验
    region_crosstab = pd.crosstab(baseline_summary['treatment_group'], baseline_summary['region'])
    chi2, p_value, dof, expected = stats.chi2_contingency(region_crosstab)
    comparison_results['region'] = {
        'test': 'Chi-squared test',
        'statistic': chi2,
        'p_value': p_value,
        'significant': p_value < 0.05
    }
    
    # 合并症数量比较 - t检验
    drug_a_comorb = baseline_summary[baseline_summary['treatment_group'] == 'DRUG_A']['comorbidity_count']
    drug_b_comorb = baseline_summary[baseline_summary['treatment_group'] == 'DRUG_B']['comorbidity_count']
    t_stat, p_value = stats.ttest_ind(drug_a_comorb, drug_b_comorb)
    comparison_results['comorbidity_count'] = {
        'test': 't-test',
        'statistic': t_stat,
        'p_value': p_value,
        'significant': p_value < 0.05
    }
    
    # 基线期急诊事件比较 - t检验
    if 'ER' in baseline_summary.columns:
        drug_a_er = baseline_summary[baseline_summary['treatment_group'] == 'DRUG_A']['ER']
        drug_b_er = baseline_summary[baseline_summary['treatment_group'] == 'DRUG_B']['ER']
        t_stat, p_value = stats.ttest_ind(drug_a_er, drug_b_er)
        comparison_results['baseline_ER_events'] = {
            'test': 't-test',
            'statistic': t_stat,
            'p_value': p_value,
            'significant': p_value < 0.05
        }
    
    # 基线期住院事件比较 - t检验
    if 'Hospitalization' in baseline_summary.columns:
        drug_a_hosp = baseline_summary[baseline_summary['treatment_group'] == 'DRUG_A']['Hospitalization']
        drug_b_hosp = baseline_summary[baseline_summary['treatment_group'] == 'DRUG_B']['Hospitalization']
        t_stat, p_value = stats.ttest_ind(drug_a_hosp, drug_b_hosp)
        comparison_results['baseline_Hospitalization_events'] = {
            'test': 't-test',
            'statistic': t_stat,
            'p_value': p_value,
            'significant': p_value < 0.05
        }
    
    # 基线期门诊事件比较 - t检验
    if 'Outpatient' in baseline_summary.columns:
        drug_a_outp = baseline_summary[baseline_summary['treatment_group'] == 'DRUG_A']['Outpatient']
        drug_b_outp = baseline_summary[baseline_summary['treatment_group'] == 'DRUG_B']['Outpatient']
        t_stat, p_value = stats.ttest_ind(drug_a_outp, drug_b_outp)
        comparison_results['baseline_Outpatient_events'] = {
            'test': 't-test',
            'statistic': t_stat,
            'p_value': p_value,
            'significant': p_value < 0.05
        }
    
    # 基线期总费用比较 - t检验
    drug_a_cost = baseline_summary[baseline_summary['treatment_group'] == 'DRUG_A']['total_baseline_cost']
    drug_b_cost = baseline_summary[baseline_summary['treatment_group'] == 'DRUG_B']['total_baseline_cost']
    t_stat, p_value = stats.ttest_ind(drug_a_cost, drug_b_cost)
    comparison_results['total_baseline_cost'] = {
        'test': 't-test',
        'statistic': t_stat,
        'p_value': p_value,
        'significant': p_value < 0.05
    }
    
    print("统计检验结果:")
    for feature, result in comparison_results.items():
        print(f"\n{feature}:")
        print(f"  检验方法: {result['test']}")
        print(f"  统计量: {result['statistic']:.4f}")
        print(f"  p值: {result['p_value']:.4f}")
        print(f"  是否显著: {'是' if result['significant'] else '否'}")
    
    return comparison_results

if __name__ == "__main__":
    # 加载数据集
    datasets = load_data()
    
    # 计算基线人口统计学特征
    baseline_demo = calculate_baseline_demographics(datasets)
    
    # 识别合并症
    comorbidities = identify_comorbidities(datasets)
    
    # 计算基线期资源利用
    resource_utilization = calculate_baseline_resource_utilization(datasets)
    
    # 生成描述性统计表
    descriptive_stats, baseline_summary = generate_descriptive_stats_tables(
        baseline_demo, 
        comorbidities, 
        resource_utilization, 
        datasets
    )
    
    # 进行统计比较
    comparison_results = perform_statistical_comparison(baseline_summary)
    
    print("\n任务2数据处理完成！")