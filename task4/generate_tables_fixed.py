import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from scipy import stats
import os

# 导入生存分析所需的库
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
import matplotlib.pyplot as plt

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

def compare_resource_utilization(outcomes):
    """
    比较两组医疗资源利用差异
    """
    print("比较两组医疗资源利用差异...")
    
    # 分离两组数据
    drug_a = outcomes[outcomes['treatment_group'] == 'DRUG_A']
    drug_b = outcomes[outcomes['treatment_group'] == 'DRUG_B']
    
    # 比较急诊就诊次数
    print("\n急诊就诊次数比较:")
    print(f"DRUG_A组: 均值={drug_a['er_visits'].mean():.2f}, 标准差={drug_a['er_visits'].std():.2f}")
    print(f"DRUG_B组: 均值={drug_b['er_visits'].mean():.2f}, 标准差={drug_b['er_visits'].std():.2f}")
    
    # 检查数据分布，选择合适的统计检验
    if stats.shapiro(drug_a['er_visits']).pvalue > 0.05 and stats.shapiro(drug_b['er_visits']).pvalue > 0.05:
        # 数据符合正态分布，使用t检验
        t_stat, p_value = stats.ttest_ind(drug_a['er_visits'], drug_b['er_visits'])
        test_type = 't-test'
    else:
        # 数据不符合正态分布，使用Mann-Whitney U检验
        t_stat, p_value = stats.mannwhitneyu(drug_a['er_visits'], drug_b['er_visits'])
        test_type = 'Mann-Whitney U test'
    
    print(f"{test_type} p值: {p_value:.4f}")
    
    # 比较住院次数
    print("\n住院次数比较:")
    print(f"DRUG_A组: 均值={drug_a['hospitalizations'].mean():.2f}, 标准差={drug_a['hospitalizations'].std():.2f}")
    print(f"DRUG_B组: 均值={drug_b['hospitalizations'].mean():.2f}, 标准差={drug_b['hospitalizations'].std():.2f}")
    
    if stats.shapiro(drug_a['hospitalizations']).pvalue > 0.05 and stats.shapiro(drug_b['hospitalizations']).pvalue > 0.05:
        t_stat, p_value = stats.ttest_ind(drug_a['hospitalizations'], drug_b['hospitalizations'])
        test_type = 't-test'
    else:
        t_stat, p_value = stats.mannwhitneyu(drug_a['hospitalizations'], drug_b['hospitalizations'])
        test_type = 'Mann-Whitney U test'
    
    print(f"{test_type} p值: {p_value:.4f}")
    
    # 比较住院天数总和
    print("\n住院天数总和比较:")
    print(f"DRUG_A组: 均值={drug_a['total_hospital_days'].mean():.2f}, 标准差={drug_a['total_hospital_days'].std():.2f}")
    print(f"DRUG_B组: 均值={drug_b['total_hospital_days'].mean():.2f}, 标准差={drug_b['total_hospital_days'].std():.2f}")
    
    if stats.shapiro(drug_a['total_hospital_days']).pvalue > 0.05 and stats.shapiro(drug_b['total_hospital_days']).pvalue > 0.05:
        t_stat, p_value = stats.ttest_ind(drug_a['total_hospital_days'], drug_b['total_hospital_days'])
        test_type = 't-test'
    else:
        t_stat, p_value = stats.mannwhitneyu(drug_a['total_hospital_days'], drug_b['total_hospital_days'])
        test_type = 'Mann-Whitney U test'
    
    print(f"{test_type} p值: {p_value:.4f}")
    
    # 比较门诊就诊次数
    print("\n门诊就诊次数比较:")
    print(f"DRUG_A组: 均值={drug_a['outpatient_visits'].mean():.2f}, 标准差={drug_a['outpatient_visits'].std():.2f}")
    print(f"DRUG_B组: 均值={drug_b['outpatient_visits'].mean():.2f}, 标准差={drug_b['outpatient_visits'].std():.2f}")
    
    if stats.shapiro(drug_a['outpatient_visits']).pvalue > 0.05 and stats.shapiro(drug_b['outpatient_visits']).pvalue > 0.05:
        t_stat, p_value = stats.ttest_ind(drug_a['outpatient_visits'], drug_b['outpatient_visits'])
        test_type = 't-test'
    else:
        t_stat, p_value = stats.mannwhitneyu(drug_a['outpatient_visits'], drug_b['outpatient_visits'])
        test_type = 'Mann-Whitney U test'
    
    print(f"{test_type} p值: {p_value:.4f}")
    
    # 创建医疗资源利用比较结果表
    resource_comparison = pd.DataFrame({
        'outcome': ['ER Visits', 'Hospitalizations', 'Total Hospital Days', 'Outpatient Visits'],
        'drug_a_mean': [drug_a['er_visits'].mean(), drug_a['hospitalizations'].mean(), 
                        drug_a['total_hospital_days'].mean(), drug_a['outpatient_visits'].mean()],
        'drug_a_std': [drug_a['er_visits'].std(), drug_a['hospitalizations'].std(), 
                       drug_a['total_hospital_days'].std(), drug_a['outpatient_visits'].std()],
        'drug_b_mean': [drug_b['er_visits'].mean(), drug_b['hospitalizations'].mean(), 
                        drug_b['total_hospital_days'].mean(), drug_b['outpatient_visits'].mean()],
        'drug_b_std': [drug_b['er_visits'].std(), drug_b['hospitalizations'].std(), 
                       drug_b['total_hospital_days'].std(), drug_b['outpatient_visits'].std()],
    })
    
    return resource_comparison

def compare_cost_difference(outcomes):
    """
    比较两组成本差异
    """
    print("比较两组成本差异...")
    
    # 分离两组数据
    drug_a = outcomes[outcomes['treatment_group'] == 'DRUG_A']
    drug_b = outcomes[outcomes['treatment_group'] == 'DRUG_B']
    
    # 比较总医疗费用
    print("\n总医疗费用比较:")
    print(f"DRUG_A组: 均值={drug_a['total_cost'].mean():.2f}, 标准差={drug_a['total_cost'].std():.2f}")
    print(f"DRUG_B组: 均值={drug_b['total_cost'].mean():.2f}, 标准差={drug_b['total_cost'].std():.2f}")
    
    # 检查数据分布，选择合适的统计检验
    if stats.shapiro(drug_a['total_cost']).pvalue > 0.05 and stats.shapiro(drug_b['total_cost']).pvalue > 0.05:
        # 数据符合正态分布，使用t检验
        t_stat, p_value = stats.ttest_ind(drug_a['total_cost'], drug_b['total_cost'])
        test_type = 't-test'
    else:
        # 数据不符合正态分布，使用Mann-Whitney U检验
        t_stat, p_value = stats.mannwhitneyu(drug_a['total_cost'], drug_b['total_cost'])
        test_type = 'Mann-Whitney U test'
    
    print(f"{test_type} p值: {p_value:.4f}")
    
    # 比较处方费用
    print("\n处方费用比较:")
    print(f"DRUG_A组: 均值={drug_a['prescription_cost'].mean():.2f}, 标准差={drug_a['prescription_cost'].std():.2f}")
    print(f"DRUG_B组: 均值={drug_b['prescription_cost'].mean():.2f}, 标准差={drug_b['prescription_cost'].std():.2f}")
    
    if stats.shapiro(drug_a['prescription_cost']).pvalue > 0.05 and stats.shapiro(drug_b['prescription_cost']).pvalue > 0.05:
        t_stat, p_value = stats.ttest_ind(drug_a['prescription_cost'], drug_b['prescription_cost'])
        test_type = 't-test'
    else:
        t_stat, p_value = stats.mannwhitneyu(drug_a['prescription_cost'], drug_b['prescription_cost'])
        test_type = 'Mann-Whitney U test'
    
    print(f"{test_type} p值: {p_value:.4f}")
    
    # 创建成本比较结果表
    cost_comparison = pd.DataFrame({
        'outcome': ['Total Cost', 'Prescription Cost'],
        'drug_a_mean': [drug_a['total_cost'].mean(), drug_a['prescription_cost'].mean()],
        'drug_a_std': [drug_a['total_cost'].std(), drug_a['prescription_cost'].std()],
        'drug_b_mean': [drug_b['total_cost'].mean(), drug_b['prescription_cost'].mean()],
        'drug_b_std': [drug_b['total_cost'].std(), drug_b['prescription_cost'].std()],
        'p_value': [p_value, None]  # 为处方费用重新计算p值
    })
    
    # 重新计算处方费用的p值
    if stats.shapiro(drug_a['prescription_cost']).pvalue > 0.05 and stats.shapiro(drug_b['prescription_cost']).pvalue > 0.05:
        t_stat, p_value_rx = stats.ttest_ind(drug_a['prescription_cost'], drug_b['prescription_cost'])
    else:
        t_stat, p_value_rx = stats.mannwhitneyu(drug_a['prescription_cost'], drug_b['prescription_cost'])
    
    cost_comparison.loc[1, 'p_value'] = p_value_rx
    
    return cost_comparison

def calculate_pdc(data):
    """
    计算每个患者的PDC（Proportion of Days Covered）
    """
    print("计算每个患者的PDC...")
    
    study_pop = data['study_population']
    prescriptions = data['prescriptions']
    
    # 筛选目标药物的处方记录
    target_prescriptions = prescriptions[prescriptions['is_target_drug'] == True].copy()
    
    # 准备PDC计算数据
    pdc_results = []
    
    for patient_id in study_pop['patient_id']:
        # 获取当前患者的随访期
        followup_info = study_pop[study_pop['patient_id'] == patient_id].iloc[0]
        followup_start = followup_info['followup_start']
        followup_end = followup_info['followup_end']
        
        # 获取当前患者的处方记录
        patient_prescriptions = target_prescriptions[target_prescriptions['patient_id'] == patient_id].copy()
        patient_prescriptions = patient_prescriptions.sort_values('prescription_date')
        
        # 计算随访天数
        followup_days = (followup_end - followup_start).days + 1  # 包含起始日
        
        # 计算覆盖天数
        # 创建一个日期范围，然后标记被药物覆盖的天数
        date_range = pd.date_range(start=followup_start, end=followup_end, freq='D')
        covered_days = set()
        
        for idx, row in patient_prescriptions.iterrows():
            # 药物覆盖的开始日期
            drug_start = max(row['prescription_date'], followup_start)
            # 药物覆盖的结束日期
            drug_end = min(row['prescription_date'] + timedelta(days=row['days_supply']-1), followup_end)
            
            # 添加覆盖的日期到集合中
            current_date = drug_start
            while current_date <= drug_end:
                covered_days.add(current_date)
                current_date += timedelta(days=1)
        
        covered_days_count = len(covered_days)
        pdc = (covered_days_count / followup_days) * 100 if followup_days > 0 else 0
        
        pdc_results.append({
            'patient_id': patient_id,
            'pdc': pdc,
            'covered_days': covered_days_count,
            'followup_days': followup_days
        })
    
    pdc_df = pd.DataFrame(pdc_results)
    
    # 与治疗组信息合并
    pdc_df = pdc_df.merge(
        study_pop[['patient_id', 'treatment_group']], 
        on='patient_id', 
        how='left'
    )
    
    print(f"PDC计算完成，共计算 {len(pdc_df)} 名患者的PDC")
    
    return pdc_df

def calculate_good_adherence(pdc_df):
    """
    计算PDC≥80%的患者比例
    """
    print("计算PDC≥80%的患者比例...")
    
    # 计算PDC≥80%的患者
    good_adherence_patients = pdc_df[pdc_df['pdc'] >= 80]
    good_adherence_pct = (len(good_adherence_patients) / len(pdc_df)) * 100
    
    print(f"PDC≥80%的患者数量: {len(good_adherence_patients)}")
    print(f"PDC≥80%的患者比例: {good_adherence_pct:.2f}%")
    
    return good_adherence_patients, good_adherence_pct

def calculate_discontinuation_time(data):
    """
    定义停药并计算治疗持续性
    """
    print("计算治疗持续性...")
    
    study_pop = data['study_population']
    prescriptions = data['prescriptions']
    
    # 筛选目标药物的处方记录
    target_prescriptions = prescriptions[prescriptions['is_target_drug'] == True].copy()
    
    # 定义停药：连续60天无目标药物处方
    discontinuation_days = 60
    
    discontinuation_results = []
    
    for patient_id in study_pop['patient_id']:
        # 获取当前患者的随访期
        followup_info = study_pop[study_pop['patient_id'] == patient_id].iloc[0]
        followup_start = followup_info['followup_start']
        followup_end = followup_info['followup_end']
        
        # 获取当前患者的处方记录
        patient_prescriptions = target_prescriptions[target_prescriptions['patient_id'] == patient_id].copy()
        patient_prescriptions = patient_prescriptions.sort_values('prescription_date')
        
        if patient_prescriptions.empty:
            # 没有处方记录，视为立即停药
            discontinuation_results.append({
                'patient_id': patient_id,
                'discontinuation_time': 0,
                'censored': 0,  # 0表示停药，1表示审查
                'reason': 'no_prescriptions'
            })
            continue
        
        # 计算停药时间
        discontinuation_time = None
        last_prescription_date = patient_prescriptions.iloc[-1]['prescription_date']
        
        # 检查最后一次处方后是否达到停药标准
        if (followup_end - last_prescription_date).days >= discontinuation_days:
            # 在随访期内停药
            discontinuation_time = (last_prescription_date + timedelta(days=discontinuation_days) - followup_start).days
            if discontinuation_time > (followup_end - followup_start).days:
                discontinuation_time = (followup_end - followup_start).days  # 限制在随访期内
        else:
            # 随访期结束时尚未停药，进行审查
            discontinuation_time = (followup_end - followup_start).days
            discontinuation_results.append({
                'patient_id': patient_id,
                'discontinuation_time': discontinuation_time,
                'censored': 1,  # 1表示审查（未观察到停药事件）
                'reason': 'censored'
            })
            continue
        
        # 如果确实发生了停药
        if discontinuation_time is not None:
            discontinuation_results.append({
                'patient_id': patient_id,
                'discontinuation_time': discontinuation_time,
                'censored': 0,  # 0表示停药（观察到停药事件）
                'reason': 'discontinued'
            })
        else:
            # 未发生停药但在随访期结束
            discontinuation_time = (followup_end - followup_start).days
            discontinuation_results.append({
                'patient_id': patient_id,
                'discontinuation_time': discontinuation_time,
                'censored': 1,  # 1表示审查
                'reason': 'censored_end_followup'
            })
    
    discontinuation_df = pd.DataFrame(discontinuation_results)
    
    # 与治疗组信息合并
    discontinuation_df = discontinuation_df.merge(
        study_pop[['patient_id', 'treatment_group']], 
        on='patient_id', 
        how='left'
    )
    
    print(f"治疗持续性计算完成，共处理 {len(discontinuation_df)} 名患者")
    print(f"停药患者数: {len(discontinuation_df[discontinuation_df['censored'] == 0])}")
    print(f"审查患者数: {len(discontinuation_df[discontinuation_df['censored'] == 1])}")
    
    return discontinuation_df

def perform_kaplan_meier_analysis(discontinuation_df):
    """
    使用Kaplan-Meier方法估计停药风险
    """
    print("执行Kaplan-Meier生存分析...")
    
    # 创建Kaplan-Meier估计器
    kmf = KaplanMeierFitter()
    
    # 按治疗组进行Kaplan-Meier分析
    treatment_groups = discontinuation_df['treatment_group'].unique()
    
    plt.figure(figsize=(10, 6))
    
    for treatment in treatment_groups:
        group_data = discontinuation_df[discontinuation_df['treatment_group'] == treatment]
        T = group_data['discontinuation_time']  # 生存时间
        E = 1 - group_data['censored']  # 事件指示器 (1=事件发生, 0=审查)
        
        # 拟合Kaplan-Meier模型
        kmf.fit(T, event_observed=E, label=f'{treatment}')
        
        # 绘制生存曲线
        ax = kmf.plot_survival_function()
        
        print(f"\n{ treatment } 组的生存分析结果:")
        print(f"  样本数: {len(group_data)}")
        print(f"  停药数: {int(E.sum())}")
        print(f"  审查数: {int(len(E) - E.sum())}")
        print(f"  180天生存概率: {kmf.survival_function_at_times([180])[0] if len(kmf.survival_function_at_times([180])) > 0 else 'N/A'}")
    
    plt.title('Kaplan-Meier 生存曲线 - 按治疗组')
    plt.xlabel('时间 (天)')
    plt.ylabel('继续治疗的概率')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # 保存图形
    os.makedirs('task4/figures', exist_ok=True)
    plt.savefig('task4/figures/kaplan_meier_survival_curve.png', dpi=300, bbox_inches='tight')
    plt.close()  # 关闭图形以避免显示
    
    return kmf

def perform_logrank_test(discontinuation_df):
    """
    执行Log-rank检验
    """
    print("执行Log-rank检验...")
    
    # 获取不同治疗组的数据
    groups = discontinuation_df['treatment_group'].unique()
    
    if len(groups) < 2:
        print("只有一组数据，无法执行Log-rank检验")
        return None
    
    # 提取第一组和第二组的数据
    group1_data = discontinuation_df[discontinuation_df['treatment_group'] == groups[0]]
    group2_data = discontinuation_df[discontinuation_df['treatment_group'] == groups[1]]
    
    # 准备生存时间和事件指示器
    T1 = group1_data['discontinuation_time']
    E1 = 1 - group1_data['censored']
    
    T2 = group2_data['discontinuation_time']
    E2 = 1 - group2_data['censored']
    
    # 执行Log-rank检验
    results = logrank_test(T1, T2, E1, E2)
    
    print(f"\nLog-rank检验结果:")
    print(f"  Chi-square 统计量: {results.test_statistic:.4f}")
    print(f"  P值: {results.p_value:.4f}")
    print(f"  是否显著 (α=0.05): {'是' if results.p_value < 0.05 else '否'}")
    
    return results

def generate_summary_tables(data):
    """
    生成所有统计结果表
    """
    print("生成所有统计结果表...")
    
    # 计算所有结局指标
    outcomes = calculate_outcomes(data)
    
    # 比较医疗资源利用
    resource_comparison = compare_resource_utilization(outcomes)
    
    # 比较成本差异
    cost_comparison = compare_cost_difference(outcomes)
    
    # 计算PDC
    pdc_df = calculate_pdc(data)
    
    # 计算高依从性患者比例
    good_adherence_patients, good_adherence_pct = calculate_good_adherence(pdc_df)
    
    # 计算停药数据
    discontinuation_df = calculate_discontinuation_time(data)
    
    # 生成主要结局指标总结表
    main_outcomes_summary = pd.DataFrame({
        'outcome': [
            'ER Visits', 'Hospitalizations', 'Total Hospital Days', 'Outpatient Visits',
            'Total Cost', 'Prescription Cost', 'PDC≥80% Proportion', 'Discontinuation Rate'
        ],
        'drug_a_mean': [
            outcomes[outcomes['treatment_group'] == 'DRUG_A']['er_visits'].mean(),
            outcomes[outcomes['treatment_group'] == 'DRUG_A']['hospitalizations'].mean(),
            outcomes[outcomes['treatment_group'] == 'DRUG_A']['total_hospital_days'].mean(),
            outcomes[outcomes['treatment_group'] == 'DRUG_A']['outpatient_visits'].mean(),
            outcomes[outcomes['treatment_group'] == 'DRUG_A']['total_cost'].mean(),
            outcomes[outcomes['treatment_group'] == 'DRUG_A']['prescription_cost'].mean(),
            (pdc_df[pdc_df['treatment_group'] == 'DRUG_A']['pdc'] >= 80).mean(),
            1 - (discontinuation_df[(discontinuation_df['treatment_group'] == 'DRUG_A') & (discontinuation_df['censored'] == 0)].shape[0] / 
                 discontinuation_df[discontinuation_df['treatment_group'] == 'DRUG_A'].shape[0])
        ],
        'drug_b_mean': [
            outcomes[outcomes['treatment_group'] == 'DRUG_B']['er_visits'].mean(),
            outcomes[outcomes['treatment_group'] == 'DRUG_B']['hospitalizations'].mean(),
            outcomes[outcomes['treatment_group'] == 'DRUG_B']['total_hospital_days'].mean(),
            outcomes[outcomes['treatment_group'] == 'DRUG_B']['outpatient_visits'].mean(),
            outcomes[outcomes['treatment_group'] == 'DRUG_B']['total_cost'].mean(),
            outcomes[outcomes['treatment_group'] == 'DRUG_B']['prescription_cost'].mean(),
            (pdc_df[pdc_df['treatment_group'] == 'DRUG_B']['pdc'] >= 80).mean(),
            1 - (discontinuation_df[(discontinuation_df['treatment_group'] == 'DRUG_B') & (discontinuation_df['censored'] == 0)].shape[0] / 
                 discontinuation_df[discontinuation_df['treatment_group'] == 'DRUG_B'].shape[0])
        ],
        'difference': [
            outcomes[outcomes['treatment_group'] == 'DRUG_A']['er_visits'].mean() - outcomes[outcomes['treatment_group'] == 'DRUG_B']['er_visits'].mean(),
            outcomes[outcomes['treatment_group'] == 'DRUG_A']['hospitalizations'].mean() - outcomes[outcomes['treatment_group'] == 'DRUG_B']['hospitalizations'].mean(),
            outcomes[outcomes['treatment_group'] == 'DRUG_A']['total_hospital_days'].mean() - outcomes[outcomes['treatment_group'] == 'DRUG_B']['total_hospital_days'].mean(),
            outcomes[outcomes['treatment_group'] == 'DRUG_A']['outpatient_visits'].mean() - outcomes[outcomes['treatment_group'] == 'DRUG_B']['outpatient_visits'].mean(),
            outcomes[outcomes['treatment_group'] == 'DRUG_A']['total_cost'].mean() - outcomes[outcomes['treatment_group'] == 'DRUG_B']['total_cost'].mean(),
            outcomes[outcomes['treatment_group'] == 'DRUG_A']['prescription_cost'].mean() - outcomes[outcomes['treatment_group'] == 'DRUG_B']['prescription_cost'].mean(),
            (pdc_df[pdc_df['treatment_group'] == 'DRUG_A']['pdc'] >= 80).mean() - (pdc_df[pdc_df['treatment_group'] == 'DRUG_B']['pdc'] >= 80).mean(),
            ((discontinuation_df[(discontinuation_df['treatment_group'] == 'DRUG_B') & (discontinuation_df['censored'] == 0)].shape[0] / 
              discontinuation_df[discontinuation_df['treatment_group'] == 'DRUG_B'].shape[0]) -
             (discontinuation_df[(discontinuation_df['treatment_group'] == 'DRUG_A') & (discontinuation_df['censored'] == 0)].shape[0] / 
              discontinuation_df[discontinuation_df['treatment_group'] == 'DRUG_A'].shape[0]))
        ]
    })
    
    print("统计结果表生成完成！")
    
    return {
        'outcomes': outcomes,
        'resource_comparison': resource_comparison,
        'cost_comparison': cost_comparison,
        'pdc_df': pdc_df,
        'good_adherence_patients': good_adherence_patients,
        'discontinuation_df': discontinuation_df,
        'main_outcomes_summary': main_outcomes_summary
    }

if __name__ == "__main__":
    # 加载数据
    data = load_data()
    
    # 生成所有统计结果表
    results = generate_summary_tables(data)
    
    # 执行生存分析
    kmf = perform_kaplan_meier_analysis(results['discontinuation_df'])
    logrank_results = perform_logrank_test(results['discontinuation_df'])
    
    print("\n任务4所有分析完成！")
    
    # 保存所有结果
    os.makedirs('task4/tables', exist_ok=True)
    
    # 保存各个表格
    results['outcomes'].to_csv('task4/tables/outcomes.csv', index=False)
    results['resource_comparison'].to_csv('task4/tables/resource_utilization_comparison.csv', index=False)
    results['cost_comparison'].to_csv('task4/tables/cost_comparison.csv', index=False)
    results['pdc_df'].to_csv('task4/tables/pdc_results.csv', index=False)
    results['good_adherence_patients'].to_csv('task4/tables/good_adherence_patients.csv', index=False)
    results['discontinuation_df'].to_csv('task4/tables/discontinuation_analysis.csv', index=False)
    results['main_outcomes_summary'].to_csv('task4/tables/main_outcomes_summary.csv', index=False)
    
    # 保存Log-rank检验结果
    if logrank_results is not None:
        logrank_summary = pd.DataFrame({
            'test': ['Log-rank'],
            'statistic': [logrank_results.test_statistic],
            'p_value': [logrank_results.p_value],
            'significant': [logrank_results.p_value < 0.05]
        })
        logrank_summary.to_csv('task4/tables/logrank_test_results.csv', index=False)
    
    print("所有结果已保存至 task4/tables/ 和 task4/figures/ 目录")