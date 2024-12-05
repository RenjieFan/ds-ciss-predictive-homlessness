import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import warnings
warnings.filterwarnings('ignore')

# 1. 加载数据
print("加载数据...")
df = pd.read_csv("Merged_Data.csv")

# 2. 特征工程
def create_features(df):
    df = df.copy()
    
    # 基础特征处理
    population_features = ['B01003_001E', 'B17001_002E', 'B25002_001E']
    for col in population_features:
        df[f'{col}_K'] = df[col] / 1000
    
    # 创建规模组
    df['Size_Group'] = pd.qcut(df['B01003_001E'], q=3, labels=['Small', 'Medium', 'Large'])
    
    # 核心指标
    df['Housing_per_Capita'] = df['B25002_001E_K'] / df['B01003_001E_K']
    df['Poverty_Rate'] = (df['B17001_002E'] / df['B01003_001E']) * 100
    df['Housing_Stress'] = df['Cost_Burdened_Rate'] * df['Renter_Household_Rate'] / 100
    
    # 时间特征
    df['Year_Norm'] = df['Year'] - df['Year'].min()
    
    return df

# 3. 特征选择
def get_features():
    return [
        'B01003_001E_K',  # 总人口
        'B17001_002E_K',  # 贫困人口
        'B25002_001E_K',  # 住房单位
        'Housing_per_Capita',
        'Poverty_Rate',
        'Housing_Stress',
        'Unemployment_Rate',
        'Vacancy_Rate',
        'Renter_Household_Rate',
        'Cost_Burdened_Rate',
        'Average Temperature',
        'Year_Norm'
    ]

# 4. 目标变量
target_variables = [
    'Overall Homeless',
    'Overall Homeless Individuals',
    'Overall Homeless People in Families',
    'Unsheltered Homeless',
    'Sheltered Total Homeless'
]

# 5. 数据处理和模型训练函数
def train_model_for_group(X, y, group_name=""):
    # 分割数据
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # 标准化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 训练模型
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=8,
        min_samples_split=10,
        min_samples_leaf=6,
        max_features='sqrt',
        random_state=42,
        n_jobs=-1
    )
    
    # 交叉验证
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
    
    # 训练最终模型
    model.fit(X_train_scaled, y_train)
    
    # 预测
    train_pred = model.predict(X_train_scaled)
    test_pred = model.predict(X_test_scaled)
    
    # 计算指标
    metrics = {
        'train_r2': r2_score(y_train, train_pred),
        'test_r2': r2_score(y_test, test_pred),
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std(),
        'mse': mean_squared_error(y_test, test_pred),
        'model': model,
        'scaler': scaler,
        'predictions': test_pred,
        'actual': y_test
    }
    
    return metrics

print("处理数据...")
# 应用特征工程
df = create_features(df)
features = get_features()

# 存储结果
results = []

print("\n开始训练模型...")
for target in target_variables:
    print(f"\n预测目标: {target}")
    
    for size_group in ['Small', 'Medium', 'Large']:
        print(f"\n处理 {size_group} 规模组...")
        
        # 获取组数据
        group_df = df[df['Size_Group'] == size_group]
        
        # 准备特征和目标
        X = group_df[features].copy()
        y = group_df[target].copy()
        
        # 处理缺失值
        X = X.fillna(X.mean())
        y = y.fillna(y.mean())
        
        # 训练模型
        metrics = train_model_for_group(X, y, size_group)
        
        # 特征重要性
        importance = pd.DataFrame({
            'feature': features,
            'importance': metrics['model'].feature_importances_
        }).sort_values('importance', ascending=False)
        
        print(f"\n{size_group} 组性能:")
        print(f"训练集 R2: {metrics['train_r2']:.4f}")
        print(f"测试集 R2: {metrics['test_r2']:.4f}")
        print(f"交叉验证 R2: {metrics['cv_mean']:.4f} (±{metrics['cv_std']:.4f})")
        
        print(f"\n前5个重要特征:")
        print(importance.head().to_string(index=False))
        
        # 保存模型
        model_name = f"rf_{target.lower().replace(' ', '_')}_{size_group.lower()}"
        joblib.dump(metrics['model'], f'{model_name}_model.joblib')
        joblib.dump(metrics['scaler'], f'{model_name}_scaler.joblib')
        
        # 存储结果
        results.append({
            'Target': target,
            'Group': size_group,
            'MSE': metrics['mse'],
            'R2_Train': metrics['train_r2'],
            'R2_Test': metrics['test_r2'],
            'R2_CV': metrics['cv_mean'],
            'R2_CV_Std': metrics['cv_std']
        })

# 显示汇总结果
print("\n最终模型评估结果:")
print("-" * 80)
results_df = pd.DataFrame(results)
pd.set_option('display.float_format', lambda x: '%.4f' % x)
print(results_df.to_string(index=False))

print("\n所有模型和预处理器已保存")