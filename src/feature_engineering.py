import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression
import joblib
import os


def preprocess_features():
    """加载原始数据，创建时间和滞后特征，然后仅在训练部分拟合缩放器和选择器。

    返回 (data_scaled, X_selected, y, scaler, selected_features)
    """
    print("开始进行特征工程（仅限训练集的缩放器/选择器）...")
    if not os.path.exists("outputs/AirCondition.csv"):
        raise FileNotFoundError("未找到 outputs/AirCondition.csv。请先运行 data_preprocessing.py。")

    data = pd.read_csv("outputs/AirCondition.csv")
    data['date'] = pd.to_datetime(data['date'])

    # 确保与原始管道中一致的确定性城市顺序
    cities = list(data['city'].unique())

    # 1. 添加时间特征（来自原始日期）
    data['month'] = data['date'].dt.month
    data['day'] = data['date'].dt.day
    data['day_of_week'] = data['date'].dt.dayofweek
    data['season'] = data['month'] % 12 // 3 + 1

    # 2. 我们将仅在训练行（每个城市时间轴的前 80%）上拟合缩放器
    numerical_cols = ['PM2.5', 'PM10', 'O3', 'SO2', 'NO2', 'CO', 'AQI', 'temperature', 'precipitation', 'wind_speed']

    train_idx = []
    for c in cities:
        idx = data.index[data['city'] == c].tolist()
        if len(idx) == 0:
            continue
        cutoff = int(0.8 * len(idx))
        train_idx.extend(idx[:cutoff])

    # 仅在训练行上拟合缩放器
    os.makedirs("./models", exist_ok=True)
    scaler = StandardScaler()
    scaler.fit(data.loc[train_idx, numerical_cols])
    joblib.dump(scaler, "./models/scaler.pkl")
    print("StandardScaler 已在训练集上拟合，并保存至 ./models/scaler.pkl")

    # 将缩放器应用于整个数据集（使用拟合的缩放器）
    scaled_array = scaler.transform(data[numerical_cols])
    data_scaled = pd.DataFrame(scaled_array, columns=numerical_cols)
    data_scaled['city'] = data['city'].values
    data_scaled['date'] = data['date'].values

    # 重新附加时间特征
    data_scaled['month'] = data['month'].values
    data_scaled['day'] = data['day'].values
    data_scaled['day_of_week'] = data['day_of_week'].values
    data_scaled['season'] = data['season'].values

    # 3. 基于缩放后的 AQI 添加滞后特征（1-7天）（与原始方法一致）
    for lag in range(1, 8):
        data_scaled[f'AQI_lag_{lag}'] = data_scaled.groupby('city')['AQI'].shift(lag)

    # 增加移动平均特征以提取趋势
    data_scaled['AQI_rolling_3'] = data_scaled.groupby('city')['AQI'].transform(lambda x: x.shift(1).rolling(3).mean())
    data_scaled['AQI_rolling_7'] = data_scaled.groupby('city')['AQI'].transform(lambda x: x.shift(1).rolling(7).mean())

    data_scaled = data_scaled.fillna(0)

    # 4. 准备输入特征 X (保留 AQI 作为自回归的重要特征，仅丢弃无关的 city 和 date)
    X = data_scaled.drop(columns=['city', 'date'])
    y = data_scaled['AQI']

    all_features = list(X.columns)
    print("使用的特征 ({0}个):".format(len(all_features)), all_features)
    
    # 移除原有的 SelectKBest 逻辑，改为全量特征输出
    X_all = X.values
    return data_scaled, X_all, y.values, scaler, all_features


if __name__ == "__main__":
    preprocess_features()
