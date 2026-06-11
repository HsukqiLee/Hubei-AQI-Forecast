import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import urllib3
import requests
import time

# 禁用 SSL 警告
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# 从训练脚本导入模型定义和坐标
from src.model_training import LSTMModel, create_sequences
from src.fetch_historical_data import CITY_COORDINATES, get_china_aqi

# 2026年 1-6月的外推验证时间范围 (152天，从 2026-01-01 到 2026-06-01)
TEST_START = "2026-01-01"
TEST_END = "2026-06-01"

def fetch_test_data():
    """
    抓取 2026 年 4-5 月的真实湖北 17 地市空气数据并计算 AQI
    """
    delta_days = (pd.to_datetime(TEST_END) - pd.to_datetime(TEST_START)).days + 1
    print(f"正在通过 Open-Meteo API 抓取 2026 年 1-6 月真实空气质量测试数据...")
    print(f"测试数据时间跨度: {TEST_START} 至 {TEST_END} (共 {delta_days} 天)")
    print("="*60)
    
    columns = ["city", "date", "PM2.5", "PM10", "O3", "SO2", "NO2", "CO", "AQI"]
    data = []
    
    for city, coord in CITY_COORDINATES.items():
        print(f"正在获取 {city} 的 4-5 月数据...")
        url = (
            f"https://air-quality-api.open-meteo.com/v1/air-quality?"
            f"latitude={coord['lat']}&longitude={coord['lon']}&"
            f"start_date={TEST_START}&end_date={TEST_END}&"
            f"hourly=pm2_5,pm10,carbon_monoxide,nitrogen_dioxide,sulphur_dioxide,ozone&"
            f"timezone=auto"
        )
        
        try:
            r = requests.get(url, verify=False, timeout=15)
            if r.status_code == 200:
                hourly = r.json().get("hourly", {})
                if hourly and "time" in hourly:
                    df_h = pd.DataFrame({
                        "datetime": pd.to_datetime(hourly["time"]),
                        "PM2.5": hourly["pm2_5"],
                        "PM10": hourly["pm10"],
                        "CO": np.array(hourly["carbon_monoxide"]) / 1000.0,
                        "NO2": hourly["nitrogen_dioxide"],
                        "SO2": hourly["sulphur_dioxide"],
                        "O3": hourly["ozone"]
                    })
                    
                    df_h["date"] = df_h["datetime"].dt.strftime("%Y-%m-%d")
                    df_daily = df_h.groupby("date").mean().reset_index()
                    
                    # 按照时间正序排列
                    df_daily = df_daily.sort_values("date")
                    
                    for _, row in df_daily.iterrows():
                        pm25 = row["PM2.5"] if not pd.isna(row["PM2.5"]) else 15.0
                        pm10 = row["PM10"] if not pd.isna(row["PM10"]) else 25.0
                        so2 = row["SO2"] if not pd.isna(row["SO2"]) else 8.0
                        no2 = row["NO2"] if not pd.isna(row["NO2"]) else 15.0
                        co = row["CO"] if not pd.isna(row["CO"]) else 0.5
                        o3 = row["O3"] if not pd.isna(row["O3"]) else 40.0
                        
                        aqi = get_china_aqi(pm25, pm10, so2, no2, co, o3)
                        data.append([
                            city, row["date"], round(pm25, 1), round(pm10, 1), 
                            round(o3, 1), round(so2, 1), round(no2, 1), round(co, 2), aqi
                        ])
            time.sleep(0.3)
        except Exception as e:
            print(f"获取 {city} 测试数据失败: {e}")
            
    test_df = pd.DataFrame(data, columns=columns)
    os.makedirs("./outputs", exist_ok=True)
    test_df.to_csv("outputs/AirCondition_Test_2026.csv", index=False)
    print(f"\n测试集数据抓取完毕，共 {len(test_df)} 条记录，保存至 outputs/AirCondition_Test_2026.csv")

def evaluate_on_test_data():
    """
    加载模型并在 4-5 月的测试集上运行预测，评估外推性能
    """
    print("\n" + "="*60)
    print("开始使用已训练的 LSTM 模型在 2026年1-6月新时间跨度上做外推预测验证...")
    print("="*60)
    
    if not os.path.exists("outputs/AirCondition_Test_2026.csv"):
        fetch_test_data()
        
    test_data = pd.read_csv("outputs/AirCondition_Test_2026.csv")
    test_data['date'] = pd.to_datetime(test_data['date']).dt.strftime('%Y-%m-%d')
    
    # 融合气象数据
    weather_path = "outputs/WeatherData.csv"
    if os.path.exists(weather_path):
        weather_df = pd.read_csv(weather_path)
        weather_df['date'] = pd.to_datetime(weather_df['date']).dt.strftime('%Y-%m-%d')
        test_data = pd.merge(test_data, weather_df, on=['city', 'date'], how='left')
        for col in ['temperature', 'precipitation', 'wind_speed']:
            test_data[col] = test_data.groupby('city')[col].ffill().fillna(0)
            
    test_data['date'] = pd.to_datetime(test_data['date'])
    
    # 1. 加载模型及缩放器/特征参数
    if not os.path.exists("./models/lstm_model.pth") or not os.path.exists("./models/scaler.pkl"):
        raise FileNotFoundError("未找到已训练好的模型或参数文件。请先运行 run_pipeline.py 训练模型。")
        
    scaler = joblib.load("./models/scaler.pkl")
    
    # 2. 特征工程 (应用与训练集相同的归一化和特征处理)
    numerical_cols = ['PM2.5', 'PM10', 'O3', 'SO2', 'NO2', 'CO', 'AQI', 'temperature', 'precipitation', 'wind_speed']
    test_scaled_arr = scaler.transform(test_data[numerical_cols])
    test_scaled = pd.DataFrame(test_scaled_arr, columns=numerical_cols)
    test_scaled['city'] = test_data['city'].values
    test_scaled['date'] = test_data['date'].values
    
    # 周期特征
    test_scaled['month'] = test_scaled['date'].dt.month
    test_scaled['day'] = test_scaled['date'].dt.day
    test_scaled['day_of_week'] = test_scaled['date'].dt.dayofweek
    test_scaled['season'] = test_scaled['month'] % 12 // 3 + 1
    
    # 滞后特征
    for lag in range(1, 8):
        test_scaled[f'AQI_lag_{lag}'] = test_scaled.groupby('city')['AQI'].shift(lag)

    # 增加移动平均特征以提取趋势
    test_scaled['AQI_rolling_3'] = test_scaled.groupby('city')['AQI'].transform(lambda x: x.shift(1).rolling(3).mean())
    test_scaled['AQI_rolling_7'] = test_scaled.groupby('city')['AQI'].transform(lambda x: x.shift(1).rolling(7).mean())

    test_scaled = test_scaled.fillna(0)
    
    # 3. 准备特征 (丢弃无关的 city 和 date)
    X_test_all = test_scaled.drop(columns=['city', 'date'])
    y_test_all = test_scaled['AQI']
    
    # 4. 滑动窗口分割 (30天滑动窗口)
    X_selected_arr = np.array(X_test_all)
    y_arr = np.array(y_test_all)
    
    time_steps = 14
    clip_size = len(test_scaled) // len(CITY_COORDINATES)
    
    X_test_seq, y_test_seq = None, None
    
    for i in range(len(X_selected_arr) // clip_size):
        city_X = X_selected_arr[i*clip_size:(i+1)*clip_size]
        city_y = y_arr[i*clip_size:(i+1)*clip_size]
        
        X_seq, y_seq = create_sequences(city_X, city_y, time_steps)
        
        X_test_seq = X_seq if X_test_seq is None else np.concatenate([X_test_seq, X_seq], axis=0)
        y_test_seq = y_seq if y_test_seq is None else np.concatenate([y_test_seq, y_seq], axis=0)
        
    X_tensor = torch.tensor(X_test_seq, dtype=torch.float32)
    y_tensor = torch.tensor(y_test_seq, dtype=torch.float32).view(-1, 1)
    
    # 5. 加载模型结构与权重
    input_size = X_tensor.shape[2]
    model = LSTMModel(input_size, hidden_size=256, num_layers=2, output_size=1)
    model.load_state_dict(torch.load("./models/lstm_model.pth"))
    model.eval()
    
    # 6. 进行预测
    with torch.no_grad():
        y_pred = model(X_tensor).numpy()
        
    # 计算评估指标
    mse = mean_squared_error(y_tensor.numpy(), y_pred)
    r2 = r2_score(y_tensor.numpy(), y_pred)
    print(f"\n[独立测试集] 4-5月独立测试集外推评估结果:")
    print(f"   - 均方误差 (MSE - 归一化): {mse:.6f}")
    print(f"   - 决定系数 (R2): {r2:.6f}")
    
    # 逆标准化
    aqi_scale = scaler.scale_[6]
    aqi_mean = scaler.mean_[6]
    y_pred_orig = y_pred * aqi_scale + aqi_mean
    y_true_orig = y_tensor.numpy() * aqi_scale + aqi_mean
    
    # 7. 绘制真实值与预测值的对比并保存
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans', 'sans-serif']
    plt.rcParams['axes.unicode_minus'] = False
    
    plt.figure(figsize=(12, 6))
    plt.plot(y_true_orig[:300], label='2026年1-6月 真实AQI (前300个样本点)', color='blue', alpha=0.8)
    plt.plot(y_pred_orig[:300], label='LSTM 跨时间预测值', color='red', alpha=0.8, linestyle='--')
    plt.legend(loc='upper right')
    plt.xlabel('样本序列 (包含多地市合并排序)')
    plt.ylabel('AQI')
    plt.title('2026年 1-6月 湖北省空气质量真实值与 LSTM 跨时间外推预测对比图')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig("outputs/Test_2026_Comparison.png", dpi=300)
    plt.close()
    
    print("预测对比图已保存至 outputs/Test_2026_Comparison.png")
    
    # 将此结果保存至 metrics 中，以便随时读取
    test_metrics = {"mse": float(mse), "r2": float(r2)}
    joblib.dump(test_metrics, "./models/test_metrics.pkl")
    return test_metrics

if __name__ == "__main__":
    evaluate_on_test_data()
