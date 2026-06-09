import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def explore_data():
    print("开始进行数据探索...")
    if not os.path.exists("outputs/AirCondition.csv"):
        raise FileNotFoundError("未找到 outputs/AirCondition.csv。请先运行 data_preprocessing.py。")
        
    data = pd.read_csv("outputs/AirCondition.csv")
    
    # 1. 按城市分组并打印平均 AQI
    print("\n各城市平均 AQI:")
    city_mean = data.groupby('city')["AQI"].mean().sort_values()
    print(city_mean)
    
    # 配置 matplotlib 以在 Windows 上显示中文字符
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans', 'sans-serif']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 2. 绘制代表性城市 AQI 趋势及湖北省平均范围
    plt.figure(figsize=(14, 7))
    
    # 按日期排序以确保时间顺序
    # 按日期分组以获取所有城市的最小值、最大值和平均值
    daily_stats = data.groupby('date')['AQI'].agg(['min', 'max', 'mean']).reset_index()
    daily_stats['date_dt'] = pd.to_datetime(daily_stats['date'])
    daily_stats = daily_stats.sort_values('date_dt')
    
    data_sorted = data.copy()
    data_sorted['date_dt'] = pd.to_datetime(data_sorted['date'])
    data_sorted = data_sorted.sort_values('date_dt')
    
    # 绘制代表所有 17 个城市波动范围的阴影区域
    plt.fill_between(daily_stats['date'], daily_stats['min'], daily_stats['max'], 
                     color='#3b82f6', alpha=0.15, label='全省17地市波动范围')
    
    # 绘制全省平均值
    plt.plot(daily_stats['date'], daily_stats['mean'], 
             color='#64748b', linestyle='--', linewidth=1.5, label='全省日均值')
    
    # 绘制代表性城市
    rep_cities = {
        "武汉市": {"color": "#dc2626", "label": "武汉市 (省会/工业重镇)", "lw": 2.2},
        "宜昌市": {"color": "#2563eb", "label": "宜昌市 (鄂西中心城市)", "lw": 1.8},
        "神农架林区": {"color": "#16a34a", "label": "神农架林区 (生态对照基线)", "lw": 1.8}
    }
    
    for city, config in rep_cities.items():
        city_data = data_sorted[data_sorted['city'] == city]
        plt.plot(city_data['date'], city_data['AQI'], 
                 color=config["color"], label=config["label"], linewidth=config["lw"], alpha=0.9)
        
    dates = daily_stats['date'].values
    
    plt.legend(loc='upper right', frameon=True, facecolor='white', framealpha=0.9)
    plt.xlabel('日期')
    plt.ylabel('AQI 指数')
    plt.title(f'湖北省代表城市与全省 AQI 变化走向 ({len(dates)} 天)')
    
    # 对刻度进行采样以避免 X 轴上的日期重叠
    step = max(1, len(dates) // 10)
    plt.xticks(dates[::step], rotation=45)
    plt.tight_layout()
    os.makedirs("./outputs", exist_ok=True)
    plt.savefig("outputs/AirTrends.png", dpi=300)
    plt.close()
    print("已将优化后的 AQI 趋势图保存为 outputs/AirTrends.png")
    
    # 3. 相关性热力图
    plt.figure(figsize=(10, 8))
    pollutant_data = data[["PM2.5", "PM10", "O3", "SO2", "NO2", "CO", "AQI"]]
    sns.heatmap(pollutant_data.corr(), annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('污染物相关性热力图')
    plt.tight_layout()
    plt.savefig("outputs/Correlation_Heatmap.png", dpi=300)
    plt.close()
    print("已将相关性热力图保存为 outputs/Correlation_Heatmap.png")

if __name__ == "__main__":
    explore_data()
