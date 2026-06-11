import os
import pandas as pd

def preprocess_data():
    print("开始数据预处理...")
    cities = [
        "武汉市", "黄石市", "十堰市", "宜昌市", "襄阳市", "鄂州市", "荆门市", 
        "孝感市", "荆州市", "黄冈市", "咸宁市", "随州市", "恩施土家族苗族自治州", 
        "仙桃市", "潜江市", "天门市", "神农架林区"
    ]
    
    columns = ["city", "date", "PM2.5", "PM10", "O3", "SO2", "NO2", "CO", "AQI"]
    data = []
    
    for i in range(17):
        city_name = cities[i]
        path = "./data/历史日数据_{}.xlsx".format(city_name)
        if not os.path.exists(path):
            raise FileNotFoundError(f"缺失数据文件: {path}。请先运行 fetch_historical_data.py 获取真实数据。")
            
        print(f"正在读取 {city_name} 的数据...")
        city_data = pd.read_excel(path)
        
        # 反转索引使其按时间顺序排列 (从旧到新)
        city_data = city_data.reindex(index=city_data.index[::-1])
        
        # 将行追加到列表
        for index, row in city_data.iterrows():
            data.append([
                city_name, 
                row["日期"], 
                row["PM2.5"], 
                row["PM10"], 
                row["O3"], 
                row["SO2"], 
                row["NO2"], 
                row["CO"], 
                row["AQI"]
            ])
            
    # 保存数据框
    os.makedirs("./outputs", exist_ok=True)
    output_df = pd.DataFrame(data=data, columns=columns)
    
    # 融合气象数据
    weather_path = "outputs/WeatherData.csv"
    if os.path.exists(weather_path):
        print("正在融合气象数据...")
        weather_df = pd.read_csv(weather_path)
        output_df['date'] = pd.to_datetime(output_df['date']).dt.strftime('%Y-%m-%d')
        weather_df['date'] = pd.to_datetime(weather_df['date']).dt.strftime('%Y-%m-%d')
        output_df = pd.merge(output_df, weather_df, on=['city', 'date'], how='left')
        
        # 填充缺失的气象数据
        for col in ['temperature', 'precipitation', 'wind_speed']:
            output_df[col] = output_df.groupby('city')[col].ffill().fillna(0)
    else:
        print("警告: 未找到 outputs/WeatherData.csv，跳过气象数据融合。")

    output_df.to_csv("outputs/AirCondition.csv", index=False)
    print("合并后的数据已保存至 outputs/AirCondition.csv")

if __name__ == "__main__":
    preprocess_data()
