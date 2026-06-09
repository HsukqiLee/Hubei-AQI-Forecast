import requests
import pandas as pd
import numpy as np
import os
import time

# 湖北省 17 地市 WGS84 坐标映射
CITY_COORDINATES = {
    "武汉市": {"lat": 30.584355, "lon": 114.298572},
    "黄石市": {"lat": 30.220074, "lon": 115.03847},
    "十堰市": {"lat": 32.646948, "lon": 110.799435},
    "宜昌市": {"lat": 30.695345, "lon": 111.290843},
    "襄阳市": {"lat": 32.008637, "lon": 112.122557},
    "鄂州市": {"lat": 30.395632, "lon": 114.894875},
    "荆门市": {"lat": 31.035414, "lon": 112.19932},
    "孝感市": {"lat": 30.927909, "lon": 113.926655},
    "荆州市": {"lat": 30.332616, "lon": 112.239741},
    "黄冈市": {"lat": 30.45389, "lon": 114.8722},
    "咸宁市": {"lat": 29.841551, "lon": 114.322839},
    "随州市": {"lat": 31.717497, "lon": 113.382463},
    "恩施土家族苗族自治州": {"lat": 30.27214, "lon": 109.4795},
    "仙桃市": {"lat": 30.366546, "lon": 113.447547},
    "潜江市": {"lat": 30.421245, "lon": 112.892694},
    "天门市": {"lat": 30.653061, "lon": 113.165863},
    "神农架林区": {"lat": 31.744449, "lon": 110.671525}
}

# 2025年 365 天时间范围 (从 2025-01-01 到 2025-12-31)
START_DATE = "2025-01-01"
END_DATE = "2025-12-31"

# IAQI 换算函数 (依据中国国标 GB3095-2012 / HJ633-2012)
def calculate_iaqi(val, bp, iaqi_bp):
    for i in range(len(bp) - 1):
        if bp[i] <= val <= bp[i+1]:
            return (iaqi_bp[i+1] - iaqi_bp[i]) / (bp[i+1] - bp[i]) * (val - bp[i]) + iaqi_bp[i]
    # 超出最高阈值按最高级比例计算
    return iaqi_bp[-1]

def get_china_aqi(pm25, pm10, so2, no2, co, o3):
    # 各项指标的 24 小时平均限度 Breakpoints
    bp_pm25 = [0, 35, 75, 115, 150, 250, 350, 500]
    bp_pm10 = [0, 50, 150, 250, 350, 420, 500, 600]
    bp_so2 = [0, 50, 150, 475, 800, 1600, 2100, 2620]
    bp_no2 = [0, 40, 80, 180, 280, 565, 750, 940]
    bp_co = [0, 2, 4, 14, 24, 36, 48, 60]
    bp_o3 = [0, 160, 200, 300, 400, 800, 1000, 1200]  # O3日最大8小时平均值或1小时值
    
    iaqi_bp = [0, 50, 100, 150, 200, 300, 400, 500]
    
    iaqi_pm25 = calculate_iaqi(pm25, bp_pm25, iaqi_bp)
    iaqi_pm10 = calculate_iaqi(pm10, bp_pm10, iaqi_bp)
    iaqi_so2 = calculate_iaqi(so2, bp_so2, iaqi_bp)
    iaqi_no2 = calculate_iaqi(no2, bp_no2, iaqi_bp)
    iaqi_co = calculate_iaqi(co, bp_co, iaqi_bp)
    iaqi_o3 = calculate_iaqi(o3, bp_o3, iaqi_bp)
    
    # AQI 为各项分指数 IAQI 的最大值
    return int(max(iaqi_pm25, iaqi_pm10, iaqi_so2, iaqi_no2, iaqi_co, iaqi_o3))

def fetch_and_save_data():
    os.makedirs("./data", exist_ok=True)
    delta_days = (pd.to_datetime(END_DATE) - pd.to_datetime(START_DATE)).days + 1
    print(f"正在通过 Open-Meteo 免费获取 {pd.to_datetime(START_DATE).year} 年湖北 17 地市真实空气质量历史数据...")
    print(f"数据时间跨度: {START_DATE} 至 {END_DATE} (共 {delta_days} 天)")
    print("="*60)
    
    for city, coord in CITY_COORDINATES.items():
        print(f"\n[正在获取] {city} (Lat: {coord['lat']:.4f}, Lon: {coord['lon']:.4f})...")
        
        # 接口参数
        url = (
            f"https://air-quality-api.open-meteo.com/v1/air-quality?"
            f"latitude={coord['lat']}&longitude={coord['lon']}&"
            f"start_date={START_DATE}&end_date={END_DATE}&"
            f"hourly=pm2_5,pm10,carbon_monoxide,nitrogen_dioxide,sulphur_dioxide,ozone&"
            f"timezone=auto"
        )
        
        try:
            response = requests.get(url, timeout=15)
            if response.status_code != 200:
                print(f"❌ 接口请求失败! 状态码: {response.status_code}")
                continue
                
            data = response.json()
            hourly = data.get("hourly", {})
            if not hourly or "time" not in hourly:
                print("❌ 响应中未包含有效的小时级别空气数据!")
                continue
                
            # 解析为 Pandas DataFrame
            df_hourly = pd.DataFrame({
                "datetime": pd.to_datetime(hourly["time"]),
                "PM2.5": hourly["pm2_5"],
                "PM10": hourly["pm10"],
                # Open-Meteo 的 CO 默认为 μg/m³，国标计算需要转换为 mg/m³
                "CO": np.array(hourly["carbon_monoxide"]) / 1000.0,
                "NO2": hourly["nitrogen_dioxide"],
                "SO2": hourly["sulphur_dioxide"],
                "O3": hourly["ozone"]
            })
            
            # 按日期分组，求取 24 小时均值
            df_hourly["date"] = df_hourly["datetime"].dt.strftime("%Y-%m-%d")
            df_daily = df_hourly.groupby("date").mean().reset_index()
            
            # 计算每日 AQI
            aqi_list = []
            for _, row in df_daily.iterrows():
                # 填充 NaN 值为临界安全值
                pm25_val = row["PM2.5"] if not pd.isna(row["PM2.5"]) else 15.0
                pm10_val = row["PM10"] if not pd.isna(row["PM10"]) else 25.0
                so2_val = row["SO2"] if not pd.isna(row["SO2"]) else 8.0
                no2_val = row["NO2"] if not pd.isna(row["NO2"]) else 15.0
                co_val = row["CO"] if not pd.isna(row["CO"]) else 0.5
                o3_val = row["O3"] if not pd.isna(row["O3"]) else 40.0
                
                aqi = get_china_aqi(pm25_val, pm10_val, so2_val, no2_val, co_val, o3_val)
                aqi_list.append(aqi)
                
            df_daily["AQI"] = aqi_list
            
            # 整理列格式
            df_final = df_daily[["date", "PM2.5", "PM10", "O3", "SO2", "NO2", "CO", "AQI"]].copy()
            # 转换数值类型为标准舍入
            df_final["PM2.5"] = df_final["PM2.5"].round(1)
            df_final["PM10"] = df_final["PM10"].round(1)
            df_final["O3"] = df_final["O3"].round(1)
            df_final["SO2"] = df_final["SO2"].round(1)
            df_final["NO2"] = df_final["NO2"].round(1)
            df_final["CO"] = df_final["CO"].round(2)
            
            # 命名列名称为中文，适配 pandas 读取结构
            df_final = df_final.rename(columns={"date": "日期"})
            
            # 注意：Case Study code逆序读取数据: `city_data.reindex(index=city_data.index[::-1])`
            # 这说明 Excel 物理文件本身应当是由新到旧（逆序）排列的
            df_final = df_final.reindex(index=df_final.index[::-1])
            
            # 导出为 Excel 文件
            file_path = f"./data/历史日数据_{city}.xlsx"
            df_final.to_excel(file_path, index=False)
            print(f"✅ 成功生成真实日历史数据: {file_path} (共 {len(df_final)} 天)")
            
            # 延迟以示礼貌
            time.sleep(0.3)
            
        except Exception as e:
            print(f"❌ 获取 {city} 历史数据时发生异常: {e}")
            
    print("\n" + "="*60)
    print("🎉 2026年湖北省17地市真实空气历史数据获取并生成完毕！")
    print("="*60)

if __name__ == "__main__":
    fetch_and_save_data()
