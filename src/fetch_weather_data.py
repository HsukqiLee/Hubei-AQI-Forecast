import requests
import pandas as pd
import time
import os

CITY_COORDINATES = {
    "武汉市": {"lat": 30.58, "lon": 114.28},
    "黄石市": {"lat": 30.23, "lon": 115.03},
    "十堰市": {"lat": 32.65, "lon": 110.78},
    "宜昌市": {"lat": 30.70, "lon": 111.28},
    "襄阳市": {"lat": 32.05, "lon": 112.13},
    "鄂州市": {"lat": 30.40, "lon": 114.88},
    "荆门市": {"lat": 31.03, "lon": 112.20},
    "孝感市": {"lat": 30.92, "lon": 113.92},
    "荆州市": {"lat": 30.33, "lon": 112.23},
    "黄冈市": {"lat": 30.45, "lon": 114.88},
    "咸宁市": {"lat": 29.83, "lon": 114.32},
    "随州市": {"lat": 31.70, "lon": 113.38},
    "恩施土家族苗族自治州": {"lat": 30.28, "lon": 109.48},
    "仙桃市": {"lat": 30.37, "lon": 113.45},
    "潜江市": {"lat": 30.42, "lon": 112.88},
    "天门市": {"lat": 30.65, "lon": 113.17},
    "神农架林区": {"lat": 31.75, "lon": 110.67}
}

START_DATE = "2023-01-01"
# Open-Meteo 历史接口要求日期必须在过去 (通常延迟5天)，系统当前时间为 2026-06-11
END_DATE = "2026-06-05"

def fetch_weather_data():
    print("开始从 Open-Meteo 获取历史气象数据...")
    os.makedirs("./outputs", exist_ok=True)
    all_weather_data = []

    for city, coord in CITY_COORDINATES.items():
        print(f"正在获取 {city} 的气象数据...")
        url = (
            f"https://archive-api.open-meteo.com/v1/archive?"
            f"latitude={coord['lat']}&longitude={coord['lon']}&"
            f"start_date={START_DATE}&end_date={END_DATE}&"
            f"daily=temperature_2m_mean,precipitation_sum,wind_speed_10m_max&"
            f"timezone=auto"
        )
        
        try:
            response = requests.get(url, timeout=15)
            if response.status_code != 200:
                print(f"[错误] {city} 接口请求失败! 状态码: {response.status_code}")
                continue
                
            data = response.json()
            daily = data.get("daily", {})
            if not daily or "time" not in daily:
                print(f"[错误] {city} 响应中未包含有效的日级别气象数据!")
                continue
                
            # 解析为 Pandas DataFrame
            df_daily = pd.DataFrame({
                "city": city,
                "date": daily["time"],
                "temperature": daily["temperature_2m_mean"],
                "precipitation": daily["precipitation_sum"],
                "wind_speed": daily["wind_speed_10m_max"]
            })
            
            all_weather_data.append(df_daily)
            time.sleep(0.5)  # 礼貌延迟
            
        except Exception as e:
            print(f"[错误] 获取 {city} 气象数据时发生异常: {e}")

    if all_weather_data:
        final_df = pd.concat(all_weather_data, ignore_index=True)
        final_df.to_csv("outputs/WeatherData.csv", index=False)
        print("\n[成功] 所有气象数据成功保存至 outputs/WeatherData.csv")
    else:
        print("\n[失败] 未能获取到任何气象数据。")

if __name__ == "__main__":
    fetch_weather_data()
