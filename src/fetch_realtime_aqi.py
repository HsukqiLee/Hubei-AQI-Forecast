import requests
import pandas as pd
import json
import os
import time

# 和风天气 (QWeather) API 接口开发模版
# 用户需要在 https://dev.qweather.com/ 注册并创建免费项目，获取 API Key (免费开发版Key即可)

API_KEY = "YOUR_QWEATHER_API_KEY"  # 请替换为您的和风天气 API Key

CITIES = [
    "武汉", "黄石", "十堰", "宜昌", "襄阳", "鄂州", "荆门", 
    "孝感", "荆州", "黄冈", "咸宁", "随州", "恩施", "仙桃", 
    "潜江", "天门", "神农架"
]

def lookup_city_id(city_name, api_key):
    """
    通过和风地理 GeoAPI 获取城市 ID
    """
    url = f"https://geoapi.qweather.com/v2/city/lookup?location={city_name}&range=cn&key={api_key}"
    try:
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            data = response.json()
            if data["code"] == "200" and len(data["location"]) > 0:
                # 返回查找到的第一个匹配项的ID和行政区划
                city_id = data["location"][0]["id"]
                adm2 = data["location"][0]["adm2"]
                return city_id
        print(f"查找 {city_name} 失败，状态码: {response.json().get('code')}")
    except Exception as e:
        print(f"查找 {city_name} 时发生错误: {e}")
    return None

def fetch_realtime_aqi(city_id, api_key):
    """
    获取指定城市ID的实时空气质量 (Now AQI)
    """
    url = f"https://devapi.qweather.com/v7/air/now?location={city_id}&key={api_key}"
    try:
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            data = response.json()
            if data["code"] == "200":
                return data["now"]
        print(f"获取 ID {city_id} 失败，状态码: {response.json().get('code')}")
    except Exception as e:
        print(f"获取 ID {city_id} 的 AQI 时发生错误: {e}")
    return None

def main():
    if API_KEY == "YOUR_QWEATHER_API_KEY":
        print("="*60)
        print("请注意：您需要先在脚本中配置您的和风天气 API Key。")
        print("获取方式：访问 https://dev.qweather.com/ 注册开发者账号，创建一个免费项目即可获得。")
        print("="*60)
        return
        
    print("开始从和风天气 API 获取湖北 17 地市实时空气质量数据...")
    
    results = []
    
    for city in CITIES:
        full_name = city + "市" if city not in ["恩施", "神农架"] else (city + "土家族苗族自治州" if city == "恩施" else "神农架林区")
        print(f"\n正在查询: {full_name} ...")
        
        # 1. 查询城市 ID
        city_id = lookup_city_id(city, API_KEY)
        if not city_id:
            print(f"无法获取 {city} 的城市ID，跳过。")
            continue
            
        print(f"获取到城市ID: {city_id}")
        
        # 2. 查询实时 AQI
        time.sleep(0.2) # 礼貌延迟，避免超频 (免费版有QPM限制)
        aqi_data = fetch_realtime_aqi(city_id, API_KEY)
        
        if aqi_data:
            results.append({
                "城市": full_name,
                "和风城市ID": city_id,
                "发布时间": aqi_data.get("pubTime"),
                "AQI": aqi_data.get("aqi"),
                "空气质量等级": aqi_data.get("category"),
                "主要污染物": aqi_data.get("primary"),
                "PM2.5": aqi_data.get("pm2p5"),
                "PM10": aqi_data.get("pm10"),
                "O3": aqi_data.get("o3"),
                "SO2": aqi_data.get("so2"),
                "NO2": aqi_data.get("no2"),
                "CO": aqi_data.get("co")
            })
            print(f"实时空气质量 -> AQI: {aqi_data.get('aqi')} ({aqi_data.get('category')}), PM2.5: {aqi_data.get('pm2p5')}")
            
    if results:
        df = pd.DataFrame(results)
        df.to_csv("realtime_aqi.csv", index=False, encoding="utf-8-sig")
        print("\n" + "="*50)
        print("成功获取所有可查询城市的实时空气质量！")
        print("结果已保存至 realtime_aqi.csv")
        print("="*50)
    else:
        print("未能获取到任何城市的空气数据。")

if __name__ == "__main__":
    main()
