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
            raise FileNotFoundError(f"缺失数据文件: {path}。请先运行 fetch_real_2026_data.py 获取真实数据。")
            
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
    output_df.to_csv("outputs/AirCondition.csv", index=False)
    print("合并后的数据已保存至 outputs/AirCondition.csv")

if __name__ == "__main__":
    preprocess_data()
