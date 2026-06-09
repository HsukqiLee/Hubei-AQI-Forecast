import pickle as pkl
import matplotlib.pyplot as plt
from pyecharts.charts import Map
from pyecharts import options as opts
import os
import requests
import urllib3
# 由于 2026 年系统日期问题而禁用 SSL 警告
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

def run_visualization():
    print("开始进行结果可视化...")
    if not os.path.exists("outputs/AirPrediction.pkl"):
        raise FileNotFoundError("未找到 outputs/AirPrediction.pkl。请先运行 model_training.py。")
        
    with open("outputs/AirPrediction.pkl", "rb") as f:
        data = pkl.load(f)
        
    cities = data["省市地区"]
    predictions = data["大气质量预测"]
    history = data["历史大气质量"]
    
    # 1. 所有 17 个城市的 Matplotlib 子图
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans', 'sans-serif']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 创建图形
    fig = plt.figure(figsize=(18, 14))
    
    def plot_city_subplot(city_history_and_pred, rows, cols, index, city_name):
        ax = fig.add_subplot(rows, cols, index)
        # 除最后一天（预测）外的所有历史天数
        hist = city_history_and_pred[:-1]
        pred = city_history_and_pred[-1]
        
        x_hist = list(range(len(hist)))
        x_pred = len(hist)
        
        ax.plot(x_hist, hist, label='History', color='blue', alpha=0.7)
        ax.scatter([x_pred], [pred], label='Predicted', color='red', s=40, zorder=5)
        
        ax.set_title(city_name, fontsize=12)
        if index == 1:
            ax.legend(loc='upper left', fontsize=8)
            
    # 我们将它们布置在 4x5 的网格中（共 20 个插槽，使用了 17 个）
    # 案例研究代码存在索引错误，导致索引越界错误（在 4x5 网格上的索引为 21）。
    # 我们通过在 4x5 网格中按顺序组织子图来修复此问题。
    for i, name in enumerate(cities):
        # 合并历史和预测
        full_series = list(history[i]) + [predictions[i]]
        plot_city_subplot(full_series, 4, 5, i + 1, name)
        
    plt.suptitle("Hubei 17 Cities AQI History and Future 1-Day Prediction", fontsize=16, y=0.98)
    plt.tight_layout()
    os.makedirs("./outputs", exist_ok=True)
    plt.savefig("outputs/AirPrediction.png", dpi=300)
    plt.close()
    print("预测子图已保存为 outputs/AirPrediction.png")
    
    # 2. Pyecharts 湖北 AQI 热力图
    # 地图坐标与 pyecharts 湖北地图中的城市名称匹配
    # 注意：pyecharts 地图值必须为数字（int 或 float）
    data_pair = [list(z) for z in zip(cities, predictions)]
    
    c = Map(init_opts=opts.InitOpts(width="100%", height="100%", page_title="湖北省空气质量预测热力图"))
    c.add(
        series_name="湖北省各市州空气质量预测(AQI)",
        data_pair=data_pair,
        maptype="湖北",
        label_opts=opts.LabelOpts(is_show=True),
        is_map_symbol_show=True,
    )
    
    # 配置可视化地图选项
    c.set_global_opts(
        title_opts=opts.TitleOpts(title="湖北省空气质量指数预测分布图", subtitle="基于LSTM模型预测未来一日AQI值"),
        visualmap_opts=opts.VisualMapOpts(
            max_=150, 
            min_=20,
            is_piecewise=True,
            pieces=[
                {"min": 150, "label": "重度污染 (>150)", "color": "#7E0023"},
                {"min": 100, "max": 150, "label": "轻度-中度污染 (100-150)", "color": "#FF7E00"},
                {"min": 75, "max": 100, "label": "良 (75-100)", "color": "#FFDD33"},
                {"min": 50, "max": 75, "label": "良-优 (50-75)", "color": "#F3E5AB"},
                {"max": 50, "label": "优 (<50)", "color": "#556B2F"}
            ]
        )
    )
    
    os.makedirs("./outputs", exist_ok=True)
    
    echarts_path = "outputs/echarts.min.js"
    hubei_path = "outputs/hubei.js"
    
    # 从 cdnjs 下载 echarts.min.js
    if not os.path.exists(echarts_path):
        print("正在从 cdnjs 下载 echarts.min.js...")
        try:
            r = requests.get("https://cdnjs.cloudflare.com/ajax/libs/echarts/5.4.3/echarts.min.js", verify=False, timeout=15)
            if r.status_code == 200:
                with open(echarts_path, "wb") as f:
                    f.write(r.content)
                print("成功下载 echarts.min.js。")
        except Exception as e:
            print(f"下载 echarts.min.js 失败: {e}")
            
    # 从 jsdelivr/pyecharts 下载 hubei.js
    if not os.path.exists(hubei_path):
        print("正在从 jsdelivr/pyecharts 下载 hubei.js...")
        try:
            r = requests.get("https://cdn.jsdelivr.net/gh/pyecharts/pyecharts-assets@master/assets/maps/hubei.js", verify=False, timeout=15)
            if r.status_code == 200:
                with open(hubei_path, "wb") as f:
                    f.write(r.content)
                print("成功下载 hubei.js。")
        except Exception as e:
            print(f"下载 hubei.js 失败: {e}")
            
    c.render("outputs/map_hubei.html")
    
    # 将生成的 HTML 中的 CDN 引用替换为本地相对路径，并设置 html/body 样式以实现 100% 尺寸且不显示滚动条
    if os.path.exists("outputs/map_hubei.html"):
        with open("outputs/map_hubei.html", "r", encoding="utf-8") as f:
            html_content = f.read()
            
        import re
        html_content = re.sub(r'https?://assets\.pyecharts\.org/assets/(?:v\d+/)?echarts\.min\.js', './echarts.min.js', html_content)
        html_content = re.sub(r'https?://assets\.pyecharts\.org/assets/(?:v\d+/)?maps/hubei\.js', './hubei.js', html_content)
        
        # 注入 CSS 使 body/html 高度为 100% 并移除边距/滚动条
        if "<head>" in html_content:
            html_content = html_content.replace(
                "<head>",
                "<head><style>html, body { width: 100%; height: 100%; margin: 0; padding: 0; overflow: hidden; }</style>"
            )
        
        with open("outputs/map_hubei.html", "w", encoding="utf-8") as f:
            f.write(html_content)
        print("成功优化了 map_hubei.html 的尺寸和路径。")

if __name__ == "__main__":
    run_visualization()
