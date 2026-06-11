import sys
import os

# 如果需要，将 src 添加到 python 路径中
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))


from src.fetch_real_2026_data import fetch_and_save_data
from src.data_preprocessing import preprocess_data
from src.data_exploration import explore_data
from src.model_training import train_model
from src.visualization import run_visualization
from src.out_of_sample_test import evaluate_on_test_data
from src.generate_contribution_excel import generate_contribution_table
from src.fill_report import fill_report

def main():
    print("====================================================")
    # 1. 获取数据 (从 Open-Meteo 获取真实的湖北各市历史数据)
    print("\n--- 步骤 1：获取湖北各市 AQI 数据 ---")
    fetch_and_save_data()
    
    # 2. 预处理数据
    print("\n--- 步骤 2：合并并预处理 Excel 文件 ---")
    preprocess_data()
    
    # 3. 探索数据
    print("\n--- 步骤 3：进行探索性数据分析 ---")
    explore_data()
    
    # 4. 训练模型
    print("\n--- 步骤 4：训练 LSTM 模型并预测 ---")
    metrics = train_model()
    
    # 5. 可视化结果
    print("\n--- 步骤 5：生成可视化图表和交互式地图 ---")
    run_visualization()
    
    # 5.5. 样本外时间验证 (2026年4-5月)
    print("\n--- 步骤 5.5：运行 2026 年 4-5 月样本外验证 ---")
    try:
        evaluate_on_test_data()
    except Exception as e:
        print(f"运行样本外验证失败: {e}")
    
    # 6. 生成贡献表格
    print("\n--- 步骤 6：生成贡献度 Excel 文件 ---")
    generate_contribution_table()
    
    # 7. 填充 docx 报告
    print("\n--- 步骤 7：自动填充最终 Word 报告 ---")
    fill_report()
    
    print("\n====================================================")
    print("所有步骤均已成功完成！")
    print("生成的交付物：")
    print("1. [合并数据集] AirCondition.csv")
    print("2. [预测序列化数据] AirPrediction.pkl")
    print("3. [AQI 趋势图] AirTrends.png")
    print("4. [相关性热力图] Correlation_Heatmap.png")
    print("5. [模型预测与实际对比图] Actual_vs_Predicted.png")
    print("6. [17城市预测子图] AirPrediction.png")
    print("7. [交互式 Pyecharts 地图] map_hubei.html")
    print("8. [样本外测试数据] AirCondition_Test_2026.csv")
    print("9. [样本外对比图] Test_2026_Comparison.png")
    print("10. [小组贡献表] 小组分工与贡献表.xlsx")
    print("11. [完成的 Word 报告] 实验报告-湖北省空气质量预测分析-[姓名].docx")
    print("====================================================")

if __name__ == "__main__":
    main()
