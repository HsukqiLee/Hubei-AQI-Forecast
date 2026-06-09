# 湖北省空气质量预测分析 (Hubei Province Air Quality Prediction Analysis)

本仓库为空气质量预测分析项目。本项目旨在通过收集湖北省内 17 个城市/林区的历史空气质量数据，运用特征工程与循环神经网络 (LSTM) 模型进行建模拟合，对未来一日的 AQI 进行预测，并对预测结果进行多维度的图形与交互式地图可视化。

---

## 目录结构说明

为了保持项目根目录的整洁和专业度，我们对工程文件进行了分类归档：

```text
q:\Python\AirQualityForecast\
├── data\                           # 存放17个地级市历史Excel数据
│   └── 历史日数据_{城市名}.xlsx
├── models\                         # 训练好的模型权重和参数数据
│   ├── lstm_model.pth
│   └── *.pkl
├── outputs\                        # 数据处理与预测过程生成的中间及图表成果
│   ├── AirCondition.csv            # 合并后的 17 城市总数据集
│   ├── AirPrediction.pkl           # 模型预测结果序列包
│   ├── AirTrends.png               # 17城市AQI走势图
│   ├── Correlation_Heatmap.png     # 7种污染物相关性热力图
│   ├── Actual_vs_Predicted.png     # LSTM在验证集上的拟合比对图
│   ├── AirPrediction.png           # 17城市未来一日AQI预测折线子图
│   └── map_hubei.html              # ECharts 交互式空气预测热力地图网页
├── src\                            # 模块化 Python 源代码目录
│   ├── fetch_real_2026_data.py     # 获取17地市真实历史数据 (Open-Meteo API)
│   ├── fetch_realtime_aqi.py       # 获取实时空气质量
│   ├── data_preprocessing.py       # 数据合并与对齐
│   ├── data_exploration.py         # 探索性数据分析 (EDA) 与基本绘图
│   ├── feature_engineering.py      # 标准化与时序特征工程
│   ├── model_training.py           # LSTM 模型训练与验证
│   ├── out_of_sample_test.py       # 样本外时间验证
│   ├── walk_forward_cv.py          # 滚动向前交叉验证
│   └── visualization.py            # 折线图与 ECharts 地图生成
└── run_pipeline.py                 # 一键运行完整流水线的总入口脚本
```

---

## 项目技术架构

本项目依托 Python 生态，深度结合了数据科学的核心库：
- **数据处理**：numpy (2.4.4) + pandas (3.0.3)
- **特征工程**：scikit-learn (1.8.0) (包含 StandardScaler 和 SelectKBest)
- **深度学习**：torch (2.12.0) (搭建 2层 LSTM 循环神经网络)
- **数据可视化**：matplotlib + seaborn (折线图/热力图/预测对比图) + pyecharts (HTML 交互式湖北热力地图)

---

## 快速开始

### 1. 环境准备
确保您的计算机上已安装 Python 3.13+ 环境。克隆本项目后，在项目根目录下安装所有必要依赖包：
```bash
pip install pandas openpyxl matplotlib seaborn scikit-learn torch pyecharts joblib requests
```

### 2. 一键运行完整流水线
我们提供了一键式流水线执行脚本。只需在终端运行主入口程序：
```bash
python run_pipeline.py
```

该脚本将自动按顺序执行以下步骤：
1. **数据抓取与准备**：默认会调用 `src/fetch_real_2026_data.py` 通过 Open-Meteo API 免费抓取湖北 17 地市的真实空气质量数据，自动计算国标 AQI 后导出为 Excel，保证流水线正常运行。
2. **数据清洗与合并**：逆序并合并所有 Excel 文件为大表 `outputs/AirCondition.csv`。
3. **数据探索分析 (EDA)**：计算各城市 AQI 均值，生成 `outputs/AirTrends.png` 和 `outputs/Correlation_Heatmap.png`。
4. **特征工程与筛选**：进行归一化、构造时间周期特征与 1-7 天滞后特征，并通过 `SelectKBest` 筛选排名前 10 的核心特征。
5. **LSTM 神经网络训练**：基于 PyTorch 构建滑动窗口为 30 天的 LSTM 模型并进行多轮次 Epoch 的训练，计算测试集 MSE 和 R-squared 决定系数，输出 `outputs/Actual_vs_Predicted.png`。
6. **成果可视化**：生成 17 城市预测折线图 `outputs/AirPrediction.png`，渲染交互式湖北地图网页 `outputs/map_hubei.html`。
7. **样本外时间验证**：在全新测试集上执行外推预测验证。

---

## 项目亮点与优化

在研发过程中完成了以下核心优化：

1. **对接真实环保历史数据**
   我们在第一步接入了全球开源的 Open-Meteo 大气历史数据库 API，支持自动查取湖北省 17 地市精准坐标的历史空气浓度并结合国标公式计算真实 AQI。

2. **修复核心渲染溢出错误**
   修复了原基础代码在画布上绘制 17 城市的预测曲线时因硬编码坐标映射存在的越界风险。我们在 `src/visualization.py` 中重构了循环逻辑，利用稳定的布局进行安全的顺序坐标定位，彻底消除了画图崩溃的隐患。

3. **健壮的时序模型验证体系**
   项目加入了专门的样本外验证机制以及滚动向前交叉验证 (Walk-forward CV)，极大地增强了预测模型的可靠性与泛化能力，为实际业务场景中的应用打下坚实基础。
