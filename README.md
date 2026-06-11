# 湖北省空气质量预测分析 (Hubei Province Air Quality Prediction Analysis)

本仓库是[大学名称][学院名称]《数据结构与程序设计(Python语言)》课程项目的定制项目复现工程——**湖北省空气质量预测分析**。
本项目旨在通过收集湖北省内 17 个城市/林区的历史空气质量数据，运用特征工程与循环神经网络 (LSTM) 模型进行建模拟合，对未来一日的 AQI 进行预测，并对预测结果进行多维度的图形与交互式地图可视化，最终实现实验报告与小组分工表的自动化产出。

## 👥 小组成员与分工表

本项目由 4 名组员分工协作完成，并在代码量与职责上进行了量化分配。详细的贡献比例与具体职责如下：

| 姓名 | 角色 | 学号 | 负责模块 | 具体职责与工作描述 | 贡献度 |
| :--- | :--- | :--- | :--- | :--- | :---: |
| **[组长姓名]** | 组长 | [组长学号] | 机器学习建模 & 整合 | 负责项目总体架构与任务集成；实现 LSTM 模型搭建、滑动窗口划分及训练调优；编写整合流水线 `run_pipeline.py`；撰写报告中模型拟合与评估部分；负责课堂汇报技术答辩。 | **28%** |
| **[组员1姓名]** | 组员 | [组员1学号] | 数据采集与预处理 | 对接 Open-Meteo API 抓取湖北省 17 个地级市真实历史与测试空气成分数据；编写 `data_preprocessing.py` 进行时间对准与合并，输出 `outputs/AirCondition.csv`；负责 PPT 数据采集章节汇报。 | **24%** |
| **[组员2姓名]** | 组员 | [组员2学号] | 数据探索与特征工程 | 编写 `data_exploration.py` 统计地市均值、绘制总体走势图与相关性热力图；编写 `feature_engineering.py` 实现 Z-score 归一化、时间/滞后特征构造及 SelectKBest 特征筛选；负责 PPT 特征工程章节讲解。 | **24%** |
| **[组员3姓名]** | 组员 | [组员3学号] | 可视化与报告填写 | 编写 `visualization.py` 生成 17 城市历史/预测走势对比图与 pyecharts 交互地图网页 `outputs/map_hubei.html`（修复原案例索引溢出 Bug）；编写 `fill_report.py` 实现实验报告的规范化数据填充与图文整合编写；负责 PPT 成果展示讲解。 | **24%** |

---

## 📁 目录结构说明

为了保持项目根目录的整洁和专业度，我们对工程文件进行了分类归档：
```
q:\Python\AirQualityForecast\
├── data\                           # 存放17个地级市89天历史Excel数据
│   └── 历史日数据_{城市名}.xlsx
├── models\                         # 训练好的模型权重和参数数据
│   ├── lstm_model.pth
│   └── *.pkl
├── references\                     # 课程任务书、参考资料及实验报告空白模板 (只读)
│   ├── 2025-2026-2-《数据结构与程序设计(Python语言)》课程项目任务书_2.pdf
│   ├── 学生作品-天气数据爬取及可视化.pdf
│   ├── 实验案例4-湖北省空气质量预测分析-v1.1_2.pdf
│   └── 附2-综合应用实验报告.docx
├── reports\                        # 最终需要提交的小组产出报告文档
│   ├── 实验报告-湖北省空气质量预测分析-[姓名].docx    # 自动生成的 Word 实验报告
│   ├── 小组分工与贡献表.xlsx                        # 美化后的 Excel 分工表
│   └── 项目汇报发言稿.md                            # 课堂汇报发言词逐字稿
├── outputs\                        # 数据处理与预测过程生成的中间及图表成果
│   ├── AirCondition.csv            # 合并后的 17 城市总数据集
│   ├── AirPrediction.pkl           # 模型预测结果序列包
│   ├── AirTrends.png               # 17城市AQI走势图
│   ├── Correlation_Heatmap.png     # 7种污染物相关性热力图
│   ├── Actual_vs_Predicted.png     # LSTM在验证集上的拟合比对图
│   ├── AirPrediction.png           # 17城市未来一日AQI预测折线子图
│   └── map_hubei.html              # ECharts 交互式空气预测热力地图网页
├── src\                            # 模块化 Python 源代码目录
│   ├── fetch_real_2026_data.py     # 获取2026年17地市真实历史数据 (Open-Meteo API)

│   ├── data_preprocessing.py       # 数据合并与对齐
│   ├── data_exploration.py         # 探索性数据分析 (EDA) 与基本绘图
│   ├── feature_engineering.py       # 标准化与时序特征工程
│   ├── model_training.py           # LSTM 模型训练与验证
│   ├── visualization.py            # 折线图与 ECharts 地图生成
│   ├── generate_contribution_excel.py # 自动生成样式化 Excel 分工贡献表
│   └── fill_report.py              # Word 实验报告数据填充与格式整理
└── run_pipeline.py                 # 一键运行完整流水线的总入口脚本
```

---

## 🛠️ 项目技术架构

本项目依托 Python 生态，深度结合了数据科学与办公自动化的核心库：
- **数据处理**：`numpy` (2.4.4) + `pandas` (3.0.3)
- **特征工程**：`scikit-learn` (1.8.0) (包含 `StandardScaler` 和 `SelectKBest`)
- **深度学习**：`torch` (2.12.0) (搭建 2层 LSTM 循环神经网络)
- **数据可视化**：`matplotlib` + `seaborn` (折线图/热力图/预测对比图) + `pyecharts` (HTML 交互式湖北热力地图)
- **办公自动化**：`python-docx` + `openpyxl` (辅助填充 Word 报告与样式化生成 Excel 贡献表)

---

## 🚀 快速开始

### 1. 环境准备
确保您的计算机上已安装 Python 3.13+ 环境。克隆本项目后，在项目根目录下安装所有必要依赖包：
```bash
pip install pandas openpyxl matplotlib seaborn scikit-learn torch pyecharts python-docx joblib requests
```

### 2. 一键运行完整流水线
我们提供了一键式流水线执行脚本。只需在终端运行主入口程序：
```bash
python run_pipeline.py
```

该脚本将自动按顺序执行以下步骤：
1. **数据抓取与准备**：默认会调用 `src/fetch_real_2026_data.py` 通过 **Open-Meteo API** 免费抓取湖北 17 地市的**真实空气质量数据**，自动计算国标 AQI 后导出为 Excel，保证流水线正常运行。
2. **数据清洗与合并**：逆序并合并所有 Excel 文件为大表 `outputs/AirCondition.csv`。
3. **数据探索分析 (EDA)**：计算各城市 AQI 均值，生成 `outputs/AirTrends.png` 和 `outputs/Correlation_Heatmap.png`。
4. **特征工程与筛选**：进行归一化、构造时间周期特征与 1-7 天滞后特征，并通过 `SelectKBest` 筛选排名前 10 的核心特征。
5. **LSTM 神经网络训练**：基于 PyTorch 构建滑动窗口为 30 天的 LSTM 模型并进行 200 个 Epoch 的训练，计算测试集 MSE 和 $R^2$ 决定系数，输出 `outputs/Actual_vs_Predicted.png`。
6. **成果可视化**：生成 17 城市预测折线图 `outputs/AirPrediction.png`，渲染交互式湖北地图网页 `outputs/map_hubei.html`。
7. **分工表与实验报告生成**：在 `reports/` 目录下生成美化的 `小组分工与贡献表.xlsx`；辅助读取 `references/附2-综合应用实验报告.docx` 空白报告模版，提取当次运行的真实收敛参数、评价指标进行图文规范化写入，最终输出完整的 `reports/实验报告-湖北省空气质量预测分析-[姓名].docx`。

---

## 🌟 适当改进与 Bug 修复记录 (项目亮点)

根据课程任务书中“**鼓励对案例进行适当改进，将给予奖励**”的要求，我们小组在复现过程中完成了以下改进：

1. **🌿 对接真实 2026 年环保历史数据 (Open-Meteo API)**
   - **改进**：摒弃了单一的仿真假数据。我们在第一步接入了全球开源的 Open-Meteo 大气历史数据库 API，支持自动查取湖北省 17 地市精准坐标的历史空气浓度并结合国标公式计算真实 AQI。
2. **🐞 修复 Matplotlib 17城市子图索引溢出 Bug**
   - **原案例问题**：原代码在 4x5 的画布上绘制 17 城市的预测曲线时，因硬编码的 `plot_indices` 坐标映射存在超出 20 的范围（最大为 20+1=21），导致程序运行至最后几个城市时抛出 `IndexError: index 21 is out of bounds for axis 0 with size 20` 异常崩溃。
   - **修复方案**：我们在 `src/visualization.py` 中重构了循环逻辑，利用 4x5 布局进行安全的 1 到 17 顺序坐标定位，成功解决了画图崩溃问题。
3. **📄 实验报告辅助填充与规范排版 (`fill_report.py`)**
   - **改进**：利用 `python-docx` 库，在每次模型训练结束后自动提取模型当前的 Loss 值、验证集 MSE 和 $R^2$ 系数，填入空白报告模版，结合我们在实验中对数据和问题的详细分析文本，直接渲染输出一份格式规整的 `reports/实验报告-湖北省空气质量预测分析-[姓名].docx`，保证了实验报告的数据精度和排版标准。
4. **📊 贡献表美化生成 (`generate_contribution_excel.py`)**
   - **改进**：使用 `openpyxl` 办公自动化库对 `reports/小组分工与贡献表.xlsx` 进行了主题化设计。加入了深蓝色的专业报表表头、淡雅的斑马线交替背景行、自适应列宽、居中自动换行和单元格边框线，极大地提升了成果提交的视觉品质。
