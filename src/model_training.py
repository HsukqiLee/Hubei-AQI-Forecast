import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
import pickle as pkl
import os
import joblib
import json

# 导入预处理函数
from src.feature_engineering import preprocess_features

# LSTM+Attention 模型定义
class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        # 使用线性层计算特征空间的注意力打分
        self.attn = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, lstm_out):
        # lstm_out shape: (batch_size, seq_len, hidden_size)
        attn_weights = torch.softmax(self.attn(lstm_out), dim=1) # (batch_size, seq_len, 1)
        # context = attn_weights * lstm_out -> sum over seq_len
        context = torch.sum(attn_weights * lstm_out, dim=1)      # (batch_size, hidden_size)
        return context, attn_weights

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        # 加深：支持多层 LSTM，如果层数>1则默认开启层间 dropout
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, 
                            dropout=0.2 if num_layers > 1 else 0)
        
        # 引入注意力机制
        self.attention = Attention(hidden_size)
        
        # 加宽：加入正则化和更厚实的全连接层
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size // 2, output_size)
        )
 
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        out, _ = self.lstm(x, (h0, c0))
        # out: (batch, seq_len, hidden_size)
        
        # 使用注意力层从整个 14 天的隐状态中提取最关键上下文
        context, _ = self.attention(out)
        
        out = self.dropout(context)
        out = self.fc(out)
        return out

def create_sequences(data, target, time_steps=30):
    X, y = [], []
    for i in range(len(data) - time_steps):
        X.append(data[i:i+time_steps])
        y.append(target[i+time_steps])
    return np.array(X), np.array(y)

def train_model():
    print("开始进行 LSTM 模型训练流程...")
    
    # 1. 获取特征
    data_scaled, X_selected, y, scaler, all_features = preprocess_features()
    
    # 2. 提取序列
    cities = [
        "武汉市", "黄石市", "十堰市", "宜昌市", "襄阳市", "鄂州市", "荆门市", 
        "孝感市", "荆州市", "黄冈市", "咸宁市", "随州市", "恩施土家族苗族自治州", 
        "仙桃市", "潜江市", "天门市", "神农架林区"
    ]
    
    X_selected = np.array(X_selected)
    y = np.array(y)
    time_steps = 14
    clip_size = len(X_selected) // len(cities)
    
    X_train_list, y_train_list = [], []
    X_val_list, y_val_list = [], []
    X_test_list = []
    city_train_labels, city_val_labels = [], []
    AQI = []
    
    original_data = pd.read_csv("outputs/AirCondition.csv")
    
    for i in range(len(X_selected) // clip_size):
        city_X = X_selected[i*clip_size:(i+1)*clip_size]
        city_y = y[i*clip_size:(i+1)*clip_size]
        
        # 创建长度为 14 的序列
        X_seq, y_seq = create_sequences(city_X, city_y, time_steps)
        
        # 按照城市内部的时间序列前 80% 作为训练，后 20% 作为验证
        city_train_size = int(0.8 * len(X_seq))
        
        X_train_list.append(X_seq[:city_train_size])
        y_train_list.append(y_seq[:city_train_size])
        city_train_labels.extend([cities[i]] * city_train_size)
        
        X_val_list.append(X_seq[city_train_size:])
        y_val_list.append(y_seq[city_train_size:])
        city_val_labels.extend([cities[i]] * (len(X_seq) - city_train_size))
        
        # 测试序列是该城市最后 time_steps 天的数据
        test_seq = np.array([city_X[-time_steps:]])
        X_test_list.append(test_seq)
        
        # 保存历史 AQI（来自 original_data）
        AQI.append(list(original_data["AQI"][i*clip_size:(i+1)*clip_size].values))
        
    X_train_tensor = torch.tensor(np.concatenate(X_train_list, axis=0), dtype=torch.float32)
    y_train_tensor = torch.tensor(np.concatenate(y_train_list, axis=0), dtype=torch.float32).view(-1, 1)
    
    X_val_tensor = torch.tensor(np.concatenate(X_val_list, axis=0), dtype=torch.float32)
    y_val_tensor = torch.tensor(np.concatenate(y_val_list, axis=0), dtype=torch.float32).view(-1, 1)
    
    X_npred = torch.tensor(np.concatenate(X_test_list, axis=0), dtype=torch.float32)
    
    city_train_labels = np.array(city_train_labels)
    city_val_labels = np.array(city_val_labels)
    
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    # 4. 初始化模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_size = len(all_features)
    # 架构加深加宽：2层LSTM，256个神经元
    model = LSTMModel(input_size=input_size, hidden_size=256, num_layers=2, output_size=1).to(device)
    criterion = nn.SmoothL1Loss() # Huber Loss 适用于包含异常尖峰的空气质量
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4) # 稍微增加L2正则防止大数据集深网络过拟合
    
    # 替换为基于余弦退火的学习率调度器 (取代 ReduceLROnPlateau)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-5)
    
    # 5. 训练循环
    num_epochs = 200
    epoch_losses = []
    best_val_loss = float('inf')
    patience = 20
    wait = 0
    best_state = None
    print(f"正在训练 LSTM 模型，共 {num_epochs} 个周期，启用早停机制 (patience={patience})...")
    for epoch in range(num_epochs):
        model.train()
        batch_losses = []
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            batch_losses.append(loss.item())
            
        epoch_loss = np.mean(batch_losses)
        epoch_losses.append(epoch_loss)

        # 验证损失
        model.eval()
        val_losses = []
        with torch.no_grad():
            for vX, vy in val_loader:
                vX, vy = vX.to(device), vy.to(device)
                vpred = model(vX)
                vloss = criterion(vpred, vy)
                val_losses.append(vloss.item())
        val_loss = np.mean(val_losses) if len(val_losses) > 0 else float('nan')
        
        scheduler.step()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            wait = 0
            best_state = model.state_dict()
            torch.save(best_state, "./models/lstm_model_best.pth")
        else:
            wait += 1

        if (epoch + 1) % 20 == 0 or epoch == 0:
            print(f'周期 [{epoch+1}/{num_epochs}], 训练损失: {epoch_loss:.6f}, 验证损失: {val_loss:.6f}')

        if wait >= patience:
            print(f"在周期 {epoch+1} 早停。最佳验证损失: {best_val_loss:.6f}")
            break
            
    # 保存最终模型（验证最好的）
    if best_state is not None:
        torch.save(best_state, "./models/lstm_model.pth")
        print("最佳模型已保存至 ./models/lstm_model.pth")
    else:
        torch.save(model.state_dict(), "./models/lstm_model.pth")
        print("模型已保存至 ./models/lstm_model.pth")
    
    # 6. 在验证集上评估
    model.eval()
    # 加载最佳模型进行评估
    model.load_state_dict(torch.load("./models/lstm_model.pth"))
    model.to('cpu')
    model.eval()
    with torch.no_grad():
        y_val_pred = model(X_val_tensor).numpy()
        
    mse = mean_squared_error(y_val_tensor.numpy(), y_val_pred)
    r2 = r2_score(y_val_tensor.numpy(), y_val_pred)
    print(f"\n验证集指标 -> 均方误差(归一化): {mse:.6f}, 决定系数(R2): {r2:.6f}")
    
    # 反向缩放以进行可视化
    # 反向 AQI = AQI * scale + mean
    aqi_scale = scaler.scale_[6]
    aqi_mean = scaler.mean_[6]
    
    def inverse_AQI(scaled_aqi):
        return scaled_aqi * aqi_scale + aqi_mean
        
    y_val_pred_orig = inverse_AQI(y_val_pred)
    y_val_tensor_orig = inverse_AQI(y_val_tensor.numpy())

    # 每个城市的验证指标
    try:
        import collections
        val_metrics_per_city = {}
        y_val_pred_flat = y_val_pred.flatten()
        y_val_true_flat = y_val_tensor.numpy().flatten()
        for c in np.unique(city_val_labels):
            mask = city_val_labels == c
            if np.sum(mask) == 0:
                continue
            mse_c = mean_squared_error(y_val_true_flat[mask], y_val_pred_flat[mask])
            r2_c = r2_score(y_val_true_flat[mask], y_val_pred_flat[mask]) if np.sum(mask) > 1 else float('nan')
            val_metrics_per_city[c] = {"mse": float(mse_c), "r2": float(r2_c), "samples": int(np.sum(mask))}
        joblib.dump(val_metrics_per_city, "./models/val_metrics_per_city.pkl")
    except Exception:
        val_metrics_per_city = {}
    
    # 在验证集上绘制实际值与预测值
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans', 'sans-serif']
    plt.rcParams['axes.unicode_minus'] = False
    
    plt.figure(figsize=(10, 6))
    plt.plot(y_val_tensor_orig, label='Actual AQI', alpha=0.8)
    plt.plot(y_val_pred_orig, label='Predicted AQI', alpha=0.8)
    plt.legend()
    plt.xlabel('Validation Samples')
    plt.ylabel('AQI')
    plt.title('Actual vs Predicted AQI (Validation Set)')
    plt.tight_layout()
    os.makedirs("./outputs", exist_ok=True)
    plt.savefig("outputs/Actual_vs_Predicted.png", dpi=300)
    plt.close()
    print("实际值与预测值对比图已保存为 outputs/Actual_vs_Predicted.png")
    
    # 7. 预测未来一天（对于 X_npred）
    with torch.no_grad():
        y_future_pred = model(X_npred).numpy()
    y_future_pred_orig = inverse_AQI(y_future_pred)
    
    future_predictions = [int(item[0]) for item in y_future_pred_orig]
    
    # 将预测保存为 pickle 文件
    pred_data = {
        "省市地区": cities,
        "大气质量预测": future_predictions,
        "历史大气质量": AQI
    }
    
    os.makedirs("./outputs", exist_ok=True)
    with open("outputs/AirPrediction.pkl", "wb") as f:
        pkl.dump(pred_data, f)
    # 也保存 JSON 用于可视化
    pred_json = {c: int(v) for c, v in zip(cities, future_predictions)}
    with open("outputs/AirPrediction.json", "w", encoding='utf-8') as jf:
        json.dump(pred_json, jf, ensure_ascii=False, indent=2)
    print("预测结果已保存至 AirPrediction.pkl 和 outputs/AirPrediction.json")
    
    # 返回指标用于生成报告
    metrics = {
        "mse": mse,
        "r2": r2,
        "final_loss": epoch_losses[-1],
        "losses": epoch_losses
    }
    joblib.dump(metrics, "./models/metrics.pkl")
    return metrics

if __name__ == "__main__":
    train_model()
