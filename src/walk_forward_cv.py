import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_squared_error, r2_score
from src.feature_engineering import preprocess_features
from src.model_training import LSTMModel, create_sequences
import joblib


def run_walk_forward(time_steps=14, initial_train_ratio=0.5, step_ratio=0.1, val_ratio=0.1, max_epochs=50):
    os.makedirs("./models", exist_ok=True)
    print("正在准备滚动向前交叉验证(walk-forward CV)数据...")
    data_scaled, X_selected, y, scaler, all_features = preprocess_features()

    # 使用与数据管道中相同的城市顺序重构每个城市的片段
    raw = pd.read_csv("outputs/AirCondition.csv")
    cities = list(raw['city'].unique())
    clip_size = len(raw) // len(cities)

    X_selected = np.array(X_selected)
    y = np.array(y)

    # 为每个城市构建序列并收集标签
    X_all, y_all, city_labels = [], [], []
    for i in range(len(X_selected) // clip_size):
        city_X = X_selected[i*clip_size:(i+1)*clip_size]
        city_y = y[i*clip_size:(i+1)*clip_size]
        X_seq, y_seq = create_sequences(city_X, city_y, time_steps)
        if len(X_seq) == 0:
            continue
        X_all.append(X_seq)
        y_all.append(y_seq)
        city_labels.extend([cities[i]] * len(X_seq))

    if len(X_all) == 0:
        raise RuntimeError("未创建任何序列。请检查 time_steps 或数据长度。")

    X_list = X_all
    y_list = y_all
    
    city_n = len(X_list[0])
    print(f"单城市序列数: {city_n}, 城市总数: {len(X_list)}")

    # 定义折叠大小 (在城市内部的时间轴上)
    initial_train = int(initial_train_ratio * city_n)
    step = max(1, int(step_ratio * city_n))
    val_size = max(1, int(val_ratio * city_n))

    folds = []
    train_end = initial_train
    while train_end + val_size <= city_n:
        val_start = train_end
        val_end = min(city_n, val_start + val_size)
        folds.append((0, train_end, val_start, val_end))
        train_end = min(city_n, train_end + step)
        if train_end == city_n:
            break

    print(f"正在运行 {len(folds)} 折滚动向前交叉验证")

    device = torch.device('cpu')

    fold_metrics = []
    per_city_records = []

    for fi, (t0, t1, v0, v1) in enumerate(folds, 1):
        print(f"折叠 {fi}: 每个城市 训练 [0:{t1}] 验证 [{v0}:{v1}]")
        
        fold_X_train, fold_y_train = [], []
        fold_X_val, fold_y_val = [], []
        fold_city_val_labels = []
        
        for i, (X_seq, y_seq) in enumerate(zip(X_list, y_list)):
            fold_X_train.append(X_seq[t0:t1])
            fold_y_train.append(y_seq[t0:t1])
            fold_X_val.append(X_seq[v0:v1])
            fold_y_val.append(y_seq[v0:v1])
            fold_city_val_labels.extend([cities[i]] * (v1 - v0))
            
        X_train_fold = np.concatenate(fold_X_train, axis=0)
        y_train_fold = np.concatenate(fold_y_train, axis=0)
        X_val_fold = np.concatenate(fold_X_val, axis=0)
        y_val_fold = np.concatenate(fold_y_val, axis=0)
        city_labels_val = np.array(fold_city_val_labels)

        X_train = torch.tensor(X_train_fold, dtype=torch.float32).to(device)
        y_train = torch.tensor(y_train_fold, dtype=torch.float32).view(-1,1).to(device)
        X_val = torch.tensor(X_val_fold, dtype=torch.float32).to(device)
        y_val = torch.tensor(y_val_fold, dtype=torch.float32).view(-1,1).to(device)

        train_ds = TensorDataset(X_train, y_train)
        val_ds = TensorDataset(X_val, y_val)
        train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)

        input_size = X_train.shape[2]
        model = LSTMModel(input_size=input_size, hidden_size=128, num_layers=2, output_size=1)
        criterion = nn.SmoothL1Loss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=False)

        best_val = float('inf')
        best_state = None
        patience = 5
        wait = 0

        for epoch in range(max_epochs):
            model.train()
            for bx, by in train_loader:
                optimizer.zero_grad()
                out = model(bx)
                loss = criterion(out, by)
                loss.backward()
                optimizer.step()

            # 验证
            model.eval()
            vals = []
            with torch.no_grad():
                for vx, vy in val_loader:
                    vout = model(vx)
                    vals.append(criterion(vout, vy).item())
            val_loss = float(np.mean(vals)) if vals else float('nan')
            scheduler.step(val_loss)
            if val_loss < best_val:
                best_val = val_loss
                best_state = model.state_dict()
                wait = 0
            else:
                wait += 1
            if wait >= patience:
                break

        # 在验证集上评估
        model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            y_pred = model(X_val).numpy().flatten()
            y_true = y_val_fold.flatten()

        mse = mean_squared_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        fold_metrics.append({'fold': fi, 'train_end': t1, 'val_start': v0, 'val_end': v1, 'mse': float(mse), 'r2': float(r2)})

        # 每个城市
        for c in np.unique(city_labels_val):
            mask = (city_labels_val == c)
            if mask.sum() == 0:
                continue
            mse_c = mean_squared_error(y_true[mask], y_pred[mask])
            r2_c = r2_score(y_true[mask], y_pred[mask]) if mask.sum() > 1 else float('nan')
            per_city_records.append({'fold': fi, 'city': c, 'mse': float(mse_c), 'r2': float(r2_c), 'samples': int(mask.sum())})

    # 保存结果
    with open('./models/walk_forward_metrics.json', 'w', encoding='utf-8') as f:
        json.dump(fold_metrics, f, ensure_ascii=False, indent=2)
    df_city = pd.DataFrame(per_city_records)
    df_city.to_csv('./models/walk_forward_per_city.csv', index=False)
    print('滚动向前交叉验证完成。指标已保存至 ./models/walk_forward_metrics.json 和 ./models/walk_forward_per_city.csv')


if __name__ == '__main__':
    run_walk_forward()
