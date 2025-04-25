import os
import torch
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import models, transforms
from PIL import Image
import torch.optim as optim
import random
import numpy as np
from sklearn.metrics import accuracy_score
def set_seed(seed=42):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# ========= Dataset =========
class MPPIconDataset(Dataset):
    def __init__(self, csv_file):
        df = pd.read_csv(csv_file)
        self.image_paths = df['image_name'].tolist()
        #prefix = 'data_regression_random10_all/'
        prefix = 'test_regression3/'
        self.image_paths = [prefix + name for name in self.image_paths]
        self.mpp_labels = df['label'].tolist()
        self.transform = transforms.Compose([
            #transforms.Resize((, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std =[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        img = self.transform(img)
        label = torch.tensor(self.mpp_labels[idx], dtype=torch.float)
        return img, label

# ========= Model =========
class MPPRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = models.resnet50(pretrained=False)
        self.backbone.fc = nn.Identity()
        self.down_proj = nn.Linear(2048, 512)
        self.regression_head = nn.Linear(512, 1)

    def forward(self, x):
        features = self.backbone(x)
        features = self.down_proj(features)
        return self.regression_head(features).squeeze(1)

# ========= Training Setup =========
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
#label_candidates=[0.24,0.29,0.33,0.4,0.49,0.67,0.98,1.96,3.92]
label_candidates=[0.25,0.5,1]
def map_to_nearest_label(value):
    return label_candidates[np.argmin(np.abs(label_candidates - value))]

def map_to_nearest_label_index(value):
    return int(np.argmin(np.abs(label_candidates - value)))
def evaluate_on_test_set(csv_path, model_path, batch_size=256, seed=42):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(seed)

    dataset = MPPIconDataset(csv_path)
    total_len = len(dataset)
    train_len = int(0.8 * total_len)
    val_len = int(0.1 * total_len)
    test_len = total_len - train_len - val_len

    # 这里演示直接用整个 dataset 来构建 test_loader
    # 如果你只想测 test_ds, 那就用 test_ds: 
    # _, _, test_ds = random_split(dataset, [train_len, val_len, test_len], ...)
    # test_loader = DataLoader(test_ds, ...)
    test_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=24,
        pin_memory=True,
        shuffle=False  # 确保顺序
    )

    model = MPPRegressionModel().to(device)
    state_dict = torch.load(model_path, map_location=torch.device('cpu'))
    new_state_dict = {}
    for k, v in state_dict.items():
        name = k[7:] if k.startswith('module.') else k
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    model.eval()

    all_preds = []
    all_labels = []
    incorrect_list = []  # 存放 (文件名, 真实标签, 预测标签)

    start_index = 0
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)

            batch_preds = outputs.cpu().numpy()
            batch_labels = labels.cpu().numpy()

            # 转成离散类别
            batch_pred_classes = [map_to_nearest_label_index(p) for p in batch_preds]
            batch_label_classes = [map_to_nearest_label_index(l) for l in batch_labels]

            # 对比每个样本是否分类错误
            batch_size = len(imgs)
            for i in range(batch_size):
                if batch_pred_classes[i] != batch_label_classes[i]:
                    # 找到该样本的全局索引
                    global_index = start_index + i
                    # 拿到对应的文件名
                    img_name = dataset.image_paths[global_index]
                    true_label = batch_labels[i]
                    pred_label = batch_preds[i]
                    incorrect_list.append((img_name, true_label, pred_label))

            start_index += batch_size
            all_preds.extend(batch_preds)
            all_labels.extend(batch_labels)

    # 打印或保存评价指标
    mae = mean_absolute_error(all_labels, all_preds)
    mse = mean_squared_error(all_labels, all_preds)
    rmse = np.sqrt(mse)
    r2 = r2_score(all_labels, all_preds)

    pred_classes = np.array([map_to_nearest_label_index(p) for p in all_preds])
    label_classes = np.array([map_to_nearest_label_index(l) for l in all_labels])
    acc = accuracy_score(label_classes, pred_classes)

    print("=== Test Set Evaluation ===")
    print(f"MAE  : {mae:.4f}")
    print(f"MSE  : {mse:.4f}")
    print(f"RMSE : {rmse:.4f}")
    print(f"R²   : {r2:.4f}")
    print(f"Cls Accuracy (rounded) : {acc:.4f}")

    # 最后输出错误分类的列表
    print("===== 以下是所有分类错误的图像信息 =====")
    for (fname, tlabel, plabel) in incorrect_list:
        print(f"文件名: {fname}, 真实MPP: {tlabel}, 预测MPP: {plabel}")

# ========= Run =========
#csv_path = "regression_file.csv"
csv_path = "camelyone.csv"
evaluate_on_test_set(
    csv_path=csv_path,
    model_path='mpp_regression_output/model_epoch100.pth'  # 替换为你的模型路径
)

