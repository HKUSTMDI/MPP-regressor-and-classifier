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
def train_model(csv_path, num_epochs=100, batch_size=256, lr=1e-3, output_dir='mpp_regression_output',seed=42):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(seed)
    os.makedirs(output_dir, exist_ok=True)

    dataset = MPPIconDataset(csv_path)
    total_len = len(dataset)
    train_len = int(0.8 * total_len)
    val_len = int(0.1 * total_len)
    test_len = total_len - train_len - val_len

    train_ds, val_ds, test_ds = random_split(dataset, [train_len, val_len, test_len],
                                             generator=torch.Generator().manual_seed(seed))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,num_workers=24, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, num_workers=24, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size,num_workers=24, pin_memory=True)
    
    model = MPPRegressionModel().to(device)
    if torch.cuda.device_count() > 1:
        print("Using", torch.cuda.device_count(), "GPUs.")
        model = nn.DataParallel(model)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)

        # ===== Validation =====
        model.eval()
        val_loss = 0
        all_preds, all_labels = [], []
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                all_preds.extend(outputs.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch {epoch + 1}: Train Loss = {avg_train_loss:.4f} | Val Loss = {avg_val_loss:.4f}")

        # ===== 保存模型 =====
        torch.save(model.state_dict(), os.path.join(output_dir, f"model_epoch{epoch+1}.pth"))
        if epoch % 20 == 0:
            plt.figure(figsize=(6, 6))
            plt.scatter(all_labels, all_preds, alpha=0.6)
            plt.plot([min(all_labels), max(all_labels)], [min(all_labels), max(all_labels)], 'r--')
            plt.xlabel("True MPP")
            plt.ylabel("Predicted MPP")
            plt.title(f"Epoch {epoch + 1}: Val Loss {avg_val_loss:.4f}")
            plt.grid(True)
            plt.savefig(os.path.join(output_dir, f"fit_epoch{epoch + 1}.png"))
    model.eval()
    test_preds, test_labels = [], []
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            test_preds.extend(outputs.cpu().numpy())
            test_labels.extend(labels.cpu().numpy())
        plt.figure(figsize=(6, 6))
        plt.scatter(test_labels, test_preds, alpha=0.6)
        plt.plot([min(test_labels), max(test_labels)], [min(test_labels), max(test_labels)], 'r--')
        plt.xlabel("True MPP")
        plt.ylabel("Predicted MPP")
        plt.title(f"Epoch {epoch + 1}: Val Loss {avg_val_loss:.4f}")
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, 'test_fit_final.png'))
        # ===== 拟合图 =====
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
#label_candidates=[0.24,0.29,0.33,0.4,0.49,0.67,0.98,1.96,3.92]
label_candidates=[0.25,0.5,1]
def map_to_nearest_label(value):
    return label_candidates[np.argmin(np.abs(label_candidates - value))]

def map_to_nearest_label_index(value):
    return int(np.argmin(np.abs(label_candidates - value)))


    # 最后输出错误分类的列表
    print("===== 以下是所有分类错误的图像信息 =====")
    for (fname, tlabel, plabel) in incorrect_list:
        print(f"文件名: {fname}, 真实MPP: {tlabel}, 预测MPP: {plabel}")

# ========= Run =========
#csv_path = "regression_file.csv"
csv_path = "camelyon2.csv"

train_model(csv_path)

