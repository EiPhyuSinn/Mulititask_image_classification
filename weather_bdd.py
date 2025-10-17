import os
import sys
import json
import argparse
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
from PIL import Image
from torch.utils.tensorboard import SummaryWriter

# ----------------------------
# --- Custom Dataset ---
# ----------------------------
class CustomImageDataset(Dataset):
    WEATHER_LABELS = ['clear', 'foggy', 'overcast', 'partly cloudy', 'rainy', 'snowy', 'undefined']
    SCENE_LABELS   = ['city street', 'gas stations', 'highway', 'parking lot', 'residential', 'tunnel', 'undefined']
    TIME_LABELS    = ['dawn/dusk', 'daytime', 'night', 'undefined']

    def __init__(self, annotation_file, img_dir, transform=None):
        with open(annotation_file, "r") as f:
            self.data = json.load(f)

        existing_imgs = set(os.listdir(img_dir))
        self.data = [d for d in self.data if d["name"] in existing_imgs]

        self.img_dir = img_dir
        self.transform = transform

        self.weather2idx = {label: i for i, label in enumerate(self.WEATHER_LABELS)}
        self.scene2idx   = {label: i for i, label in enumerate(self.SCENE_LABELS)}
        self.time2idx    = {label: i for i, label in enumerate(self.TIME_LABELS)}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        ann = self.data[idx]
        img_path = os.path.join(self.img_dir, ann['name'])
        img = Image.open(img_path).convert("L")
        if self.transform:
            img = self.transform(img)

        weather_label = self.weather2idx.get(ann["attributes"]["weather"], self.weather2idx['undefined'])
        scene_label   = self.scene2idx.get(ann["attributes"]["scene"], self.scene2idx['undefined'])
        time_label    = self.time2idx.get(ann["attributes"]["timeofday"], self.time2idx['undefined'])

        return img, weather_label, scene_label, time_label

# ----------------------------
# --- Cross-Stitch Layer ---
# ----------------------------
class CrossStitch(nn.Module):
    def __init__(self, dim1, dim2):
        super().__init__()
        self.stitch = nn.Parameter(torch.eye(dim1 + dim2))

    def forward(self, x1, x2):
        batch_size = x1.size(0)
        x1_flat = x1.view(batch_size, -1)
        x2_flat = x2.view(batch_size, -1)
        x_cat = torch.cat([x1_flat, x2_flat], dim=1)
        out = x_cat @ self.stitch
        x1_out = out[:, :x1_flat.shape[1]].view_as(x1)
        x2_out = out[:, x1_flat.shape[1]:].view_as(x2)
        return x1_out, x2_out

# ----------------------------
# --- Multi-task CNN with Cross-Stitch ---
# ----------------------------
class MultiTaskCNN(nn.Module):
    def __init__(self, keep_prob=0.8, cross_stitch=True):
        super().__init__()
        self.keep_prob = keep_prob
        self.cross_stitch = cross_stitch

        # Conv layers
        self.conv1_1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv1_2 = nn.Conv2d(1, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2,2)
        self.conv2_1 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv2_2 = nn.Conv2d(32, 64, 3, padding=1)

        # Cross-stitch layers
        self.cs1 = CrossStitch(14*14*32, 14*14*32) if cross_stitch else None
        self.cs2 = CrossStitch(7*7*64, 7*7*64) if cross_stitch else None
        self.cs_fc = CrossStitch(512, 512) if cross_stitch else None

        # Fully connected
        self.fc1 = nn.Linear(7*7*64, 512)
        self.fc2 = nn.Linear(7*7*64, 512)

        # Output heads
        self.weather_head = nn.Linear(512, len(CustomImageDataset.WEATHER_LABELS))
        self.scene_head   = nn.Linear(512, len(CustomImageDataset.SCENE_LABELS))
        self.time_head    = nn.Linear(512, len(CustomImageDataset.TIME_LABELS))

        # BatchNorm
        self.bn1_1 = nn.BatchNorm2d(32)
        self.bn1_2 = nn.BatchNorm2d(32)
        self.bn2_1 = nn.BatchNorm2d(64)
        self.bn2_2 = nn.BatchNorm2d(64)

    def forward(self, x):
        # Conv1
        x1 = F.relu(self.bn1_1(self.conv1_1(x)))
        x2 = F.relu(self.bn1_2(self.conv1_2(x)))
        x1 = self.pool(x1)
        x2 = self.pool(x2)

        if self.cross_stitch:
            x1, x2 = self.cs1(x1, x2)

        # Conv2
        x1 = F.relu(self.bn2_1(self.conv2_1(x1)))
        x2 = F.relu(self.bn2_2(self.conv2_2(x2)))
        x1 = self.pool(x1)
        x2 = self.pool(x2)

        if self.cross_stitch:
            x1, x2 = self.cs2(x1, x2)

        # Flatten and FC
        x1 = x1.view(x1.size(0), -1)
        x2 = x2.view(x2.size(0), -1)
        x1 = F.relu(self.fc1(x1))
        x2 = F.relu(self.fc2(x2))

        if self.cross_stitch:
            x1, x2 = self.cs_fc(x1, x2)

        x1 = F.dropout(x1, p=1-self.keep_prob, training=self.training)
        x2 = F.dropout(x2, p=1-self.keep_prob, training=self.training)

        # Outputs
        return self.weather_head(x1), self.scene_head(x2), self.time_head(x2)

# ----------------------------
# --- Transform ---
# ----------------------------
transform = transforms.Compose([
    transforms.Resize((28,28)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# ----------------------------
# --- Training function ---
# ----------------------------
def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    full_dataset = CustomImageDataset(args.annotations, args.img_dir, transform=transform)
    n_total = len(full_dataset)
    n_train = int(n_total * 0.8)
    n_test = n_total - n_train
    train_dataset, test_dataset = random_split(full_dataset, [n_train, n_test])

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader  = DataLoader(test_dataset, batch_size=args.batch_size)

    model = MultiTaskCNN(keep_prob=args.keep_prob, cross_stitch=True).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.reg_lambda)
    criterion = nn.CrossEntropyLoss()

    writer = SummaryWriter(f"./logs/{datetime.now().timestamp()}")
    save_dir = "./model"
    os.makedirs(save_dir, exist_ok=True)

    for epoch in range(args.epochs):
        model.train()
        for imgs, weather, scene, time in train_loader:
            imgs, weather, scene, time = imgs.to(device), weather.to(device), scene.to(device), time.to(device)
            optimizer.zero_grad()
            out_w, out_s, out_t = model(imgs)
            loss = criterion(out_w, weather) + criterion(out_s, scene) + criterion(out_t, time)
            loss.backward()
            optimizer.step()

        # Evaluation
        model.eval()
        with torch.no_grad():
            total, correct_w, correct_s, correct_t = 0, 0, 0, 0
            for imgs, weather, scene, time in test_loader:
                imgs, weather, scene, time = imgs.to(device), weather.to(device), scene.to(device), time.to(device)
                out_w, out_s, out_t = model(imgs)
                pred_w = out_w.argmax(dim=1)
                pred_s = out_s.argmax(dim=1)
                pred_t = out_t.argmax(dim=1)
                total += imgs.size(0)
                correct_w += (pred_w == weather).sum().item()
                correct_s += (pred_s == scene).sum().item()
                correct_t += (pred_t == time).sum().item()
            print(f"Epoch {epoch}: Weather {correct_w/total:.3f}, Scene {correct_s/total:.3f}, Time {correct_t/total:.3f}")

        torch.save(model.state_dict(), os.path.join(save_dir, "final_model.pt"))

# ----------------------------
# --- Argument parser ---
# ----------------------------
def parse_args(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", type=str, default="prepare_dataset/bdd10k.json")
    parser.add_argument("--img_dir", type=str, default="/home/gwm-279/Downloads/10k_images_train/bdd100k/images/10k/train")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--reg_lambda", type=float, default=1e-5)
    parser.add_argument("--keep_prob", type=float, default=0.8)
    return parser.parse_args(argv)

# ----------------------------
# --- Main ---
# ----------------------------
if __name__ == "__main__":
    args = parse_args(sys.argv[1:])
    train(args)
