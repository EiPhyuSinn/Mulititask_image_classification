import os
import sys
import json
import argparse
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from torch.utils.tensorboard import SummaryWriter

# ----------------------------
# --- Custom Dataset ---
# ----------------------------
class CustomImageDataset(Dataset):
    def __init__(self, annotation_file, img_dir, transform=None):
        with open(annotation_file, "r") as f:
            self.data = json.load(f)["annotations"]
        self.img_dir = img_dir
        self.transform = transform

        # Collect all unique labels for both tasks
        self.period_labels = sorted(list({ann["period"] for ann in self.data}))
        self.weather_labels = sorted(list({ann["weather"] for ann in self.data}))
        self.period2idx = {label: i for i, label in enumerate(self.period_labels)}
        self.weather2idx = {label: i for i, label in enumerate(self.weather_labels)}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        ann = self.data[idx]
        # img_path = os.path.join(self.img_dir, ann["filename"])
        # img_path = ann['filename'].replace("\\", "/")

        img_path = ann['filename'].replace("\\", "/")
        img_path = os.path.join(self.img_dir, img_path)

        img = Image.open(img_path).convert("L")  # grayscale for CNN
        if self.transform:
            img = self.transform(img)
        # Multi-task labels
        period_label = self.period2idx[ann["period"]]
        weather_label = self.weather2idx[ann["weather"]]
        return img, period_label, weather_label

# ----------------------------
# --- Transform ---
# ----------------------------
transform = transforms.Compose([
    transforms.Resize((28,28)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# --- Cross-Stitch Layer ---
class CrossStitch(nn.Module):
    def __init__(self, dim1, dim2):
        super().__init__()
        self.dim1 = dim1
        self.dim2 = dim2
        # Cross-stitch matrix initialized to identity
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

# --- Multi-task CNN ---
class MultiTaskCNN(nn.Module):
    def __init__(self, cross_stitch_enabled=True, keep_prob=0.8):
        super().__init__()
        self.cross_stitch_enabled = cross_stitch_enabled
        self.keep_prob = keep_prob

        # Conv layers
        self.conv1_1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv1_2 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2,2)

        self.conv2_1 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv2_2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)

        # Cross-stitch layers (after pool1 and pool2)
        self.cs1 = None
        self.cs2 = None

        # Fully connected
        self.fc3_1 = nn.Linear(7*7*64, 1024)
        self.fc3_2 = nn.Linear(7*7*64, 1024)
        self.cs3 = None

        # Output layers
        self.output_1 = nn.Linear(1024, 10)
        self.output_2 = nn.Linear(1024, 3)

        if self.cross_stitch_enabled:
            # Compute flattened sizes dynamically
            self.cs1 = CrossStitch(14*14*32, 14*14*32)
            self.cs2 = CrossStitch(7*7*64, 7*7*64)
            self.cs3 = CrossStitch(1024, 1024)

        # BatchNorm layers
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

        # Cross-stitch after pool1
        if self.cross_stitch_enabled:
            x1, x2 = self.cs1(x1, x2)

        # Conv2
        x1 = F.relu(self.bn2_1(self.conv2_1(x1)))
        x2 = F.relu(self.bn2_2(self.conv2_2(x2)))
        x1 = self.pool(x1)
        x2 = self.pool(x2)

        # Cross-stitch after pool2
        if self.cross_stitch_enabled:
            x1, x2 = self.cs2(x1, x2)

        # Flatten and FC3
        x1 = x1.view(x1.size(0), -1)
        x2 = x2.view(x2.size(0), -1)
        x1 = F.relu(self.fc3_1(x1))
        x2 = F.relu(self.fc3_2(x2))

        # Cross-stitch after FC3
        if self.cross_stitch_enabled:
            x1, x2 = self.cs3(x1, x2)

        # Dropout
        x1 = F.dropout(x1, p=1-self.keep_prob, training=self.training)
        x2 = F.dropout(x2, p=1-self.keep_prob, training=self.training)

        # Output
        out1 = self.output_1(x1)
        out2 = self.output_2(x2)
        return out1, out2

# ----------------------------
# --- Training ---
# ----------------------------
def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ----- Load dataset -----
    train_dataset = CustomImageDataset(
        annotation_file=args.train_annotations,
        img_dir=args.train_img_dir,
        transform=transform
    )
    test_dataset = CustomImageDataset(
        annotation_file=args.test_annotations,
        img_dir=args.test_img_dir,
        transform=transform
    )

    train_loader = DataLoader(train_dataset, batch_size=args.n_batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=128)

    # ----- Model -----
    model = MultiTaskCNN(cross_stitch_enabled=args.cross_stitch_enabled, keep_prob=args.keep_prob).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.reg_lambda)
    criterion = nn.CrossEntropyLoss()

    # ----- TensorBoard -----
    log_dir = f"./tf_logs/custom_dataset_multi_task/{datetime.now().timestamp()}"
    writer = SummaryWriter(log_dir)
    save_dir = "./custom_dataset_multi_task_model"
    os.makedirs(save_dir, exist_ok=True)

    global_step = 0
    for epoch in range(args.n_epoch):
        model.train()
        for X_batch, y_batch1, y_batch2 in train_loader:
            X_batch = X_batch.to(device)
            y_batch1 = y_batch1.to(device)
            y_batch2 = y_batch2.to(device)

            optimizer.zero_grad()
            out1, out2 = model(X_batch)
            loss1 = criterion(out1, y_batch1)
            loss2 = criterion(out2, y_batch2)
            loss = loss1 + loss2
            loss.backward()
            optimizer.step()

            if global_step % 100 == 0:
                model.eval()
                with torch.no_grad():
                    # Train accuracy
                    pred1, pred2 = out1.argmax(dim=1), out2.argmax(dim=1)
                    acc_train1 = (pred1 == y_batch1).float().mean()
                    acc_train2 = (pred2 == y_batch2).float().mean()
                    acc_train = (acc_train1 + acc_train2)/2

                    # Test accuracy
                    all_accs = []
                    for X_test, y_test1, y_test2 in test_loader:
                        X_test = X_test.to(device)
                        y_test1 = y_test1.to(device)
                        y_test2 = y_test2.to(device)
                        out1_test, out2_test = model(X_test)
                        pred1_test = out1_test.argmax(dim=1)
                        pred2_test = out2_test.argmax(dim=1)
                        acc1 = (pred1_test == y_test1).float().mean()
                        acc2 = (pred2_test == y_test2).float().mean()
                        all_accs.append((acc1 + acc2)/2)
                    acc_test = torch.stack(all_accs).mean()

                    print(global_step, epoch, loss.item(), acc_train.item(), acc_test.item())
                    writer.add_scalar("Loss/total", loss.item(), global_step)
                    writer.add_scalar("Accuracy/train", acc_train.item(), global_step)
                    writer.add_scalar("Accuracy/test", acc_test.item(), global_step)

                model.train()

            # Save checkpoint
            if global_step % 1000 == 0:
                torch.save(model.state_dict(), os.path.join(save_dir, f"model_step_{global_step}.pt"))
            global_step += 1

    # Save final model
    torch.save(model.state_dict(), os.path.join(save_dir, "final_model.pt"))
    print("Final model saved.")

# ----------------------------
# --- Argument parser ---
# ----------------------------
def parse_args(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--n_epoch", type=int, default=3)
    parser.add_argument("--n_batch_size", type=int, default=128)
    parser.add_argument("--reg_lambda", type=float, default=1e-5)
    parser.add_argument("--keep_prob", type=float, default=0.8)
    parser.add_argument("--cross_stitch_enabled", type=bool, default=True)
    parser.add_argument("--train_annotations", type=str, default='archive/train_dataset/train.json')
    parser.add_argument("--train_img_dir", type=str, default='archive/train_dataset')
    parser.add_argument("--test_annotations", type=str, default='archive/train_dataset/train.json')
    parser.add_argument("--test_img_dir", type=str, default='archive/train_dataset')
    return parser.parse_args(argv)

# ----------------------------
# --- Main ---
# ----------------------------
if __name__ == "__main__":
    args = parse_args(sys.argv[1:])
    train(args)
