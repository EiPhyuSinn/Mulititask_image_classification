import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
from weather_time_classification import MultiTaskCNN  # your training model file

# ----- Device -----
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----- Transform -----
transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# ----- Load model -----
model_path = "custom_dataset_multi_task_model/final_model.pt"  # change to your checkpoint
model = MultiTaskCNN()
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

# ----- Load label mapping from JSON -----
def load_label_mapping(annotation_file):
    with open(annotation_file, "r") as f:
        data = json.load(f)["annotations"]
    period_labels = sorted(list({ann["period"] for ann in data}))
    weather_labels = sorted(list({ann["weather"] for ann in data}))
    return period_labels, weather_labels

period_labels, weather_labels = load_label_mapping("archive/train_dataset/train.json")

# ----- Inference function -----
def predict_image(image_path):
    img_orig = Image.open(image_path).convert("RGB")  # original color for plotting
    img_gray = img_orig.convert("L")                  # grayscale for model input
    img_tensor = transform(img_gray).unsqueeze(0).to(device)
    with torch.no_grad():
        out1, out2 = model(img_tensor)
        pred1 = torch.argmax(out1, dim=1).item()
        pred2 = torch.argmax(out2, dim=1).item()
    return img_orig, period_labels[pred1], weather_labels[pred2]

# ----- Plot image with labels -----
def plot_image_with_labels(img, period, weather, save_path=None):
    plt.figure(figsize=(3,3))
    plt.imshow(img)  # RGB image
    plt.title(f"Period: {period}\nWeather: {weather}")
    plt.axis('off')
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.close()

# ----- Inference for single image or folder -----
def predict_and_plot(input_path, save_dir=None):
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir)

    if os.path.isfile(input_path):
        img, period, weather = predict_image(input_path)
        save_path = os.path.join(save_dir, os.path.basename(input_path)) if save_dir else None
        plot_image_with_labels(img, period, weather, save_path)
    elif os.path.isdir(input_path):
        for fname in os.listdir(input_path):
            if fname.lower().endswith(('.png','.jpg','.jpeg')):
                fpath = os.path.join(input_path, fname)
                img, period, weather = predict_image(fpath)
                save_path = os.path.join(save_dir, fname) if save_dir else None
                plot_image_with_labels(img, period, weather, save_path)
    print('All done')

# ----- Example usage -----
if __name__ == "__main__":
    input_path = "/home/gwm-279/Desktop/multitask_classification/Cross-stitch-Networks-for-Multi-task-Learning/archive/test_dataset/test_images"       # single image or folder
    save_dir = "predictions"    # optional: save plotted images
    predict_and_plot(input_path, save_dir)
