import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
from fashion import MultiTaskCNN

# ----- Device -----
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----- Transform -----
transform = transforms.Compose([
    transforms.Grayscale(),          # ensure single channel
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# ----- Load model -----
model_path = "fashion_mnist_multi_task_model/final_model.pt"
model = MultiTaskCNN()
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

# ----- Label mapping -----
task1_labels = ["T-shirt/top","Trouser","Pullover","Dress","Coat","Sandal","Shirt","Sneaker","Bag","Ankle boot"]
task2_labels = ["shoe","girl","other"]

# ----- Inference function -----
def predict_image(image_path):
    img_orig = Image.open(image_path).convert("RGB")  # original color for plotting
    img_gray = img_orig.convert("L")                  # grayscale for model input
    img_tensor = transform(img_gray).unsqueeze(0).to(device)  # batch dim
    with torch.no_grad():
        out1, out2 = model(img_tensor)
        pred1 = torch.argmax(out1, dim=1).item()
        pred2 = torch.argmax(out2, dim=1).item()
    return img_orig, task1_labels[pred1], task2_labels[pred2]

# ----- Plot image with labels -----
def plot_image_with_labels(img, task1, task2, save_path=None):
    plt.figure(figsize=(3,3))
    plt.imshow(img)  # RGB image, no cmap needed
    plt.title(f"Task1: {task1}\nTask2: {task2}")
    plt.axis('off')
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')


# ----- Inference for single image or folder -----
def predict_and_plot(input_path, save_dir=None):
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    if os.path.isfile(input_path):
        img, task1, task2 = predict_image(input_path)
        save_path = os.path.join(save_dir, os.path.basename(input_path)) if save_dir else None
        plot_image_with_labels(img, task1, task2, save_path)
    elif os.path.isdir(input_path):
        for fname in os.listdir(input_path):
            if fname.lower().endswith(('.png','.jpg','.jpeg')):
                fpath = os.path.join(input_path, fname)
                img, task1, task2 = predict_image(fpath)
                save_path = os.path.join(save_dir, fname) if save_dir else None
                plot_image_with_labels(img, task1, task2, save_path)

# ----- Example usage -----
if __name__ == "__main__":
    input_path = "images"       # single image or folder
    save_dir = "predictions"    # optional: save plotted images
    predict_and_plot(input_path, save_dir)
