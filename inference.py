import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
from weather_bdd import MultiTaskCNN  # your training model file

# ----- Device -----
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----- Transform -----
transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# ----- Label mappings (predefined) -----
WEATHER_LABELS = ['clear', 'foggy', 'overcast', 'partly cloudy', 'rainy', 'snowy', 'undefined']
SCENE_LABELS   = ['city street', 'gas stations', 'highway', 'parking lot', 'residential', 'tunnel', 'undefined']
TIME_LABELS    = ['dawn/dusk', 'daytime', 'night', 'undefined']

# ----- Load model -----
model_path = "model/final_model.pt"  # change to your checkpoint
model = MultiTaskCNN()
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

# ----- Inference function -----
def predict_image(image_path):
    img_orig = Image.open(image_path).convert("RGB")  # original color for plotting
    img_gray = img_orig.convert("L")                  # grayscale for model input
    img_tensor = transform(img_gray).unsqueeze(0).to(device)
    with torch.no_grad():
        out_weather, out_scene, out_time = model(img_tensor)
        pred_weather = torch.argmax(out_weather, dim=1).item()
        pred_scene   = torch.argmax(out_scene, dim=1).item()
        pred_time    = torch.argmax(out_time, dim=1).item()
    return img_orig, WEATHER_LABELS[pred_weather], SCENE_LABELS[pred_scene], TIME_LABELS[pred_time]

# ----- Plot image with labels -----
def plot_image_with_labels(img, weather, scene, time, save_path=None):
    plt.figure(figsize=(10,10))
    plt.imshow(img)
    plt.title(f"Weather: {weather}\nScene: {scene}\nTime: {time}")
    plt.axis('off')
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.close()

# ----- Inference for single image or folder -----
def predict_and_plot(input_path, save_dir=None):
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir)

    if os.path.isfile(input_path):
        img, weather, scene, time = predict_image(input_path)
        save_path = os.path.join(save_dir, os.path.basename(input_path)) if save_dir else None
        plot_image_with_labels(img, weather, scene, time, save_path)
    elif os.path.isdir(input_path):
        for fname in os.listdir(input_path):
            if fname.lower().endswith(('.png','.jpg','.jpeg')):
                fpath = os.path.join(input_path, fname)
                img, weather, scene, time = predict_image(fpath)
                save_path = os.path.join(save_dir, fname) if save_dir else None
                plot_image_with_labels(img, weather, scene, time, save_path)
    print('All done')

# ----- Example usage -----
if __name__ == "__main__":
    # input_path = "/home/gwm-279/Downloads/10k_images_train/bdd100k/images/10k/train"
    input_path = "weather_old/archive/test_dataset/test_images"
    save_dir = "predictions"    # optional: save plotted images
    predict_and_plot(input_path, save_dir)
