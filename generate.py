import torch
import argparse
from models.decoder import ConvDecoder
import matplotlib.pyplot as plt

# Параметры по умолчанию
latent_dim = 64
num_classes = 10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Парсинг аргументов командной строки
parser = argparse.ArgumentParser(description="Generate images using a trained CVAE-GAN model.")
parser.add_argument("--checkpoint", type=str, required=True, help="Path to the checkpoint file (decoder weights).")
parser.add_argument("--num_samples", type=int, default=10, help="Number of images to generate.")
args = parser.parse_args()

# Загрузка модели
decoder = ConvDecoder(latent_dim, num_classes).to(device)
decoder.load_state_dict(torch.load(args.checkpoint, map_location=device))
decoder.eval()

# Генерация изображений
z = torch.randn(args.num_samples, latent_dim).to(device)  # Случайный шум
labels = torch.arange(0, args.num_samples).to(device) % num_classes  # Метки от 0 до 9

with torch.no_grad():
    generated_images = decoder(z, labels).cpu()

# Визуализация
fig, axes = plt.subplots(1, args.num_samples, figsize=(args.num_samples * 2, 2))
for i, ax in enumerate(axes):
    ax.imshow(generated_images[i].squeeze(), cmap="gray")
    ax.axis("off")
plt.show()
