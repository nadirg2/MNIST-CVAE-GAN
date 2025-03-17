import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from models.encoder import ConvEncoder
from models.decoder import ConvDecoder
from models.discriminator import ConvDiscriminator
import os

# Параметры
latent_dim = 64
num_classes = 10
batch_size = 128
epochs = 20
learning_rate_E = 0.0002
learning_rate_D = 0.0005
learning_rate_Dis = 0.00005
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Загрузка данных
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
train_dataset = datasets.MNIST(root="./data", train=True, transform=transform, download=True)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Инициализация моделей
encoder = ConvEncoder(latent_dim, num_classes).to(device)
decoder = ConvDecoder(latent_dim, num_classes).to(device)
discriminator = ConvDiscriminator(num_classes).to(device)

encoder.train()
decoder.train()
discriminator.train()

# Оптимизаторы
optimizer_E = optim.Adam(encoder.parameters(), lr=learning_rate_E)
optimizer_D = optim.Adam(decoder.parameters(), lr=learning_rate_D)
optimizer_Dis = optim.Adam(discriminator.parameters(), lr=learning_rate_Dis)

# Функции потерь
criterion_gan = nn.BCELoss()
criterion_recon = nn.MSELoss()
for epoch in range(epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        # ====== Обучение энкодера и декодера ======
        optimizer_E.zero_grad()
        optimizer_D.zero_grad()

        real_labels = torch.ones(batch_size, 1).to(device)
        fake_labels = torch.zeros(batch_size, 1).to(device)

        # Кодирование
        mu, logvar = encoder(images, labels)
        z = mu + torch.exp(0.5 * logvar) * torch.randn_like(logvar)

        # Декодирование
        recon_images = decoder(z, labels)

        # Потери VAE
        loss_recon = criterion_recon(recon_images, images)
        loss_kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        loss_vae = loss_recon + 0.0001 * loss_kl

        # Потери GAN
        fake_score = discriminator(recon_images, labels)
        loss_gan = criterion_gan(fake_score, real_labels)

        # Общий loss
        loss_E_D = loss_vae + loss_gan
        loss_E_D.backward()
        optimizer_E.step()
        optimizer_D.step()

        # ====== Обучение дискриминатора ======
        optimizer_Dis.zero_grad()

        # Потери на реальных данных
        real_score = discriminator(images, labels)
        loss_real = criterion_gan(real_score, real_labels)

        # Потери на фейковых данных
        fake_score = discriminator(recon_images.detach(), labels)
        loss_fake = criterion_gan(fake_score, fake_labels)

        # Общий loss
        loss_Dis = loss_real + loss_fake
        loss_Dis.backward()
        optimizer_Dis.step()

        if i % 100 == 0:
            print(f"Epoch [{epoch}/{epochs}], Step [{i}/{len(train_loader)}], "
                  f"Loss VAE: {loss_vae.item():.4f}, Loss GAN: {loss_gan.item():.4f}, "
                  f"Loss D: {loss_Dis.item():.4f}")

    # Сохранение модели
    if not os.path.exists("checkpoints"):
        os.makedirs("checkpoints")
    torch.save(encoder.state_dict(), "checkpoints/encoder.pth")
    torch.save(decoder.state_dict(), "checkpoints/decoder.pth")
    torch.save(discriminator.state_dict(), "checkpoints/discriminator.pth")
