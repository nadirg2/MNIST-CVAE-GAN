import torch
import torch.nn as nn


class ConvDecoder(nn.Module):
    def __init__(self, latent_dim, num_classes):
        super(ConvDecoder, self).__init__()
        self.label_embedding = nn.Embedding(num_classes, num_classes)

        # Полносвязный слой для преобразования латентного вектора
        self.fc = nn.Linear(latent_dim + num_classes, 256 * 2 * 2)

        # Сверточные слои
        self.conv_layers = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )


    def forward(self, z, labels):
        # Встраивание меток и конкатенация с латентным вектором
        c = self.label_embedding(labels)
        z = torch.cat([z, c], dim=1)

        # Преобразование в пространство для сверток
        x = self.fc(z)
        x = x.view(-1, 256, 2, 2)

        # Применение сверточных слоев
        x = self.conv_layers(x)
        return x