import torch
import torch.nn as nn


class ConvEncoder(nn.Module):
    def __init__(self, latent_dim, num_classes):
        super(ConvEncoder, self).__init__()
        self.label_embedding = nn.Embedding(num_classes, num_classes)

        # Сверточные слои
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1),  # 1 канал для изображения
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2)
        )

        self.adaptive_pool = nn.AdaptiveAvgPool2d((2, 2))  # Приводим к размеру [batch_size, 256, 2, 2]

        # Полносвязные слои для mu и logvar
        self.fc_mu = nn.Linear(256 * 2 * 2 + num_classes, latent_dim)  # Добавляем метки к латентному вектору
        self.fc_logvar = nn.Linear(256 * 2 * 2 + num_classes, latent_dim)

    def forward(self, x, labels):
        # Применение сверточных слоев
        x = self.conv_layers(x)

        x = self.adaptive_pool(x)

        x = x.view(x.size(0), -1)  # Выравнивание для полносвязного слоя

        # Встраивание меток
        c = self.label_embedding(labels)

        # Конкатенация с латентным вектором
        x = torch.cat([x, c], dim=1)

        # Получение mu и logvar
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar