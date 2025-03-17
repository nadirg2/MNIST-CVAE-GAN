import torch
import torch.nn as nn

class ConvDiscriminator(nn.Module):
    def __init__(self, num_classes):
        super(ConvDiscriminator, self).__init__()
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

        # Полносвязный слой для классификации
        self.fc = nn.Sequential(
            nn.Linear((256 + num_classes), 1),  # Добавляем метки к латентному вектору
            nn.Sigmoid()
        )

    def forward(self, x, labels):
        # Применение сверточных слоев
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)

        # Встраивание меток
        c = self.label_embedding(labels)

        # Конкатенация с латентным вектором
        x = torch.cat([x, c], dim=1)

        x = x.view(x.size(0), -1)  # Выравнивание для полносвязного слоя

        # Классификация
        return self.fc(x)