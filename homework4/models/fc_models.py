import torch.nn as nn

class DeepFCNet(nn.Module):
    """
    Глубокая полносвязная сеть для сравнения со сверточными сетями.
    Включает в себя BatchNorm и Dropout для стабильного обучения.
    """
    def __init__(self, input_size=784, num_classes=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_size, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.net(x)