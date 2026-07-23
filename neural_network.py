import torch.nn as nn

class SymptomClassifier(nn.Module):

    def __init__(self, input=131, output=41):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, output)
        )

    def forward(self, x):
        return self.net(x)