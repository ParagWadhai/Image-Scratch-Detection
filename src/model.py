import torch
import torch.nn as nn

class ScratchDetector(nn.Module):
    def __init__(self):
        super(ScratchDetector, self).__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),  # Reduced filters
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # Reduced filters
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(64 * 56 * 56, 128),  # Adjusted for reduced layers
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(128, 2)
        )
        
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
