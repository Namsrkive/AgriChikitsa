import torch
import torch.nn as nn

class CropDiseaseModel(nn.Module):
    def __init__(self, num_classes=38):
        super(CropDiseaseModel, self).__init__()
        # Example of model architecture
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(64, 128)
        self.layer2 = self._make_layer(128, 256)
        self.fc = nn.Linear(256, num_classes)

    def _make_layer(self, in_channels, out_channels):
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        ]
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.bn1(self.conv1(x))
        x = torch.relu(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

def load_model(model_path, num_classes=38):
    model = CropDiseaseModel(num_classes=num_classes)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load the state_dict and map it to the model
    checkpoint = torch.load(model_path, map_location=device)
    
    # The issue comes from mismatching keys in the saved state_dict, we need to handle that
    # Fix key names in the checkpoint (e.g., 'network.' prefix in the keys)
    checkpoint = {key.replace('network.', ''): value for key, value in checkpoint.items()}
    
    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()
    return model
model = load_model("crop_disease_detection.pth", num_classes=38)
