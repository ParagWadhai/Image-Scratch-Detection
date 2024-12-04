import torch
from torch.utils.data import DataLoader
from .dataset import ScratchDataset
from .model import ScratchDetector
from .augmentations import get_test_transforms
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

def test_model(model, test_loader):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    return np.array(all_preds), np.array(all_labels)

def main():
    # Load test dataset
    test_dataset = ScratchDataset('data', transform=get_test_transforms(), train=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Load model
    model = ScratchDetector()
    model.load_state_dict(torch.load('scratch_detector.pth'))
    
    # Test model
    predictions, labels = test_model(model, test_loader)
    
    # Print metrics
    print("\nClassification Report:")
    print(classification_report(labels, predictions))
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(labels, predictions))

if __name__ == '__main__':
    main()