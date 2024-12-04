import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from .dataset import ScratchDataset
from .model import ScratchDetector
from .augmentations import get_train_transforms, get_test_transforms
import matplotlib.pyplot as plt

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    train_losses = []
    val_losses = []
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
        epoch_loss = running_loss / len(train_loader)
        train_losses.append(epoch_loss)
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
        val_loss = val_loss / len(val_loader)
        val_losses.append(val_loss)
        
        print(f'Epoch {epoch+1}/{num_epochs}')
        print(f'Train Loss: {epoch_loss:.4f}, Val Loss: {val_loss:.4f}')
    
    return train_losses, val_losses

def main():
    # Dataset setup with limits and specific text filtering
    train_dataset = ScratchDataset('data_small', transform=get_train_transforms(), train=True)
    val_dataset = ScratchDataset('data_small', transform=get_test_transforms(), train=False)
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)  # Smaller batch size
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    
    # Model setup
    model = ScratchDetector()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Train model
    train_losses, val_losses = train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=5)  # Fewer epochs
    
    # Save model
    torch.save(model.state_dict(), 'scratch_detector.pth')
    
    # Plot losses
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('training_loss.png')
if __name__ == '__main__':
    main()
