import os
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import torchvision.transforms as transforms

class ScratchDataset(Dataset):
    def __init__(self, root_dir, transform=None, train=True, limit=None, specific_text=None):
        self.root_dir = os.path.join(root_dir, 'train' if train else 'test')
        self.transform = transform
        self.samples = []

        # Load good images
        good_dir = os.path.join(self.root_dir, 'good')
        for img_name in os.listdir(good_dir):
            # Optional: Filter by specific text
            if specific_text and specific_text not in img_name:
                continue
            self.samples.append((os.path.join(good_dir, img_name), 0))
        
        # Load bad images
        bad_dir = os.path.join(self.root_dir, 'bad')
        for img_name in os.listdir(bad_dir):
            if specific_text and specific_text not in img_name:
                continue
            self.samples.append((os.path.join(bad_dir, img_name), 1))

        # Limit dataset size
        if limit:
            self.samples = self.samples[:limit]
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        """Fetch an image and its label by index."""
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')  # Open image and convert to RGB
        
        # Convert image to NumPy array
        image = np.array(image)
        
        # Apply transformations (if provided)
        if self.transform:
            image = self.transform(image=image)['image']
        
        return image, label
