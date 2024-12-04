# Scratch Detection Project

This project implements a deep learning solution for detecting scratches in images.

## Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (recommended) or CPU
- Good and bad sample images for training

## Directory Structure

```
scratch_detection/
├── data/
│   ├── train/
│   │   ├── good/     # Place good training images here
│   │   └── bad/      # Place scratched training images here
│   └── test/
│       ├── good/     # Place good test images here
│       └── bad/      # Place scratched test images here
├── src/
│   ├── dataset.py
│   ├── model.py
│   ├── augmentations.py
│   ├── train.py
│   └── test.py
├── main.py
├── requirements.txt
└── README.md
```

## Setup Instructions

1. Create and activate a virtual environment:

```bash
# Create virtual environment
python -m venv venv

# Activate on Windows
venv\Scripts\activate

# Activate on Linux/Mac
source venv/bin/activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Prepare your dataset:
   - Place your "good" (non-scratched) images in `data/train/good/` and `data/test/good/`
   - Place your "bad" (scratched) images in `data/train/bad/` and `data/test/bad/`
   - Supported image formats: JPG, PNG

## Running the Project

1. To train the model:
```bash
python main.py --mode train
```
This will:
- Train the model on your dataset
- Save the trained model as 'scratch_detector.pth'
- Generate a training loss plot as 'training_loss.png'

2. To test the model:
```bash
python main.py --mode test
```
This will:
- Load the trained model
- Evaluate it on the test dataset
- Display classification metrics and confusion matrix


### Training Mode
```
Epoch 1/5
Train Loss: 1.2638, Val Loss: 0.4744
Epoch 2/5
Train Loss: 0.4536, Val Loss: 0.4475
Epoch 3/5
Train Loss: 0.4172, Val Loss: 0.4396
Epoch 4/5
Train Loss: 0.4018, Val Loss: 0.4165
Epoch 5/5
Train Loss: 0.3970, Val Loss: 0.3781
...
```

### Testing Mode
```
Classification Report:
              precision    recall  f1-score   support
           0       0.88      0.96      0.91        4157
           1       0.71      0.45      0.55        1023

    accuracy                           0.86        5180
   macro avg       0.79      0.70      0.73        5180
weighted avg       0.84      0.86      0.84        5180

Confusion Matrix:
[[3973  184]
 [ 567 456]]
```

## Troubleshooting

1. If you get CUDA errors:
   - The model will automatically fall back to CPU if CUDA is not available
   - Ensure you have PyTorch installed with CUDA support if using GPU

2. If you get image loading errors:
   - Ensure your images are in supported formats (JPG, PNG)
   - Check that your directory structure matches the expected layout

3. For memory issues:
   - Reduce batch_size in train.py and test.py
   - Resize images to smaller dimensions in augmentations.py