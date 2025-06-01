# CIFAR-100 Image Classification

This repository contains implementations of various deep learning models for image classification on the CIFAR-100 dataset. The project includes both traditional deep learning architectures and modern transformer-based models.

**All training results and model checkpoints are stored in the 'training' folder.**

## Models Implemented

- **Custom CNN**: A custom Convolutional Neural Network architecture
- **ResNet50**: Deep residual learning network with 50 layers
- **VGG16**: Visual Geometry Group network with 16 layers
- **DenseNet**: Densely Connected Convolutional Networks
- **ConvNeXt**: Modern ConvNet architecture
- **Vision Transformer (ViT)**: Transformer-based architecture for image classification
- **Swin Transformer**: Hierarchical Vision Transformer
- **Support Vector Machine (SVM)**: Traditional machine learning approach

## Dataset

The project uses the CIFAR-100 dataset, which consists of:
- 60,000 32x32 color images
- 100 classes
- 600 images per class
- 50,000 training images and 10,000 test images

## Requirements

- Python 3.x
- PyTorch
- torchvision
- torchsummary
- dataset
- transformer
- tqdm

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/Image-Classification-CIFAR100.git
cd Image-Classification-CIFAR100
```

2. Install the required packages:
```bash
pip install -r requirements.txt
```


## Usage

To train and evaluate models, use the following command with the desired model option:

```bash
python main.py --option <number>
```

Available model options:
- `--option 1`: Custom Convolutional Neural Network (BasicCNN)
- `--option 2`: ResNet50
- `--option 3`: VGG16
- `--option 4`: DenseNet121
- `--option 5`: EfficientNet
- `--option 6`: ConvNeXt
- `--option 7`: Vision Transformer
- `--option 8`: Swin Transformer
- `--option 9`: Support Vector Machine with Feature Extraction

Example:
```bash
python main.py --option 2  # To train and evaluate the ResNet50 model
```

## Features

- Multiple state-of-the-art model implementations
- Easy-to-use configuration system
- Training and evaluation pipeline
- Support for both CNN and Transformer architectures
- Traditional machine learning approach with SVM

## Contributing

Feel free to submit issues and enhancement requests!

## License

This project is licensed under the MIT License - see the LICENSE file for details.