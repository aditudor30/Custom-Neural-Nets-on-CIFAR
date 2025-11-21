# Custom-Neural-Nets-on-CIFAR
Implementing custom neural network architectures from scratch on the CIFAR datasets

## Overview
This repository explores custom implementations of convolutional neural networks (CNNs), including architectures inspired by VGG and ResNet, trained on the CIFAR-10 and CIFAR-100 datasets.

The goals of the project are:
1. Understand how popular deep learning architectures are built from first principles.
2. Study how hardware limits (GPU memory, batch size, training time) influence model performance.

## Motivation
This project is part of my ongoing learning process in deep learning and neural network architecture design.
I aim to build models manually, experiment with deeper structures, and analyze how training behaves under different constraints.

## Repository Structure
| File / Directory        | Description                                                         |
|-------------------------|---------------------------------------------------------------------|
| `/data/`                | Scripts and notebooks for data download and preprocessing           |
| `simple_models.ipynb`   | Baseline models and initial experiments                            |
| `vgg-notebook.ipynb`    | VGG-style architecture implementation from scratch                 |
| `vgg-notebook_100.ipynb`| Extended VGG experiment with deeper layers                         |

## Key Features
- Custom CNN architectures implemented manually.
- Comparison of baseline, VGG-style, and residual-inspired architectures.
- Experiments involving:
  - Hardware limitations
  - GPU memory constraints
  - Batch size and epoch variation
  - Data augmentation
- Visualization of training curves:
  - Loss
  - Accuracy
  - Overfitting and generalization behaviour

## Requirements
- Python 3.x
- NumPy
- Matplotlib
- PyTorch
- Jupyter Notebook
- Optional GPU support

## Usage
### 1. Clone repository
```bash
git clone https://github.com/aditudor30/Custom-Neural-Nets-on-CIFAR.git
cd Custom-Neural-Nets-on-CIFAR
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run notebooks
```bash
jupyter notebook
```

## Experimentation Ideas
- Implement a ResNet-style architecture with skip connections.
- Compare performance on CIFAR-10 vs CIFAR-100.
- Perform hyperparameter tuning.
- Add regularization (dropout, batchnorm, weight decay).
- Test behaviour on limited hardware.

## Results & Observations
Initial experiments with a VGG-based model achieved around **87% accuracy** on CIFAR-10 after 50 epochs.
Increasing depth without skip connections caused instability and overfitting.

## Future Improvements
As we can see from the `vgg-notebook_100.ipynb`, our VGG-based model could not achieve the same performance on CIFAR-100.
In order to make improvements a new notebook will be built testing a ResNet-based model that will showcase how much further can this project go on a PC.

## License
This project is for educational purposes. You may fork, modify, or extend it freely.

