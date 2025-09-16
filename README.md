# Handwritten Digits Classification with Artificial Neural Network

A machine learning project that classifies handwritten digits (0-9) using an Artificial Neural Network (ANN). This project implements digit recognition using the MNIST dataset with neural network architectures built from scratch or using popular ML frameworks.

## 🎯 Objective

Build and train an artificial neural network to accurately classify handwritten digits from the MNIST dataset, achieving high accuracy in digit recognition tasks.

## 🔧 Technologies Used

- Python
- TensorFlow/Keras or PyTorch
- NumPy
- Matplotlib
- Pandas
- Scikit-learn
- Jupyter Notebook

## 📊 Dataset

- **MNIST Dataset**: 70,000 images of handwritten digits (28x28 pixels)
- **Training Set**: 60,000 images
- **Test Set**: 10,000 images
- **Classes**: 10 (digits 0-9)

## 🏗️ Model Architecture

- Input Layer: 784 neurons (28x28 flattened)
- Hidden Layers: Fully connected layers with ReLU activation
- Output Layer: 10 neurons with softmax activation
- Loss Function: Categorical crossentropy
- Optimizer: Adam

## 🚀 Quick Start

1. Clone the repository:
```bash
git clone https://github.com/bruteforce1127/HandwrittenDigitsClassificationAnn.git
cd HandwrittenDigitsClassificationAnn
```

2. Install dependencies:
```bash
pip install tensorflow numpy matplotlib pandas scikit-learn jupyter
```

3. Run the Jupyter notebook or Python script to train and evaluate the model.

