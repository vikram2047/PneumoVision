# Pediatric Chest Pneumonia Classification: Leveraging Traditional CNN with GAN for Data Balancing

[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Flask](https://img.shields.io/badge/Flask-Web%20App-000000?logo=flask&logoColor=white)](https://flask.palletsprojects.com/)

## Abstract

This project presents an innovative approach to pediatric chest pneumonia classification using deep learning techniques. We address the critical challenge of class imbalance in medical imaging datasets by employing Generative Adversarial Networks (GANs) for synthetic data generation, combined with traditional data augmentation methods. Our methodology demonstrates significant improvements in classification accuracy and model robustness for pneumonia detection in pediatric chest X-rays.

## Table of Contents

- [Introduction](#introduction)
- [Methodology](#methodology)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Experimental Setup](#experimental-setup)
- [Results](#results)
- [Demo](#demo)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [Citation](#citation)
- [License](#license)

## Introduction

Pneumonia is a leading cause of mortality in children under 5 years globally. Early and accurate diagnosis through chest X-ray analysis is crucial for effective treatment. However, medical imaging datasets often suffer from class imbalance, where normal cases significantly outnumber pneumonia cases. This project addresses this challenge by:

1. **Data Balancing** — Using GANs to generate synthetic normal chest X-ray images
2. **Data Augmentation** — Applying traditional augmentation techniques
3. **Deep Learning** — Implementing a CNN architecture optimized for binary classification
4. **Model Evaluation** — Comprehensive comparison across different data balancing strategies

## Methodology

### 1. Data Preprocessing

- **Image Resizing**: All images resized to 148×148 pixels
- **Normalization**: Pixel values normalized to [0, 1] range
- **Grayscale Conversion**: Converted to single-channel grayscale images

### 2. Data Balancing Strategies

#### Strategy 1: Traditional Data Augmentation

- Rotation (±35°)
- Width/Height shifts (0.1 / 0.08)
- Shear transformation (0.2)
- Zoom (0.2)

#### Strategy 2: GAN-based Data Generation

- **Architecture**: Deep Convolutional GAN (DCGAN)
- **Generator**: 4-layer transposed convolution network
- **Discriminator**: 4-layer convolution network
- **Training**: 40,000 iterations with RMSprop optimizer
- **Output**: 2,534 synthetic normal X-ray images

#### Strategy 3: Combined Approach

- Integration of augmented and GAN-generated images
- Comprehensive dataset balancing

### 3. CNN Architecture

Our CNN architecture consists of:

- **Convolutional Layers**: Multiple Conv2D layers with ReLU activation
- **Pooling Layers**: MaxPooling2D for dimensionality reduction
- **Normalization**: BatchNormalization for training stability
- **Regularization**: Dropout layers to prevent overfitting
- **Output**: Single sigmoid neuron for binary classification

## Dataset

- **Source**: [Kaggle Chest X-Ray Pneumonia Dataset](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
- **Total Images**: 5,863 X-ray images
- **Classes**: Normal, Pneumonia
- **Split**: Train (5,216), Validation (16), Test (624)
- **Image Format**: JPEG
- **Resolution**: Variable (resized to 148×148 for model input)

### Class Distribution

| Split    | Normal | Pneumonia |
| -------- | ------ | --------- |
| Training | 1,341  | 3,875     |
| Test     | 234    | 390       |

## Model Architecture

The deployed model uses the following Sequential architecture:

```python
model = Sequential([
    Conv2D(32, (3, 3), input_shape=(148, 148, 1), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    BatchNormalization(),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(1, activation='sigmoid')
])
```

**Hyperparameters**:

| Parameter       | Value              |
| --------------- | ------------------ |
| Optimizer       | Adam (lr = 0.001)  |
| Loss Function   | Binary Crossentropy|
| Batch Size      | 132                |
| Epochs          | 15–50 (varies)     |
| Input Size      | 148 × 148 × 1     |

## Experimental Setup

We conducted four comprehensive experiments:

1. **Baseline** — Original imbalanced dataset
2. **Augmentation** — Traditional data augmentation
3. **GAN Generation** — Synthetic data generation
4. **Combined** — Augmentation + GAN generation

### Hyperparameter Search Space

| Component            | Configurations Explored                                      |
| -------------------- | ------------------------------------------------------------ |
| Conv Layers          | [32,64], [32,64,128], [32,64,128,128], [32,64,128,256], [64,128,256,256] |
| Dense Layers         | [], [512], [256], [256,256], [512,256], [512,512]            |
| Dropout Rates        | [0.2], [0.5], [0.2,0.2], [0.2,0.5], [0.5,0.5]              |
| Batch Normalization  | True / False                                                 |

## Results

### Performance Metrics

| Experiment            | Accuracy   | Precision | Recall   | F1-Score |
| --------------------- | ---------- | --------- | -------- | -------- |
| Baseline (Imbalanced) | 85.26%     | 0.85      | 0.85     | 0.85     |
| Augmentation          | 86.06%     | 0.86      | 0.86     | 0.86     |
| GAN Generation        | 84.78%     | 0.85      | 0.85     | 0.85     |
| **Combined Approach** | **87.18%** | **0.87**  | **0.87** | **0.87** |

### Key Findings

1. **Data Balancing Effectiveness** — Combined approach achieved highest accuracy (87.18%)
2. **GAN Quality** — Generated images maintained realistic chest X-ray characteristics
3. **Model Robustness** — Balanced datasets showed improved generalization
4. **Clinical Relevance** — High recall for pneumonia detection (87%)

## Demo

### Live Demo

- **Hugging Face Space**: [Try the model online](https://huggingface.co/spaces/AbdulManaf12/Pediatric-Chest-Xray-Pneumonia-Classification-System)
- **Web Application**: [abdulmanaf12.pythonanywhere.com](http://abdulmanaf12.pythonanywhere.com/)

### Features

- Real-time chest X-ray classification
- Confidence-based probability scores for predictions
- Drag-and-drop image upload interface
- Support for PNG, JPG, and JPEG image formats
- Responsive, mobile-friendly design

## Installation

### Prerequisites

- Python 3.10
- pip package manager

### Setup

```bash
# Clone the repository
git clone https://github.com/vikram2047/PneumoVision.git
cd PneumoVision

# Create a virtual environment (recommended)
python -m venv venv

# Activate the virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Dependencies

The project requires the following Python packages (listed in `requirements.txt`):

```
flask
tensorflow>=2.0.0
numpy
opencv-python
```

> **Note**: The pre-trained model weights (`weights.h5`) should be placed in the `static/` directory before running the application.

## Usage

### Running the Web Application

```bash
python main.py
```

The Flask development server will start at `http://127.0.0.1:5000`. Open the URL in your browser to access the classification interface.

**How to use:**

1. Open the web interface in your browser
2. Drag and drop a chest X-ray image (or click to browse)
3. Click **Classify X-Ray**
4. View the prediction (Normal / Pneumonia) along with the model's confidence score

### Programmatic Prediction

```python
from Model import MyModel

# Initialize the model (loads weights from static/weights.h5)
model = MyModel()

# Place your image as static/image.png, then predict
prediction, probability = model.predict()
print(f"Prediction: {prediction}, Probability: {probability:.4f}")
```

### Training New Models

Run the Jupyter notebooks in the following order:

1. **`notebooks/image_generation.ipynb`** — GAN training and synthetic data generation
2. **`notebooks/Project.ipynb`** — Main training pipeline with hyperparameter tuning
3. **`notebooks/xai_experiment.ipynb`** — Explainable AI experiments

## Project Structure

```
├── main.py                         # Flask web application entry point
├── Model.py                        # CNN model definition and prediction logic
├── requirements.txt                # Python dependencies
├── README.md                       # Project documentation
├── .gitignore                      # Git ignore rules
│
├── static/                         # Static assets
│   ├── weights.h5                  # Pre-trained model weights
│   └── image.png                   # Uploaded image (created at runtime)
│
├── templates/                      # Jinja2 HTML templates
│   ├── index.html                  # Upload interface (with inline CSS)
│   └── results.html                # Classification results page
│
├── notebooks/                      # Jupyter notebooks
│   ├── Project.ipynb               # Main training & evaluation pipeline
│   ├── image_generation.ipynb      # DCGAN implementation & data generation
│   └── xai_experiment.ipynb        # Explainable AI experiments
│
└── Radiography/                    # Dataset directory (not tracked in git)
    ├── train/                      # Training images
    ├── val/                        # Validation images
    └── test/                       # Test images
```

## Technical Implementation

### GAN Architecture

**Generator Network**:

```
Dense(image_resize × image_resize × 128)
  → Reshape
  → Conv2DTranspose(128) → BatchNorm → ReLU
  → Conv2DTranspose(64)  → BatchNorm → ReLU
  → Conv2DTranspose(32)  → BatchNorm → ReLU
  → Conv2DTranspose(1)   → Sigmoid
```

**Discriminator Network**:

```
Conv2D(32)  → LeakyReLU
  → Conv2D(64)  → LeakyReLU
  → Conv2D(128) → LeakyReLU
  → Conv2D(256) → LeakyReLU
  → Flatten → Dense(1) → Sigmoid
```

### Data Pipeline

1. **Loading** — Multi-threaded image loading with OpenCV
2. **Preprocessing** — Resize to 148×148, normalize to [0, 1], convert to grayscale
3. **Augmentation** — Real-time augmentation during training
4. **Generation** — Batch generation of synthetic images via DCGAN
5. **Balancing** — Strategic dataset composition for improved training

## Future Work

- **Multi-class Classification** — Extend to differentiate bacterial vs. viral pneumonia
- **Advanced Architectures** — Implement ResNet, DenseNet, or Vision Transformers
- **Federated Learning** — Enable privacy-preserving collaborative training
- **Mobile Deployment** — Optimize with TensorFlow Lite for on-device inference
- **Clinical Validation** — Collaborate with medical institutions for real-world validation

## Contributing

Contributions are welcome! To contribute:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request
```

## Acknowledgments

- [Kaggle](https://www.kaggle.com/) for providing the chest X-ray pneumonia dataset
- The medical imaging research community
- Open-source contributors to TensorFlow, Flask, and related libraries

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

> **⚠️ Disclaimer**: This tool is for research and educational purposes only. It should not be used as a substitute for professional medical diagnosis. Always consult with qualified healthcare professionals for medical decisions.
