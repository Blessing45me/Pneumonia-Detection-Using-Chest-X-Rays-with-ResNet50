## Overview

Pneumonia diagnosis commonly relies on expert interpretation of chest X-rays. This project explores an automated binary image-classification workflow that predicts whether a chest X-ray belongs to the **NORMAL** or **PNEUMONIA** class.

The implementation uses a pretrained **ResNet50** model as a feature extractor and then fine-tunes selected deeper layers. Adaptive training callbacks help control overfitting and adjust the learning rate when validation performance stops improving.

> **Important:** This is an educational machine-learning project and is not a medical device or a substitute for diagnosis by a qualified healthcare professional.

## Problem

Chest X-ray interpretation requires trained radiologists and may be affected by workload, time constraints, and variation in experience. The project investigates whether transfer learning can provide a reproducible baseline for AI-assisted pneumonia image classification.

## Solution

The pipeline:

1. Downloads the chest X-ray dataset through KaggleHub.
2. Loads training, validation, and testing images from class folders.
3. Resizes images to `224 × 224`.
4. Applies ResNet50 preprocessing and training-time augmentation.
5. Trains a binary classifier on frozen ResNet50 features.
6. Fine-tunes the final ResNet50 layers with a smaller learning rate.
7. Evaluates the model using accuracy, precision, recall, F1-score, and a confusion matrix.

## Technology Stack

- Python
- TensorFlow and Keras
- ResNet50 with ImageNet weights
- KaggleHub
- Scikit-learn
- Matplotlib
- NumPy
- Google Colab / Jupyter Notebook

## Dataset

The notebooks use the Kaggle chest X-ray pneumonia dataset:

`paultimothymooney/chest-xray-pneumonia`

Directory counts produced by the main notebook:

| Split | Images |
|---|---:|
| Training | 4,173 |
| Validation | 1,043 |
| Testing | 624 |

The test set contains:

- 234 normal images
- 390 pneumonia images

## Model Architecture

- ResNet50 convolutional base with ImageNet weights
- Global Average Pooling
- Dense layer with 128 units and ReLU activation
- Dropout with a rate of 0.5
- Sigmoid output for binary classification

## Adaptive Training Strategy

The project uses:

- **Transfer learning:** freezes the pretrained model during initial training.
- **Fine-tuning:** unfreezes the final 30 ResNet50 layers.
- **EarlyStopping:** restores the best weights when validation loss stops improving.
- **ReduceLROnPlateau:** lowers the learning rate when validation performance plateaus.
- **Data augmentation:** uses rotation, zoom, and horizontal flipping.

## Reproduced Notebook Results

Results saved in `pneumonia_detection_resnet50.ipynb`:

| Metric | Result |
|---|---:|
| Test accuracy | 88.46% |
| Pneumonia precision | 90% |
| Pneumonia recall | 92% |
| Pneumonia F1-score | 91% |

Confusion matrix:

```text
[[193, 41],
 [ 31, 359]]
```

This means the model correctly classified 193 normal X-rays and 359 pneumonia X-rays, while producing 41 false positives and 31 false negatives.

## Repository Structure

```text
pneumonia-detection-resnet50/
├── README.md
├── requirements.txt
├── .gitignore
├── pneumonia_detection_resnet50.ipynb
├── self_adaptive_visualization.ipynb
└── Pneumonia_Detection_Using_Chest_X_Rays.docx
```

## Running the Project

### Google Colab

1. Open `pneumonia_detection_resnet50.ipynb` in Google Colab.
2. Select a GPU runtime.
3. Run the notebook cells in order.
4. KaggleHub downloads the dataset automatically.
5. Review the final classification report and confusion matrix.

### Local Jupyter Environment

```bash
git clone https://github.com/YOUR-USERNAME/pneumonia-detection-resnet50.git
cd pneumonia-detection-resnet50

python -m venv .venv
```

Activate the environment:

**Windows PowerShell**

```powershell
.venv\Scripts\Activate.ps1
```

**macOS or Linux**

```bash
source .venv/bin/activate
```

Install dependencies and launch Jupyter:

```bash
pip install -r requirements.txt
jupyter notebook
```

## Limitations

- The dataset is imbalanced toward pneumonia images.
- Performance may not generalize to images from other hospitals or imaging devices.
- The current model does not include an explainability method such as Grad-CAM.
- The project has not undergone clinical validation.
- Model behavior can vary across TensorFlow versions, hardware, random initialization, and data-loading conditions.

## Future Improvements

- Add Grad-CAM visualizations for model interpretability.
- Use class weighting or balanced sampling.
- Compare ResNet50 with EfficientNet, DenseNet, and MobileNet.
- Add ROC-AUC and precision-recall curve evaluation.
- Save trained models and experiment configurations.
- Add a Streamlit or FastAPI interface for research demonstrations.
- Test on an external dataset before making generalization claims.

## Reproducibility Note

The accompanying paper draft reports different performance values and a different training configuration from the saved notebook outputs. Before using the paper as a final publication artifact, align the dataset split, batch size, epoch count, augmentation settings, and reported metrics with one fully reproducible experiment.

## Author

**Blessing Ugwuoti**  
University of Central Missouri
