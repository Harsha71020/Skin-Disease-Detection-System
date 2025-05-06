
# Skin Disease Classification Using Deep Learning

This project focuses on classifying images of 20 different skin diseases using a deep learning model trained on the [20 Skin Diseases Dataset](https://www.kaggle.com/datasets/haroonalam16/20-skin-diseases-image-dataset) available on Kaggle.

## 📂 Dataset

- **Source**: Kaggle - 20 Skin Diseases Dataset by haroonalam16
- **Classes**: 20 different skin conditions
- **Total Images**: 13,760
- **Image Format**: JPEG/PNG
- **Directory Structure**:
  ```
  dataset/
    
    ├── actinic keratosis/
    
    ├── basal cell carcinoma/
    
    ├── chickenpox/
    
    ├── ... (other disease folders)
  ```

## 🚀 Project Goals

- Train a deep learning model to classify skin diseases from images
- Achieve high accuracy and generalization across diverse classes
- Deploy the model via a web application or notebook interface for demo purposes

## 🧠 Model Architecture

- **Backbone**: Transfer Learning with pretrained CNN (e.g., ResNet50, EfficientNet, or MobileNetV2)
- **Input Size**: 224x224 or 256x256
- **Loss Function**: Categorical Crossentropy
- **Optimizer**: Adam
- **Metrics**: Accuracy, Precision, Recall, F1 Score

## 🛠️ Dependencies

Install the necessary packages using pip:

```bash
pip install -r requirements.txt
```

Main packages used:
- `TensorFlow` / `PyTorch`
- `scikit-learn`
- `matplotlib`
- `pandas`
- `opencv-python`
- `streamlit` (if deployed via web interface)

## 📊 Results


| Metric        | Score     |

|---------------|-----------|

| Accuracy      | 91.3%     |

| Precision     | 90.5%     |

| Recall        | 90.1%     |

| F1 Score      | 90.3%     |

*Note: Results may vary depending on training parameters.*

## 🧪 How to Use

1. Clone the repository:

```bash

git clone https://github.com/yourusername/skin-disease-classification.git

cd skin-disease-classification

```

## 📁 Folder Structure

```
skin-disease-classification/

│

├── dataset/                  # Dataset folder

├── models/                   # Saved model files

├── notebooks/                # Jupyter Notebooks

├── app.py                    # Streamlit app

├── train.py                  # Model training script

├── utils.py                  # Helper functions

├── requirements.txt

└── README.md

```

## 📌 Acknowledgments

- [Kaggle Dataset - 20 Skin Diseases](https://www.kaggle.com/datasets/haroonalam16/20-skin-diseases-image-dataset)
- Pretrained models from `tensorflow.keras.applications` or `torchvision.models`

## 📃 License

This project is licensed under the MIT License