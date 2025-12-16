# 🍽️ FoodAdvisor AI

<div align="center">

![Python](https://img.shields.io/badge/Python-3.11.9-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.20.0-orange)
![Keras](https://img.shields.io/badge/Keras-3.12.0-red)
![License](https://img.shields.io/badge/License-MIT-green)

**An intelligent food classification system that can identify 101 different types of food from images**

[Dataset](https://www.kaggle.com/datasets/dansbecker/food-101) • [License](https://github.com/Jinery/foodadvisor-ai/blob/master/LICENSE)

</div>

## 📌 Overview

FoodAdvisor AI is a deep learning-based computer vision system that can automatically classify food images into 101 different categories. The system uses transfer learning with MobileNetV2 architecture and achieves high accuracy in food recognition tasks.

## ✨ Features

- **101 Food Categories**: Classify images into 101 different food types (pizza, sushi, burger, etc.)
- **High Accuracy**: State-of-the-art deep learning model with validation accuracy > 85%
- **Easy to Use**: Simple API for training and prediction
- **Visualizations**: Automatic generation of training graphs and results
- **Web-Ready**: Can process both local images and URLs

## 🚀 Quick Start

### Prerequisites
- Python 3.11.9
- 8GB+ RAM (16GB recommended)
- 10GB+ free disk space (for dataset)

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/Jinery/foodadvisor-ai.git
cd foodadvisor-ai
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Download and prepare the dataset**

Download Food-101 dataset from [Kaggle](https://www.kaggle.com/datasets/dansbecker/food-101)

After downloading, unpack all to root project folder.

Alternative Source: [Official Food 101 Source](https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/)

# 🎯 Usage
1. **Training the model**

```bash
# For only train use: 
python model/train.py

# Or for train and test:
python main.py
```

*During training, you'll see:*
* Real-time accuracy and loss metrics
* Best weights saved automatically
* Training graphs saved to visualizations/

2. **Making your custom Predictions**
 
For your custom predictions add to your code:
```python
from model.predict import predict_single_image

# Predict from local file
result = predict_single_image('pizza.jpg')
print(f"This looks like {result['prediction']} with {result['confidence']:.2f}% confidence")

# Predict from URL
result = predict_single_image('https://example.com/food.jpg')
```

# 📊 Model Architecture
```Text
Input (224x224x3)
     ↓
Data Augmentation
     ↓
MobileNetV2 (pretrained on ImageNet)
     ↓
Global Average Pooling
     ↓
Dense(1024) + BatchNorm + Dropout(0.5)
     ↓
Dense(512)
     ↓
Output: Dense(101) with Softmax
```
Training Details:

* Base Model: MobileNetV2
* Input Size: 224×224 pixels
* Optimizer: Adam
* Loss: Categorical Crossentropy
* Batch Size: 32
* Epochs: 20 (with early stopping)

# Sample Predictions
```text
🍕 pizza.jpg → "pizza" (98.7% confidence)
🍰 cake.jpg → "chocolate_cake" (92.3% confidence)
🍣 sushi.jpg → "sushi" (95.1% confidence)
```

# 🗂️ Dataset Information
Food-101 Dataset
* Total Images: 101,000
* Categories: 101 food types
* Images per Class: 1,000 (750 train / 250 test)
* Image Size: Variable (resized to 224x224)
* Format: JPEG

# 🚀 Roadmap
* Mobile app integration
* Calorie estimation feature
* Recipe suggestions
* Multi-language support
