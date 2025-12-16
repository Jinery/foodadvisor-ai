"""
Utility functions for FoodAdvisor AI project.
"""

import os
import json
import numpy as np
from datetime import datetime


def create_directory_structure():
    """Create necessary directories for the project."""
    directories = [
        '../datasets',
        '../check_pictures',
        '../exit',
        '../visualizations',
        '../model'
    ]

    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"Created/verified directory: {directory}")


def save_class_names(class_names, filepath="../exit/class_names.json"):
    """Save class names to a JSON file."""
    with open(filepath, 'w') as f:
        json.dump(class_names, f, indent=2)
    print(f"Class names saved to {filepath}")


def load_class_names(filepath="../exit/class_names.json"):
    """Load class names from a JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)


def get_timestamp():
    """Get current timestamp for file naming."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def calculate_model_size(model_path):
    """Calculate and print model file size."""
    if os.path.exists(model_path):
        size_bytes = os.path.getsize(model_path)
        size_mb = size_bytes / (1024 * 1024)
        print(f"Model size: {size_mb:.2f} MB")
        return size_mb
    return 0


def print_project_info():
    """Print project information."""
    info = """
    ============================================
    FoodAdvisor AI - Project Information
    ============================================
    Description: Food image classification system
    Classes: 101 food categories(https://www.kaggle.com/datasets/dansbecker/food-101)
    Base Model: MobileNetV2
    Input Size: 224x224 pixels
    Output: Food class with confidence score
    ============================================
    """
    print(info)