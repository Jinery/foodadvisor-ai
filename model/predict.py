"""
Prediction module for FoodAdvisor AI.
Handles loading trained model and making predictions on new images.
"""

import os
import numpy as np
import tensorflow as tf
from io import BytesIO
import requests
from PIL import Image
import matplotlib.pyplot as plt

# Constants.
MODEL_PATH = "../exit/FoodAdvisor_AI_model.keras"
IMG_SIZE = (224, 224)

# Food class names (101 classes).
CLASS_NAMES = [
    'apple_pie', 'baby_back_ribs', 'baklava', 'beef_carpaccio', 'beef_tartare',
    'beet_salad', 'beignets', 'bibimbap', 'bread_pudding', 'breakfast_burrito',
    'bruschetta', 'caesar_salad', 'cannoli', 'caprese_salad', 'carrot_cake',
    'ceviche', 'cheese_plate', 'cheesecake', 'chicken_curry', 'chicken_quesadilla',
    'chicken_wings', 'chocolate_cake', 'chocolate_mousse', 'churros', 'clam_chowder',
    'club_sandwich', 'crab_cakes', 'creme_brulee', 'croque_madame', 'cup_cakes',
    'deviled_eggs', 'donuts', 'dumplings', 'edamame', 'eggs_benedict',
    'escargots', 'falafel', 'filet_mignon', 'fish_and_chips', 'foie_gras',
    'french_fries', 'french_onion_soup', 'french_toast', 'fried_calamari',
    'fried_rice', 'frozen_yogurt', 'garlic_bread', 'gnocchi', 'greek_salad',
    'grilled_cheese_sandwich', 'grilled_salmon', 'guacamole', 'gyoza', 'hamburger',
    'hot_and_sour_soup', 'hot_dog', 'huevos_rancheros', 'hummus', 'ice_cream',
    'lasagna', 'lobster_bisque', 'lobster_roll_sandwich', 'macaroni_and_cheese',
    'macarons', 'miso_soup', 'mussels', 'nachos', 'omelette', 'onion_rings',
    'oysters', 'pad_thai', 'paella', 'pancakes', 'panna_cotta', 'peking_duck',
    'pho', 'pizza', 'pork_chop', 'poutine', 'prime_rib', 'pulled_pork_sandwich',
    'ramen', 'ravioli', 'red_velvet_cake', 'risotto', 'samosa', 'sashimi',
    'scallops', 'seaweed_salad', 'shrimp_and_grits', 'spaghetti_bolognese',
    'spaghetti_carbonara', 'spring_rolls', 'steak', 'strawberry_shortcake',
    'sushi', 'tacos', 'takoyaki', 'tiramisu', 'tuna_tartare', 'waffles'
]


def load_model():
    """
    Load the trained FoodAdvisor AI model.

    Returns:
        tf.keras.Model: Loaded model

    Raises:
        Exception: If model loading fails
    """
    try:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

        model = tf.keras.models.load_model(MODEL_PATH)
        print(f"Model loaded successfully from {MODEL_PATH}")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        raise


def preprocess_image(image_path):
    """
    Preprocess an image for model prediction.

    Args:
        image_path (str): Path or URL to the image

    Returns:
        np.ndarray: Preprocessed image array

    Raises:
        FileNotFoundError: If image file doesn't exist
        Exception: For other processing errors
    """
    try:
        # Load image from URL or local path.
        if image_path.startswith(('http://', 'https://')):
            response = requests.get(image_path)
            img = Image.open(BytesIO(response.content))
        else:
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image not found: {image_path}")
            img = Image.open(image_path)

        # Convert to RGB if needed.
        if img.mode != 'RGB':
            img = img.convert('RGB')

        # Resize and preprocess.
        img = img.resize(IMG_SIZE)
        img_array = np.array(img)
        img_array = img_array / 255.0  # Normalize
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension.

        return img_array
    except Exception as e:
        raise Exception(f"Error preprocessing image: {e}")


def predict_food(model, processed_image):
    """
    Make prediction on a preprocessed image.

    Args:
        model (tf.keras.Model): Loaded model
        processed_image (np.ndarray): Preprocessed image array

    Returns:
        tuple: (predicted_class, confidence, all_predictions)
    """
    predictions = model.predict(processed_image, verbose=0)
    predicted_index = np.argmax(predictions[0])
    predicted_class = CLASS_NAMES[predicted_index]
    confidence = predictions[0][predicted_index] * 100

    # Get top 3 predictions.
    top_indices = np.argsort(predictions[0])[-3:][::-1]
    top_predictions = [
        (CLASS_NAMES[i], predictions[0][i] * 100)
        for i in top_indices
    ]

    return predicted_class, confidence, top_predictions


def display_prediction(image_path, predicted_class, confidence, top_predictions):
    """
    Display prediction results in a formatted way.

    Args:
        image_path (str): Path to the image
        predicted_class (str): Predicted food class
        confidence (float): Prediction confidence
        top_predictions (list): Top 3 predictions
    """
    print("\n" + "=" * 50)
    print(f"📷 Image: {os.path.basename(image_path)}")
    print("-" * 50)
    print(f"🍽️  Predicted Food: {predicted_class}")
    print(f"📊 Confidence: {confidence:.2f}%")
    print("-" * 50)
    print("Top 3 Predictions:")
    for i, (class_name, conf) in enumerate(top_predictions, 1):
        print(f"  {i}. {class_name}: {conf:.2f}%")
    print("=" * 50)


def test_model_with_images(image_paths):
    """
    Test the model with a list of image paths.

    Args:
        image_paths (list): List of image paths to test
    """
    print("🧪 Testing FoodAdvisor AI Model")
    print("-" * 40)

    # Load model.
    try:
        model = load_model()
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    # Test each image.
    for img_path in image_paths:
        try:
            # Preprocess and predict
            processed_image = preprocess_image(img_path)
            predicted_class, confidence, top_predictions = predict_food(
                model, processed_image
            )

            # Display results.
            display_prediction(img_path, predicted_class, confidence, top_predictions)

        except FileNotFoundError as e:
            print(f"Skipping: {e}")
            continue
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            continue

    print("\nTesting completed")


def predict_single_image(image_path):
    """
    Predict a single image and return detailed results.

    Args:
        image_path (str): Path to the image

    Returns:
        dict: Prediction results
    """
    try:
        model = load_model()
        processed_image = preprocess_image(image_path)
        predicted_class, confidence, top_predictions = predict_food(
            model, processed_image
        )

        return {
            'success': True,
            'image': os.path.basename(image_path),
            'prediction': predicted_class,
            'confidence': confidence,
            'top_predictions': top_predictions
        }
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }


if __name__ == '__main__':
    # Example usage.
    test_images = [
        "../check_pictures/pizza.jpg",
        "../check_pictures/apple_pie.jpg",
        "../check_pictures/baby_back_ribs.jpg"
    ]

    test_model_with_images(test_images)