"""
FoodAdvisor AI - Main Entry Point
This script provides the main interface for training and testing the food classification model.
"""

from model.train import train_model
from model.predict import test_model_with_images

# Default path for test images.
TEST_IMAGES_BASE_PATH = "../check_pictures/"

def main():
    """Main function to run FoodAdvisor AI."""
    print("=" * 50)
    print("FoodAdvisor AI - Food Classification System")
    print("=" * 50)

    # Train the model
    train_model()

    # Test the model with sample images.
    test_images()


def test_images():
    """Test the model with sample images from check_pictures directory."""
    print("\nTesting model with sample images...")

    # List of test images in check_pictures directory.
    test_images = [
        f"{TEST_IMAGES_BASE_PATH}pizza.jpg",
        f"{TEST_IMAGES_BASE_PATH}apple_pie.jpg",
        f"{TEST_IMAGES_BASE_PATH}baby_back_ribs.jpg",
        f"{TEST_IMAGES_BASE_PATH}hamburger.jpg",
        f"{TEST_IMAGES_BASE_PATH}sushi.jpg"
    ]

    # Test the model with these images.
    test_model_with_images(test_images)


if __name__ == "__main__":
    main()