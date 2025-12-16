"""
Training module for FoodAdvisor AI model.
Handles data loading, model architecture, training, and saving.
"""

import os
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import numpy

# Constants.
CHECKPOINT_PATH = "../exit/FoodAdvisor_AI_best_weights.weights.h5"
MODEL_SAVE_PATH = "../exit/FoodAdvisor_AI_model.keras"
FOOD_101_PATH = "../datasets/food-101/images"
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 20


def create_data_pipeline():
    """
    Create training and validation data pipelines.

    Returns:
        tuple: (train_dataset, validation_dataset, num_classes, class_names)
    """
    if not os.path.isdir(FOOD_101_PATH):
        raise FileNotFoundError(f"Dataset directory not found: {FOOD_101_PATH}")

    def preprocess(image, label):
        """Preprocess images: normalize to [0, 1] range."""
        image = tf.cast(image, tf.float32)
        image = image / 255.0
        return image, label

    # Load training dataset.
    train_dataset = tf.keras.utils.image_dataset_from_directory(
        FOOD_101_PATH,
        labels='inferred',
        label_mode='categorical',
        image_size=IMG_SIZE,
        interpolation='nearest',
        batch_size=BATCH_SIZE,
        subset='training',
        validation_split=0.2,
        seed=42
    )

    # Load validation dataset.
    validation_dataset = tf.keras.utils.image_dataset_from_directory(
        FOOD_101_PATH,
        labels='inferred',
        label_mode='categorical',
        image_size=IMG_SIZE,
        interpolation='nearest',
        batch_size=BATCH_SIZE,
        subset='validation',
        validation_split=0.2,
        seed=42
    )

    # Get class information.
    num_classes = len(train_dataset.class_names)
    class_names = train_dataset.class_names

    # Apply preprocessing and optimization.
    train_dataset = (
        train_dataset
        .map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
        .prefetch(buffer_size=tf.data.AUTOTUNE)
    )

    validation_dataset = (
        validation_dataset
        .map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
        .prefetch(buffer_size=tf.data.AUTOTUNE)
    )

    return train_dataset, validation_dataset, num_classes, class_names


def build_model(num_classes):
    """
    Build the FoodAdvisor AI model architecture.

    Args:
        num_classes (int): Number of food classes to predict

    Returns:
        tf.keras.Model: Compiled model
    """
    # Data augmentation layers.
    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.15),
        tf.keras.layers.RandomZoom(0.2),
        tf.keras.layers.RandomContrast(0.1),
    ])

    # Input layer.
    inputs = tf.keras.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))

    # Apply augmentation.
    x = data_augmentation(inputs)

    # Base model (MobileNetV2).
    base_model = tf.keras.applications.MobileNetV2(
        weights='imagenet',
        include_top=False,
        input_tensor=x,
    )
    base_model.trainable = False

    # Custom head.
    x = base_model.output
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(1024, activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(512, activation='relu')(x)
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax', name='prediction_output')(x)

    # Create model.
    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    # Compile model.
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


def plot_training_history(history, save_path="../visualizations/training_history.png"):
    """
    Plot and save training history graphs.

    Args:
        history: Keras History object from model.fit()
        save_path (str): Path to save the visualization
    """
    # Create figure with subplots.
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Plot accuracy.
    ax1.plot(history.history['accuracy'], label='Training Accuracy')
    ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax1.set_title('Model Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot loss.
    ax2.plot(history.history['loss'], label='Training Loss')
    ax2.plot(history.history['val_loss'], label='Validation Loss')
    ax2.set_title('Model Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Adjust layout and save.
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Training history saved to {save_path}")
    plt.show()


def save_training_artifacts(history, model, class_names, save_dir="../visualizations"):
    """
    Save all training artifacts: history, model summary, class names.

    Args:
        history: Keras History object
        model: Trained model
        class_names: List of class names
        save_dir: Directory to save artifacts
    """
    import pickle
    import json
    from datetime import datetime
    from pathlib import Path

    # Create dir.
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 1. Save train history.
    history_data = {
        'history': history.history,
        'timestamp': timestamp,
        'epochs_completed': len(history.history['accuracy'])
    }

    with open(save_dir / f"training_history_{timestamp}.pkl", "wb") as f:
        pickle.dump(history_data, f)

    # 2. Save history to JSON.
    json_history = {}
    for key, values in history.history.items():
        json_history[key] = [float(v) for v in values]  # Convert numpy to float.

    with open(save_dir / f"training_history_{timestamp}.json", "w") as f:
        json.dump(json_history, f, indent=2)

    # 3. Save model metadata.
    metadata = {
        'class_names': class_names,
        'num_classes': len(class_names),
        'input_shape': model.input_shape,
        'output_shape': model.output_shape,
        'total_params': model.count_params(),
        'timestamp': timestamp
    }

    with open(save_dir / f"model_metadata_{timestamp}.json", "w") as f:
        json.dump(metadata, f, indent=2)

    # 4. Save model summary for txt file.
    with open(save_dir / f"model_summary_{timestamp}.txt", "w") as f:
        model.summary(print_fn=lambda x: f.write(x + '\n'))

    print(f"\nAll training artifacts saved to {save_dir}/")
    print(f"  - training_history_{timestamp}.pkl (pickle)")
    print(f"  - training_history_{timestamp}.json (JSON)")
    print(f"  - model_metadata_{timestamp}.json")
    print(f"  - model_summary_{timestamp}.txt")

    return str(save_dir)

def train_model():
    """
    Main training function for FoodAdvisor AI model.
    Handles the complete training pipeline.
    """
    print("=" * 50)
    print("FoodAdvisor AI - Training Pipeline")
    print("=" * 50)

    # Step 1: Create data pipeline.
    print("\n[1/6] Loading and preprocessing data...")
    train_dataset, validation_dataset, num_classes, class_names = create_data_pipeline()
    print(f"   Detected {num_classes} food classes")
    print(f"   Training samples: {len(train_dataset) * BATCH_SIZE}")
    print(f"   Validation samples: {len(validation_dataset) * BATCH_SIZE}")

    # Step 2: Build model.
    print("\n[2/6] Building model architecture...")
    model = build_model(num_classes)
    model.summary()

    # Step 3: Setup callbacks.
    print("\n[3/6] Setting up training callbacks...")
    checkpoint = ModelCheckpoint(
        CHECKPOINT_PATH,
        monitor='val_accuracy',
        verbose=1,
        save_best_only=True,
        mode='max'
    )

    early_stopping = EarlyStopping(
        monitor='val_accuracy',
        patience=3,
        restore_best_weights=True
    )

    callbacks_list = [checkpoint, early_stopping]

    # Step 4: Train model.
    print(f"\n[4/6] Training model for {EPOCHS} epochs...")
    print("-" * 40)

    history = model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=EPOCHS,
        callbacks=callbacks_list,
        verbose=1
    )

    # Step 5: Save model and plot results.
    print("\n[5/6] Saving model and generating visualizations...")
    model.save(MODEL_SAVE_PATH)
    print(f"   Model saved to: {MODEL_SAVE_PATH}")

    # Plot training history.
    plot_training_history(history)

    # Save model history.
    print("\n[6/6] Saving training artifacts...")
    artifacts_dir = save_training_artifacts(history, model, class_names)
    print(f"   Artifacts saved to: {artifacts_dir}")

    # Final summary.
    print("\n" + "=" * 50)
    print("Training completed successfully!")
    print(f"Final training accuracy: {history.history['accuracy'][-1]:.4f}")
    print(f"Final validation accuracy: {history.history['val_accuracy'][-1]:.4f}")
    print("=" * 50)


if __name__ == "__main__":
    train_model()

