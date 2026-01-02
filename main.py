import os
from io import BytesIO

import kagglehub
import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.keras.layers import (
    Conv2D,
    Dense,
    Dropout,
    Flatten,
    MaxPooling2D,
)
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import plot_model
from tqdm import tqdm

BATCH_SIZE = 32
IMAGE_SIZE = (224, 224)


def ela(image_path, scale=(224, 224), quality=90):
    """
    Performs Error Level Analysis (ELA) on an image and returns a 3-channel RGB result.

    Args:
        image_path (str): Path to the image file.
        scale (tuple): Resize dimensions (width, height).
        quality (int): JPEG quality for recompression.

    Returns:
        np.ndarray: 3-channel ELA image in RGB format (uint8).
    """
    # Load and resize image
    image = Image.open(image_path).convert("RGB")
    image = image.resize(scale)

    # Save recompressed image to memory (not disk)
    buffer = BytesIO()
    image.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)

    compressed = Image.open(buffer)

    # Compute ELA
    diff = np.abs(
        np.array(image, dtype=np.int16) - np.array(compressed, dtype=np.int16)
    )
    diff = np.clip(diff * 10, 0, 255).astype(np.uint8)

    return diff


def build_model(input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3)):
    model = Sequential(
        [
            Conv2D(32, (3, 3), activation="relu", input_shape=input_shape),
            MaxPooling2D((2, 2)),
            Conv2D(64, (3, 3), activation="relu"),
            MaxPooling2D((2, 2)),
            Conv2D(128, (3, 3), activation="relu"),
            MaxPooling2D((2, 2)),
            Flatten(),
            Dense(128, activation="relu"),
            Dropout(0.5),
            Dense(2, activation="softmax"),
        ]
    )
    optimizer = Adam(learning_rate=0.0001)
    model.compile(
        optimizer=optimizer,
        loss="categorical_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
        ],
    )
    return model


def main():
    path = kagglehub.dataset_download("xhlulu/140k-real-and-fake-faces")
    print("Path to dataset files:", path)

    base_path = os.path.join(path, "real_vs_fake", "real-vs-fake")

    ela_dir = os.path.join(base_path, "ela_images")
    os.makedirs(ela_dir, exist_ok=True)

    for stage in ["train", "valid", "test"]:
        stage_dir = os.path.join(base_path, stage)
        os.makedirs(stage_dir, exist_ok=True)

        for category in ["real", "fake"]:
            input_dir = os.path.join(stage_dir, category)
            output_dir = os.path.join(ela_dir, stage, category)

            # if os.path.exists(output_dir) and os.listdir(output_dir):
            #     print(f"Skipping {stage}/{category} (already processed)")
            #     continue

            os.makedirs(output_dir, exist_ok=True)

            for filename in tqdm(
                os.listdir(input_dir), desc=f"Processing {stage}/{category}"
            ):
                if filename.lower().endswith((".jpg", ".jpeg", ".png")):
                    input_path = os.path.join(input_dir, filename)
                    output_path = os.path.join(output_dir, filename)
                    ela_image = ela(input_path, IMAGE_SIZE)
                    Image.fromarray(ela_image).save(output_path)

    ela_dir = os.path.join(base_path, "ela_images")
    train_dir = os.path.join(ela_dir, "train")
    val_dir = os.path.join(ela_dir, "valid")
    test_dir = os.path.join(ela_dir, "test")

    datagen = ImageDataGenerator(
        rescale=1.0 / 255,
    )

    train_generator = datagen.flow_from_directory(
        train_dir,
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
    )

    val_generator = datagen.flow_from_directory(
        val_dir,
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
    )

    test_generator = datagen.flow_from_directory(
        test_dir,
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        shuffle=False,
    )

    model = build_model()
    model.summary()
    plot_model(
        model, to_file="model_architecture.png", show_shapes=True, show_layer_names=True
    )
    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=10,
    )


if __name__ == "__main__":
    main()
