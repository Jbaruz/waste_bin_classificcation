import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from tensorflow.keras.preprocessing import image
from tkinter import filedialog, Tk
import tensorflow as tf
import wandb
from wandb.integration.keras import WandbCallback
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import plot_model
import matplotlib.pyplot as plt
from sklearn.utils.class_weight import compute_class_weight
import tempfile
# pip install -r requirements.txt
# Print to verify WandB installation
print("WandB is installed correctly.")

# Log in to WandB using the API key from the environment variable
api_key = os.getenv("WANDB_API_KEY")
if api_key:
    wandb.login(key=api_key)
else:
    raise ValueError("WANDB_API_KEY environment variable not set")

# Define the sweep configuration for hyperparameter tuning
sweep_config = {
    'method': 'random',  # or 'grid', 'bayes'
    'metric': {
        'name': 'val_accuracy',
        'goal': 'maximize'
    },
    'parameters': {
        'learning_rate': {
            'values': [0.001, 0.0001, 0.00001]
        },
        'batch_size': {
            'values': [16, 32, 64]
        },
        'epochs': {
            'values': [50, 100, 150]
        },
        'dropout': {
            'values': [0.3, 0.5, 0.7]
        }
    }
}

# Initialize the sweep
sweep_id = wandb.sweep(sweep_config, project="waste_bin_prediction")

# Function to resize images to a consistent size
def resize_images(directory_path, desired_size=(224, 224)):
    """
    Resize all images in the specified directory to the desired size.
    """
    for filename in os.listdir(directory_path):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            img_path = os.path.join(directory_path, filename)
            img = Image.open(img_path)
            img = img.resize(desired_size, Image.Resampling.LANCZOS)
            img.save(img_path)

# Resize images in the specified directories
resize_images("ajax/full")
resize_images("ajax/empty")

# Function to rename image files in a directory
def rename_files(directory_path, base_filename):
    """
    Rename image files in the specified directory with a base filename.
    """
    counter = 1
    for filename in os.listdir(directory_path):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            new_filename = f"{base_filename}_{counter}.jpg"
            source = os.path.join(directory_path, filename)
            destination = os.path.join(directory_path, new_filename)

            # Check if the destination file already exists
            if not os.path.exists(destination):
                try:
                    os.rename(source, destination)
                    counter += 1
                except PermissionError:
                    print(f"Could not rename file: {source}. It might be in use by another process.")
            else:
                print(f"Cannot rename {source} to {destination}: destination file already exists.")
                counter += 1  # Increment the counter to avoid retrying with the same file name

# Rename files in the specified directories
rename_files("ajax/full", "full")
rename_files("ajax/empty", "empty")

# Define the path to your dataset
dataset_dir = 'ajax'

# Function to check and print the sizes of images in a directory
def check_image_sizes(directory_path):
    """
    Check and print the sizes of images in the specified directory.
    """
    sizes = []
    for filename in os.listdir(directory_path):
        if filename.endswith((".jpg", ".png")):  # Check for jpg and png files
            img_path = os.path.join(directory_path, filename)
            with Image.open(img_path) as img:
                print(f"{filename}: {img.size}")  # Print filename and its size
                sizes.append(img.size)  # Add the size to the list
    return sizes

# Check the sizes of images in the specified directories
sizes_full = check_image_sizes("ajax/full")
sizes_empty = check_image_sizes("ajax/empty")

# Set parameters for image processing
img_height, img_width = 150, 150  # Resize all images to 150x150

# Set a custom temporary directory for temporary files
tempfile.tempdir = r'C:\Users\Jbaru\OneDrive\Documents\temp'

# Ensure the custom temporary directory exists
if not os.path.exists(tempfile.tempdir):
    os.makedirs(tempfile.tempdir)

# Define a global variable for the model
model = None

def train():
    """
    Train the CNN model with the specified configuration.
    """
    global model

    # Initialize WandB run
    wandb.init(config=sweep_config, project="waste_bin_prediction")

    # Retrieve hyperparameters from WandB config
    config = wandb.config

    # Data augmentation
    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        rotation_range=45,
        width_shift_range=0.3,
        height_shift_range=0.3,
        shear_range=0.3,
        zoom_range=0.3,
        horizontal_flip=True,
        fill_mode='reflect',
        brightness_range=[0.5, 1.5],
        channel_shift_range=150.0,
        validation_split=0.1
    )

    # Load and preprocess the training data
    train_generator = train_datagen.flow_from_directory(
        dataset_dir,
        target_size=(img_height, img_width),
        batch_size=config.batch_size,
        class_mode='binary',
        subset='training'
    )

    # Load and preprocess the validation data
    validation_generator = train_datagen.flow_from_directory(
        dataset_dir,
        target_size=(img_height, img_width),
        batch_size=config.batch_size,
        class_mode='binary',
        subset='validation'
    )

    # Define the CNN model
    model = Sequential([
        # First convolutional block
        Conv2D(32, (3, 3), activation='relu', input_shape=(img_height, img_width, 3)),
        BatchNormalization(),
        MaxPooling2D(2, 2),

        # Second convolutional block
        Conv2D(64, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(2, 2),

        # Third convolutional block
        Conv2D(128, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(2, 2),

        # Fourth convolutional block
        Conv2D(256, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(2, 2),

        # Flattening and fully connected layers
        Flatten(),
        Dropout(0.5),
        Dense(512, activation='relu'),
        Dense(1, activation='sigmoid')  # Use 'softmax' if more than 2 classes
    ])

    # Compile the model
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=config.learning_rate), loss='binary_crossentropy',
                  metrics=['accuracy'])

    # Model Summary
    model.summary()

    # Generate a plot of the model
    plot_model(model,
               to_file='model.png',
               show_shapes=True,
               show_layer_names=True)

    # Callbacks
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1, mode='min')
    model_checkpoint = ModelCheckpoint('best_model.h5', monitor='val_loss', verbose=1, save_best_only=True, mode='min')

    # Calculate class weights
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(train_generator.classes),
        y=train_generator.classes)

    class_weights_dict = {i: weight for i, weight in enumerate(class_weights)}

    # Train the Model
    history = model.fit(
        train_generator,
        epochs=config.epochs,  # Adjust based on how the model performs
        validation_data=validation_generator,
        class_weight=class_weights_dict,  # Use the class weights here
        callbacks=[early_stopping, model_checkpoint,
                   WandbCallback(log_weights=True, log_evaluation=True, log_best_prefix='best',
                                 training_data=train_generator, validation_data=validation_generator,
                                 log_batch_frequency=10, log_epoch_frequency=1, log_graph=False)]
    )

    # Log metrics to WandB during training
    num_epochs_trained = len(history.history['accuracy'])

    for epoch in range(num_epochs_trained):
        wandb.log({
            'epoch': epoch + 1,
            'accuracy': history.history['accuracy'][epoch],
            'val_accuracy': history.history['val_accuracy'][epoch],
            'loss': history.history['loss'][epoch],
            'val_loss': history.history['val_loss'][epoch]
        })

    # Plot training & validation accuracy and loss
    plt.figure(figsize=(10, 5))

    # Accuracy plot
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend(loc='lower right')

    # Loss plot
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend(loc='upper right')

    # Display plots
    plt.tight_layout()
    plt.show()

# Define the sweep agent to run multiple hyperparameter configurations
wandb.agent(sweep_id, function=train, count=7)  # Adjust the count based on the number of runs desired

# Function to make a prediction on a given image
def make_prediction(image_path):
    """
    Make a prediction on the given image path.
    """
    global model

    if model is None:
        # Load the best saved model if it hasn't been loaded yet
        model = Sequential([
            # First convolutional block
            Conv2D(32, (3, 3), activation='relu', input_shape=(img_height, img_width, 3)),
            BatchNormalization(),
            MaxPooling2D(2, 2),

            # Second convolutional block
            Conv2D(64, (3, 3), activation='relu'),
            BatchNormalization(),
            MaxPooling2D(2, 2),

            # Third convolutional block
            Conv2D(128, (3, 3), activation='relu'),
            BatchNormalization(),
            MaxPooling2D(2, 2),

            # Fourth convolutional block
            Conv2D(256, (3, 3), activation='relu'),
            BatchNormalization(),
            MaxPooling2D(2, 2),

            # Flattening and fully connected layers
            Flatten(),
            Dropout(0.5),
            Dense(512, activation='relu'),
            Dense(1, activation='sigmoid')  # Use 'softmax' if more than 2 classes
        ])

        # Compile the model
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='binary_crossentropy',
                      metrics=['accuracy'])

        # Load the saved weights
        model.load_weights('best_model.h5')

    try:
        img = image.load_img(image_path, target_size=(img_height, img_width))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0
        prediction = model.predict(img_array)

        # Determine the prediction label and confidence
        if prediction[0][0] > 0.5:
            label = "empty"
            confidence = prediction[0][0]
        else:
            label = "full"
            confidence = 1 - prediction[0][0]

        # Load the image again for drawing
        img = Image.open(image_path)
        draw = ImageDraw.Draw(img)
        font = ImageFont.load_default()
        text = f"{label} ({confidence:.2%})"

        # Use ImageFont to get the text size
        text_size = draw.textbbox((0, 0), text, font=font)[2:]  # Get the width and height of the text

        # Draw text on the image
        draw.text((10, 10), text, fill="red", font=font)

        # Display the image
        img.show()

        return label, confidence
    except Exception as e:
        print(f"Error making prediction: {e}")
        return None, None

# Verify the temporary directory setup
print(f"Custom temporary directory: {tempfile.gettempdir()}")

# Example prediction usage
root = Tk()
root.withdraw()  # Hide the main window
image_path = filedialog.askopenfilename(title='Select an Image to Predict')
if image_path:
    result, confidence = make_prediction(image_path)
    if result:
        print(f"Prediction: {result} with confidence: {confidence:.2%}")
    else:
        print("Failed to make a prediction.")
else:
    print("No image selected.")
