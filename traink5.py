import os
import random
import shutil
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import DenseNet121  # ✅ Replacing ResNet-50
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

# ✅ Define dataset paths
BASE_DIR = "/kaggle/input/diabetic-retinopathy-balanced/content/Diabetic_Balanced_Data"
REDUCED_DIR = "/kaggle/working/reduced_dataset"
train_dir = os.path.join(REDUCED_DIR, "train")
test_dir = os.path.join(REDUCED_DIR, "test")

os.makedirs(REDUCED_DIR, exist_ok=True)

# ✅ Function to reduce dataset to 1000 images per class
def reduce_dataset(source_dir, target_dir, max_images=1000):
    os.makedirs(target_dir, exist_ok=True)
    
    for class_name in os.listdir(source_dir):
        class_path = os.path.join(source_dir, class_name)
        if os.path.isdir(class_path):
            images = os.listdir(class_path)
            sampled_images = random.sample(images, min(max_images, len(images)))  # 1000 or available
            
            new_class_path = os.path.join(target_dir, class_name)
            os.makedirs(new_class_path, exist_ok=True)
            
            for img in sampled_images:
                shutil.copy(os.path.join(class_path, img), os.path.join(new_class_path, img))

reduce_dataset(os.path.join(BASE_DIR, "train"), train_dir)
reduce_dataset(os.path.join(BASE_DIR, "test"), test_dir)

print("Dataset reduced to 1000 images per class!")

# ✅ Optimized Image Data Augmentation
train_datagen = ImageDataGenerator(
    rescale=1.0/255,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True
)

test_datagen = ImageDataGenerator(rescale=1.0/255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=64,
    class_mode='categorical'
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

# ✅ Compute Class Weights
train_classes = train_generator.classes
class_weights = compute_class_weight('balanced', classes=np.unique(train_classes), y=train_classes)
class_weights_dict = {i: w for i, w in enumerate(class_weights)}

print(f"Class Weights: {class_weights_dict}")

# ✅ Load Pre-trained DenseNet-121
base_model = DenseNet121(
    weights='imagenet',  
    include_top=False,
    input_shape=(224, 224, 3)
)

# ✅ Initially Freeze the Entire Model
base_model.trainable = False

# ✅ Build the model on top of DenseNet-121
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(5, activation='softmax')  # Assuming 5 classes
])

# ✅ Compile the model
model.compile(
    optimizer=Adam(learning_rate=0.0001),  
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# ✅ Train the Model (Initial Training)
history = model.fit(
    train_generator,
    epochs=10,  
    validation_data=test_generator,
    class_weight=class_weights_dict  
)

# ✅ Unfreeze Some Layers for Fine-tuning
base_model.trainable = True
for layer in base_model.layers[:100]:  # Keep first 100 layers frozen
    layer.trainable = False

# ✅ Recompile with Lower Learning Rate
model.compile(
    optimizer=Adam(learning_rate=0.00001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# ✅ Fine-tuning Phase
history_finetune = model.fit(
    train_generator,
    epochs=20,
    validation_data=test_generator,
    class_weight=class_weights_dict
)

# ✅ Save model
model.save("/kaggle/working/diabetic_retinopathy_densenet121_finetuned.h5")
print("DenseNet-121 model saved successfully!")

# ✅ Evaluate on test data
test_loss, test_acc = model.evaluate(test_generator)
print(f"Test Accuracy: {test_acc:.4f}")

# After training your model, save it
model.save("/kaggle/working/diabetic_retinopathy_densenet121_finetuned.h5")
print("Model saved successfully!")
