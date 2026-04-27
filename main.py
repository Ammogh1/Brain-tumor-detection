import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.applications.densenet import preprocess_input
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.utils.class_weight import compute_class_weight
import seaborn as sns

# ==========================================
# CONFIGURATION
# ==========================================
# User enforced correct dataset location
DATASET_DIR = r"D:\DEEP LEARNING\Project Documents\dataset"
TRAIN_DIR = os.path.join(DATASET_DIR, "Training")
TEST_DIR = os.path.join(DATASET_DIR, "Testing")

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
CLASSES = ["glioma", "meningioma", "notumor", "pituitary"]

# ==========================================
# 1. ERROR HANDLING & VALIDATION
# ==========================================
def validate_dataset():
    if not os.path.exists(DATASET_DIR):
        print(f"ERROR: Dataset path not found: {DATASET_DIR}")
        sys.exit(1)
        
    for subset_dir in [TRAIN_DIR, TEST_DIR]:
        if not os.path.exists(subset_dir):
            print(f"ERROR: '{subset_dir}' folder is missing!")
            sys.exit(1)
            
        for cls in CLASSES:
            class_path = os.path.join(subset_dir, cls)
            if not os.path.isdir(class_path):
                print(f"ERROR: Missing class folder '{class_path}'. Script stopped to avoid errors.")
                sys.exit(1)
    
    print("Dataset structure validation passed.")

# ==========================================
# 2. DATA PREPROCESSING
# ==========================================
def crop_brain_contour_pipeline(img):
    """
    OpenCV Interceptor for Keras. Strips skull and margins out blindly!
    """
    img_uint8 = np.uint8(np.clip(img, 0, 255))
    gray = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(gray, 45, 255, cv2.THRESH_BINARY)
    thresh = cv2.erode(thresh, None, iterations=2)
    thresh = cv2.dilate(thresh, None, iterations=2)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        c = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(c)
        cropped_img = img_uint8[y:y+h, x:x+w]
        cropped_img = cv2.resize(cropped_img, IMG_SIZE)
    else:
        cropped_img = img_uint8
        
    return preprocess_input(cropped_img.astype(np.float32))

def create_data_generators():
    # Training generator with Augmentation and 20% validation split
    train_datagen = ImageDataGenerator(
        preprocessing_function=crop_brain_contour_pipeline,
        rotation_range=15,
        horizontal_flip=True,
        zoom_range=0.1,
        brightness_range=[0.8, 1.2],
        validation_split=0.2
    )

    # Test generator (only proper ResNet preprocessing!)
    test_datagen = ImageDataGenerator(preprocessing_function=crop_brain_contour_pipeline)

    print("Loading Training Data...")
    train_generator = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        classes=CLASSES,
        subset='training'
    )

    print("Loading Validation Data...")
    val_generator = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        classes=CLASSES,
        subset='validation'
    )

    print("Loading Testing Data...")
    test_generator = test_datagen.flow_from_directory(
        TEST_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        classes=CLASSES,
        shuffle=False
    )
    
    return train_generator, val_generator, test_generator

# ==========================================
# 3. MODEL ARCHITECTURE
# ==========================================
def build_model():
    base_model = DenseNet121(include_top=False, weights='imagenet', input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
    
    # Freeze base layers initially
    base_model.trainable = False
    
    x = GlobalAveragePooling2D()(base_model.output)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    outputs = Dense(4, activation='softmax')(x)
    
    model = Model(inputs=base_model.input, outputs=outputs)
    return model, base_model

# ==========================================
# 4. GRAD-CAM IMPLEMENTATION
# ==========================================
def get_gradcam_heatmap(img_array, model, last_conv_layer_name="relu"):
    # First, we create a model that maps the input image to the activations
    # of the last conv layer as well as the output predictions
    grad_model = tf.keras.models.Model(
        [model.inputs], 
        [model.get_layer(last_conv_layer_name).output, model.output]
    )

    # Compute the gradient of the top predicted class for the input image
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    # Gradient of the output neuron with respect to the output feature map
    grads = tape.gradient(class_channel, last_conv_layer_output)

    # Pool gradients over spatial dimensions
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Multiply each channel by its weight (pooled gradient)
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # Normalize heatmap
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def display_gradcam(img_path, heatmap, alpha=0.4):
    # Load original image
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, IMG_SIZE)

    # Resize heatmap to match image size
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    
    # Convert heatmap to RGB format (OpenCV returns BGR by default)
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    # Overlay
    superimposed_img = heatmap * alpha + img
    superimposed_img = np.clip(superimposed_img, 0, 255).astype(np.uint8)

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow(img)
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.title("Grad-CAM Overlay")
    plt.imshow(superimposed_img)
    plt.axis("off")
    
    plt.tight_layout()
    plt.savefig('gradcam_demo.png')
    plt.close()

# ==========================================
# 5. PREDICTION SAFETY LOGIC
# ==========================================
def predict_with_safety(img_path, model):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, IMG_SIZE)
    img_array = crop_brain_contour_pipeline(img)
    img_array = np.expand_dims(img_array, axis=0)

    preds = model.predict(img_array, verbose=0)
    max_prob = np.max(preds)
    pred_class_idx = np.argmax(preds)
    pred_class_name = CLASSES[pred_class_idx]

    print("\n" + "="*40)
    print("INFERENCE RESULT:")
    print(f"Predicted Class: {pred_class_name}")
    print(f"Confidence Score: {max_prob * 100:.2f}%")
    
    if max_prob < 0.80:
        print("ALERT: Uncertain prediction – requires medical review.")
        print("Prediction confidence is too low to be fully trusted.")
    else:
        print("Prediction is based on highlighted abnormal region in the MRI scan.")
    print("="*40)
    
    # Grad-CAM logic
    heatmap = get_gradcam_heatmap(img_array, model, last_conv_layer_name="relu")
    display_gradcam(img_path, heatmap)
    
    return pred_class_name, max_prob

# ==========================================
# 6. PIPELINE EXECUTION
# ==========================================
def main():
    # Validation step
    validate_dataset()
    
    # Get Generators
    train_gen, val_gen, test_gen = create_data_generators()
    
    # Build Model
    model, base_model = build_model()
    model.compile(
        optimizer=Adam(learning_rate=0.0001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6),
        ModelCheckpoint('brain_tumor_detector.h5', monitor='val_loss', save_best_only=True, mode='min', verbose=1)
    ]
    
    # Compute class weights to handle imbalances
    class_weights = compute_class_weight(
        'balanced',
        classes=np.unique(train_gen.classes),
        y=train_gen.classes
    )
    class_weights_dict = dict(enumerate(class_weights))
    print(f"Computed Class Weights: {class_weights_dict}")

    print("\nStarting Initial Phase Training (Base Model Frozen)...")
    # Set to a small number of epochs for demonstration.
    # User prompt stated 25-50 total
    epochs_phase1 = 15
    model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=epochs_phase1,
        callbacks=callbacks,
        class_weight=class_weights_dict
    )
    
    print("\nStarting Fine-Tuning Phase (Unfreezing top layers)...")
    # Unfreeze top layers of DenseNet121
    base_model.trainable = True
    for layer in base_model.layers[:-100]: # Freeze bottom layers, unfreeze top 100 layers for detailed fine-tuning
        layer.trainable = False
        
    # Recompile with smaller learning rate for fine-tuning
    model.compile(
        optimizer=Adam(learning_rate=1e-5),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    epochs_phase2 = 25
    model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=epochs_phase2,
        callbacks=callbacks,
        class_weight=class_weights_dict
    )
    
    # Evaluation
    print("\nEvaluating Model on Test Data...")
    loss, accuracy = model.evaluate(test_gen)
    print(f"Test Accuracy: {accuracy * 100:.2f}%")
    
    # Confusion Matrix & Classification Report
    predictions = model.predict(test_gen)
    y_pred = np.argmax(predictions, axis=1)
    y_true = test_gen.classes
    
    cm = confusion_matrix(y_true, y_pred)
    cr = classification_report(y_true, y_pred, target_names=CLASSES)
    
    print("\nClassification Report:\n", cr)
    
    # Plot Confusion Matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=CLASSES, yticklabels=CLASSES)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('confusion_matrix.png')
    plt.close()

    # Save final model explicitly just in case EarlyStopping wasn't triggered
    model_save_path = "brain_tumor_detector.h5"
    model.save(model_save_path)
    print(f"\nFinal model saved successfully to {model_save_path}")

    # Demo Inference on a sample image (if test dataset has samples)
    try:
        sample_batch, _ = next(test_gen)
        sample_img_path = test_gen.filepaths[0]
        predict_with_safety(sample_img_path, model)
    except Exception as e:
        print("Could not run demo inference:", e)

if __name__ == "__main__":
    main()
