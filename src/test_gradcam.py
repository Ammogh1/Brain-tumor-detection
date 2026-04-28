import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
import glob

# ==========================================
# CONFIGURATION
# ==========================================
DATASET_DIR = r"D:\DEEP LEARNING\Project Documents\dataset"
TEST_DIR = os.path.join(DATASET_DIR, "Testing")
IMG_SIZE = (224, 224)
CLASSES = ["glioma", "meningioma", "notumor", "pituitary"]

# ==========================================
# GRAD-CAM LOGIC
# ==========================================
def get_gradcam_heatmap(img_array, model, last_conv_layer_name="conv5_block3_out"):
    grad_model = tf.keras.models.Model(
        [model.inputs], 
        [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    
    return heatmap.numpy()

def display_gradcam(img_path, heatmap, alpha=0.4):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, IMG_SIZE)

    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    
    # FIXED: Convert OpenCV heatmap (BGR) to RGB format so colors map correctly
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

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
    plt.show()

# ==========================================
# EXECUTION
# ==========================================
def main():
    model_path = "brain_tumor_detector.h5"
    if not os.path.exists(model_path):
        print(f"ERROR: {model_path} not found!")
        print("You must let main.py finish training at least once to save a model.")
        sys.exit(1)

    print("Loading saved model...")
    model = load_model(model_path)
    
    # Grab the same first test image (likely the glioma sample from your screenshot)
    test_images = glob.glob(os.path.join(TEST_DIR, "*", "*.jpg"))
    if not test_images:
        print(f"ERROR: No .jpg images found in {TEST_DIR}.")
        sys.exit(1)
        
    sample_img_path = test_images[0]
    
    print(f"Running inference on: {sample_img_path}")
    img = cv2.imread(sample_img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, IMG_SIZE)
    
    # IMPORTANT: Use the old 1/255 scale logic since we are loading the old model
    img_array = img / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    preds = model.predict(img_array, verbose=0)
    max_prob = np.max(preds)
    pred_class_idx = np.argmax(preds)
    pred_class_name = CLASSES[pred_class_idx]

    print("\n" + "="*40)
    print("INFERENCE RESULT:")
    print(f"Predicted Class: {pred_class_name}")
    print(f"Confidence Score: {max_prob * 100:.2f}%")
    print("="*40)
    
    heatmap = get_gradcam_heatmap(img_array, model, last_conv_layer_name="conv5_block3_out")
    display_gradcam(sample_img_path, heatmap)

if __name__ == "__main__":
    main()
