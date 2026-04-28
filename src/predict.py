import os
import sys
import numpy as np
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.densenet import preprocess_input

# ==========================================
# CONFIGURATION
# ==========================================
IMG_SIZE = (224, 224)
CLASSES = ["glioma", "meningioma", "notumor", "pituitary"]
MODEL_PATH = "brain_tumor_detector.h5"

sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))
try:
    from utils import crop_brain_contour
except ImportError:
    # Fallback if utils not accessible
    def crop_brain_contour(img): return img

# Load the model strictly once when this file is imported (Good for UI/backend servers)
if os.path.exists(MODEL_PATH):
    print(f"[INFO] Loading production model from {MODEL_PATH}...")
    model = load_model(MODEL_PATH)
else:
    model = None
    print(f"[WARNING] Model file {MODEL_PATH} not found. Please run main.py once to generate it.")

# ==========================================
# GRAD-CAM LOGIC
# ==========================================
def get_gradcam_heatmap(img_array, target_class_idx=None, last_conv_layer_name="relu"):
    if model is None: return None
    
    grad_model = tf.keras.models.Model(
        [model.inputs], 
        [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if target_class_idx is not None:
            pred_index = target_class_idx
        else:
            pred_index = tf.argmax(preds[0])
            
        class_channel = preds[:, pred_index]

    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    
    return heatmap.numpy()

def generate_and_save_gradcam(img_path, heatmap, save_path=None, alpha=0.4, text=None, show=False):
    img = cv2.imread(img_path)
    if img is None: return None
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # DO NOT resize the base image down to 224! Keep it high resolution for the UI.
    # We only resize the generated heatmap to match the original High-Res Image.
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    superimposed_img = heatmap * alpha + img
    superimposed_img = np.clip(superimposed_img, 0, 255).astype(np.uint8)

    # Draw the prediction text directly onto the image
    if text:
        # Dynamically calculate professional font size based on the high-res image width
        font_scale = max(0.4, img.shape[1] / 800.0)
        thickness = max(1, int(font_scale * 1.5))
        font = cv2.FONT_HERSHEY_DUPLEX
        
        # Create a sleek, semi-transparent or tight black background box so the text is readable
        (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
        
        # Draw background rectangle
        padding = int(10 * font_scale)
        cv2.rectangle(superimposed_img, (10, 10), (10 + tw + padding*2, 10 + th + padding*2), (0, 0, 0), -1)
        
        # Draw white text
        cv2.putText(superimposed_img, text, (10 + padding, 10 + th + padding), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

    # Convert back to BGR for saving with OpenCV
    output_bgr = cv2.cvtColor(superimposed_img, cv2.COLOR_RGB2BGR)
    
    if save_path:
        cv2.imwrite(save_path, output_bgr)
        print(f"[INFO] Saved heatmap visualization to {save_path}")
        
    if show:
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.title("Original Image")
        plt.imshow(img)
        plt.axis("off")

        plt.subplot(1, 2, 2)
        plt.title(f"Prediction: {text}" if text else "Grad-CAM Overlay")
        plt.imshow(superimposed_img)
        plt.axis("off")
        
        plt.tight_layout()
        plt.show(block=True)
        
    return output_bgr

# ==========================================
# CORE PREDICTION FUNCTION FOR UI / DB
# ==========================================
def predict_image(img_path, output_heatmap_path=None, show_heatmap=False):
    """
    Use this function in your UI or Backend. 
    Pass the path of the uploaded image.
    It returns a dictionary with the results.
    """
    if model is None:
        return {"error": "Model not loaded"}

    if not os.path.exists(img_path):
        return {"error": f"Image file not found: {img_path}"}

    # Load and Preprocess for ResNet50
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, IMG_SIZE)
    
    # Apply Skull-Stripping Contour to match training pipeline
    img_cropped = crop_brain_contour(img)
    
    img_array = preprocess_input(img_cropped.astype(np.float32))
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    preds = model.predict(img_array, verbose=0)
    
    # Extract all probabilities
    probs = {CLASSES[i]: float(preds[0][i]) for i in range(len(CLASSES))}
    
    max_prob = float(np.max(preds))
    pred_class_idx = int(np.argmax(preds))
    pred_class_name = CLASSES[pred_class_idx]

    # Safety Logic Validation
    is_uncertain = max_prob < 0.80
    review_status = "Uncertain prediction – requires medical review" if is_uncertain else "Confident prediction"

    # Generate Heatmap
    # If it is uncertain and predicting notumor, let's force the heatmap to show the highest tumor class candidate
    if pred_class_name == "notumor" and is_uncertain:
        tumor_probs = {k: v for k, v in probs.items() if k != "notumor"}
        highest_tumor_class = max(tumor_probs.items(), key=lambda x: x[1])[0]
        highest_tumor_idx = CLASSES.index(highest_tumor_class)
        heatmap = get_gradcam_heatmap(img_array, target_class_idx=highest_tumor_idx)
        text_to_draw = f"{pred_class_name} ({max_prob*100:.1f}%) [UNCERTAIN]"
        text_to_draw += f" -> GradCAM: {highest_tumor_class}"
    else:
        heatmap = get_gradcam_heatmap(img_array)
        text_to_draw = f"{pred_class_name} ({max_prob*100:.1f}%)"
        if is_uncertain:
            text_to_draw += " [UNCERTAIN]"
    
    if output_heatmap_path or show_heatmap:
        generate_and_save_gradcam(img_path, heatmap, save_path=output_heatmap_path, text=text_to_draw, show=show_heatmap)

    result = {
        "predicted_class": pred_class_name,
        "confidence": max_prob,
        "all_probabilities": probs,
        "flag_requires_review": is_uncertain,
        "status_message": review_status,
        "heatmap_file": output_heatmap_path
    }
    
    return result

# ==========================================
# COMMAND LINE TESTER
# ==========================================
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict.py <path_to_image.jpg>")
    else:
        test_img = sys.argv[1]
        out_img = "output_heatmap.jpg"
        
        print("\nRunning Inference...")
        res = predict_image(test_img, output_heatmap_path=out_img, show_heatmap=True)
        
        print("\n--- RESULTS ---")
        for k, v in res.items():
            print(f"{k}: {v}")
