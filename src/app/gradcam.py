import tensorflow as tf
import numpy as np
import cv2
from scipy.ndimage import gaussian_filter
from model import get_model

CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]


def get_gradcam_heatmap(
    img_array,
    last_conv_layer_name: str = "conv5_block16_concat",
) -> tuple[np.ndarray, int, float, np.ndarray]:

    model = get_model()

    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[model.get_layer(last_conv_layer_name).output, model.output],
    )

    with tf.GradientTape() as tape:
        last_conv_output, preds = grad_model(img_array)
        preds = tf.cast(preds, tf.float32)
        pred_class = int(tf.argmax(preds[0]))
        class_score = tf.reduce_sum(preds[:, pred_class])

    grads = tape.gradient(class_score, last_conv_output)
    grads = tf.nn.relu(grads)

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_out = last_conv_output[0]
    heatmap = conv_out @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.nn.relu(heatmap).numpy().astype(np.float32)

    if heatmap.max() > 0:
        heatmap /= heatmap.max()

    # The DenseNet model already outputs Softmax probabilities (see model.py output layer)
    # Applying softmax again flattens the confidence. We should just use the raw output.
    probs = preds[0].numpy()
    confidence = float(probs[pred_class])

    return heatmap, pred_class, confidence, probs


def extract_brain_mask(gray_img: np.ndarray) -> np.ndarray:
    """
    Extracts a binary mask of the brain region only.
    Excludes skull edges and black background so heatmap
    doesn't bleed outside the brain.
    """
    # Threshold to remove pure black background
    _, brain_thresh = cv2.threshold(gray_img, 15, 255, cv2.THRESH_BINARY)

    # Morphological close to fill internal gaps
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    brain_mask = cv2.morphologyEx(brain_thresh, cv2.MORPH_CLOSE, kernel)
    brain_mask = cv2.morphologyEx(brain_mask, cv2.MORPH_OPEN, kernel)

    # Keep only the largest connected region (the brain)
    contours, _ = cv2.findContours(brain_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        brain_mask = np.zeros_like(brain_mask)
        cv2.drawContours(brain_mask, [largest], -1, 255, thickness=cv2.FILLED)

    # Slightly erode to avoid heatmap bleeding onto skull ring
    erode_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
    brain_mask = cv2.erode(brain_mask, erode_kernel, iterations=1)

    return brain_mask


def overlay_heatmap(
    original_img_rgb: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.5,
    sigma: float = 8.0,
    activation_threshold: float = 0.3,  # ignore low activation areas
) -> np.ndarray:
    """
    Overlays heatmap ONLY on brain region with strong activation.
    Low-activation pixels remain as the original MRI — brain detail preserved.
    """
    H, W = original_img_rgb.shape[:2]

    # ── 1. Resize + smooth heatmap ────────────────────────────────────────
    heatmap_resized = cv2.resize(heatmap, (W, H), interpolation=cv2.INTER_CUBIC)
    heatmap_smooth = gaussian_filter(heatmap_resized, sigma=sigma)
    if heatmap_smooth.max() > 0:
        heatmap_smooth /= heatmap_smooth.max()
    heatmap_smooth = np.clip(heatmap_smooth, 0, 1)

    # ── 2. Extract brain mask to confine heatmap inside brain only ────────
    gray = cv2.cvtColor(original_img_rgb, cv2.COLOR_RGB2GRAY)
    brain_mask = extract_brain_mask(gray)                    # uint8, 0 or 255
    brain_mask_f = (brain_mask / 255.0).astype(np.float32)  # float [0,1]

    # ── 3. Apply brain mask — zero out heatmap outside brain ──────────────
    heatmap_masked = heatmap_smooth * brain_mask_f

    # ── 4. Threshold: only show heatmap where activation is meaningful ────
    heatmap_masked = np.where(heatmap_masked >= activation_threshold, heatmap_masked, 0.0)

    # ── 5. Colorise with JET ──────────────────────────────────────────────
    heatmap_8bit = np.uint8(255 * heatmap_masked)
    heatmap_colored = cv2.applyColorMap(heatmap_8bit, cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

    # ── 6. Per-pixel blend: weight = alpha * activation ───────────────────
    #    → hotspot areas are vivid, cool/zero areas show original MRI
    weight = (alpha * heatmap_masked)[..., np.newaxis]       # (H, W, 1)
    base = original_img_rgb.astype(np.float32)
    colored = heatmap_colored.astype(np.float32)

    blended = (colored * weight + base * (1.0 - weight)).clip(0, 255).astype(np.uint8)

    # ── 7. Outside brain mask: always show original MRI ───────────────────
    brain_mask_3d = np.repeat(brain_mask[:, :, np.newaxis], 3, axis=2)
    result = np.where(brain_mask_3d > 0, blended, original_img_rgb)

    return result


def run_gradcam(
    img_array: np.ndarray,
    original_img_rgb: np.ndarray,
    last_conv_layer_name: str = "relu",
    alpha: float = 0.5,
    sigma: float = 8.0,
    activation_threshold: float = 0.3,
) -> dict:

    heatmap, pred_class, confidence, probs = get_gradcam_heatmap(
        img_array,
        last_conv_layer_name=last_conv_layer_name,
    )

    overlay = overlay_heatmap(
        original_img_rgb,
        heatmap,
        alpha=alpha,
        sigma=sigma,
        activation_threshold=activation_threshold,
    )

    all_probs = {CLASS_NAMES[i]: round(float(probs[i]), 4) for i in range(len(CLASS_NAMES))}

    return {
        "overlay"    : overlay,
        "heatmap"    : heatmap,
        "pred_class" : pred_class,
        "class_name" : CLASS_NAMES[pred_class],
        "confidence" : confidence,
        "all_probs"  : all_probs,
    }