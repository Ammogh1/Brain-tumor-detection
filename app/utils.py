import cv2
import numpy as np
from tensorflow.keras.applications.densenet import preprocess_input

IMG_SIZE = (224, 224)
CLASSES = ["glioma", "meningioma", "notumor", "pituitary"]

def crop_brain_contour(img):
    """
    Destroys Shortcut Learning bias by generating a mathematical threshold and extracting only the true brain mass!
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
        
    return cropped_img

def prepare_image_for_model(img_bytes):
    """
    Decodes the byte stream into an image, extracts the brain logically,
    and returns both the cropped visual image (for UI reflection) and the tensor (for models).
    """
    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if img is None:
        return None, None
        
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Initially resize down perfectly
    img_resized = cv2.resize(img_rgb, IMG_SIZE)
    
    # 💥 Obliterate the Shortcut/neck learning feature here:
    cropped_rgb = crop_brain_contour(img_resized)
    
    # Preprocessing for Inference
    img_array = preprocess_input(cropped_rgb.astype(np.float32))
    img_array = np.expand_dims(img_array, axis=0)
    
    return cropped_rgb, img_array

def encode_image_for_db(img_array):
    """
    Compresses an RGB image back into bytes for efficient DB storage (e.g., JPEG).
    """
    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    _, buffer = cv2.imencode('.jpg', img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
    return buffer.tobytes()
