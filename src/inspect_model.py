import os
import sys

# Add app directory to path so we can import
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

from tensorflow.keras.models import load_model

model_path = "brain_tumor_detector.h5"
if not os.path.exists(model_path):
    print("Model not found")
    sys.exit(1)

model = load_model(model_path)
print("Model loaded successfully.")

# Get last few layers
print("\n--- Last 10 layers ---")
for layer in model.layers[-10:]:
    print(f"Name: {layer.name}, Type: {layer.__class__.__name__}, Output Shape: {layer.output_shape}")

# Print the names of the final conv layers
conv_layers = [layer.name for layer in model.layers if 'conv' in layer.name or 'concat' in layer.name or type(layer).__name__ == 'Conv2D']
print("\n--- Potential last conv layers ---")
print(conv_layers[-5:])
