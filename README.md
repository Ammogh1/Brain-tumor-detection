# Brain Tumor Detection Pipeline 🧠

A robust, end-to-end deep learning pipeline for the precise classification of MRI brain scans. This system utilizes a fine-tuned **DenseNet121** architecture to accurately diagnose tumors across four distinct categories: Glioma, Meningioma, Pituitary, and No Tumor (Normal).

## 🌟 Key Features

*   **High-Accuracy Classification:** Powered by a customized DenseNet121 model with an ablation-tested architecture (Global Average Pooling, Dropout, Dense Layer, and Class Weights).
*   **Explainable AI (XAI):** Integrated **Grad-CAM** (Gradient-weighted Class Activation Mapping) visually highlights the exact regions of the MRI that led to the model's prediction, providing doctors with transparent diagnostic evidence.
*   **Intelligent Preprocessing:** Employs mathematical **Skull-Stripping** via OpenCV to extract the true brain mass. This prevents the model from relying on background noise or skull shapes (Shortcut Learning).
*   **Modern Web UI:** Features a sleek, responsive, and professional frontend built with **Streamlit**, styled with custom CSS for a production-ready medical software feel (dark mode, glassmorphism).

## 🏗️ Architecture

The pipeline follows a straightforward flowchart:
1.  **Input:** Raw MRI Image (224x224x3).
2.  **Preprocessing:** Grayscale conversion, Gaussian Blur, Thresholding, and Contour Bounding (Skull Stripping).
3.  **Feature Extraction:** Pre-trained DenseNet121 Base (unfrozen later layers).
4.  **Classification Head:** Global Average Pooling $\rightarrow$ Dropout (0.5) $\rightarrow$ Dense Layer (512, ReLU) $\rightarrow$ Softmax Output.
5.  **Output:** Probabilities across 4 tumor classes.

## 🚀 Getting Started

### Prerequisites
*   Python 3.8+
*   TensorFlow
*   OpenCV
*   Streamlit

### Installation & Execution
1.  Clone the repository:
    ```bash
    git clone https://github.com/Ammogh1/Brain-tumor-detection.git
    cd Brain-tumor-detection
    ```
2.  Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```
3.  Run the Streamlit application:
    ```bash
    streamlit run main.py
    ```

## 📊 Dataset & Metrics
The model was trained on an imbalanced dataset, which was mitigated using precise class weighting. It achieves exceptional performance across all metrics (Accuracy, Precision, Recall, and F1-Score).

*(Detailed ablation studies, confusion matrices, and ROC curves are generated programmatically and available in the `figures/` directory).*
