# QR Code Classification

## Overview
This project classifies QR code images as "Original" or "Counterfeit" using traditional machine learning techniques. The pipeline involves feature engineering, model training, and evaluation using Support Vector Machine (SVM) and Logistic Regression models. Features like brightness, contrast, Local Binary Patterns (LBP), and a custom CDP degradation metric are extracted from grayscale QR code images. The SVM model achieved a test accuracy of 97%, while Logistic Regression performance is estimated at ~90% based on visualizations. The trained models are saved and can be used for inference on new QR code images.

## Prerequisites
- Python 3.11 or higher
- Required libraries:
  - opencv-python (for image processing)
  - scikit-learn (for machine learning models and evaluation)
  - numpy (for numerical operations)
  - matplotlib (for visualizations)
  - joblib (for saving/loading models)

## Installation
1. Clone the repository or download the project files.
2. Install the required dependencies using pip:
   ```bash
   pip install opencv-python scikit-learn numpy matplotlib
   ```
3. Ensure you have the dataset directories (/content/first_print for original QR codes and /content/Secondprint for counterfeit QR codes) or update the paths in the code to match your dataset location.

## Usage
1. **Run the Notebook:**
   - Open the Jupyter notebook (`qr_code_classification.ipynb`) in a Jupyter environment.
   - Execute the cells sequentially to perform data exploration, feature engineering, model training, and evaluation.
   - The notebook includes visualizations such as confusion matrices and model comparison bar charts.

2. **Train Models:**
   - The notebook trains two models: SVM and Logistic Regression.
   - SVM is trained with a polynomial kernel, and hyperparameters are tuned using GridSearchCV (parameters: C, gamma, degree).
   - Logistic Regression is trained with hyperparameters tuned for C, solver, penalty, and class_weight.
   - Models are saved as `svm_model.pkl` and `lr_model.pkl` using joblib.

3. **Inference on New Images:**
   - Use the provided inference script to classify new QR code images:
     - Load the saved models (`svm_model.pkl` and `lr_model.pkl`).
     - Preprocess new images using the `preprocess_image` function (resizes to 150x150 and normalizes pixel values).
     - Predict using the `predict_image` function for both SVM and Logistic Regression.

## Project Structure
- `qr_code_classification.ipynb`: Main Jupyter notebook containing the full pipeline (data exploration, feature engineering, model training, evaluation, and inference).
- `svm_model.pkl`: Saved SVM model for inference.
- `lr_model.pkl`: Saved Logistic Regression model for inference (note: saving fails in the notebook due to a NameError for `lr_grid_search`).
- **Dataset (not included but required):**
  - `/content/first_print`: Directory containing original QR code images.
  - `/content/Secondprint`: Directory containing counterfeit QR code images.

## Key Features
- **Feature Engineering:** Extracts brightness, contrast, LBP texture histograms, and a CDP degradation metric to capture differences between original and counterfeit QR codes.
- **Preprocessing:** Applies `StandardScaler` for feature scaling and PCA (retaining 95% variance) for dimensionality reduction.
- **Models:** Trains SVM (97% test accuracy) and Logistic Regression (~90% accuracy, estimated from visualizations) with hyperparameter tuning via GridSearchCV.
- **Evaluation:** Includes classification reports, confusion matrices, and a model comparison bar chart.
- **Inference:** Provides functions to preprocess and predict on new QR code images using saved models.

## Notes
- **Error in Code:** The notebook attempts to save the Logistic Regression model (`lr_grid_search.best_estimator_`) but encounters a NameError because `lr_grid_search` is not defined in the scope. Ensure the Logistic Regression training cell is executed before saving the model.
- **Dataset Dependency:** The code assumes the dataset is available at `/content/first_print` and `/content/Secondprint`. Update the paths in the notebook if your dataset is located elsewhere.
- **Inference Limitations:** The `predict_image` function flattens images for SVM and Logistic Regression, which may not align with the PCA-transformed features used during training. Ensure the inference pipeline matches the training pipeline (e.g., apply the same PCA transformation).
- **Future Improvements:** Consider integrating deep learning models (e.g., CNNs) for better feature extraction, as hinted in the code comments, and test the models under varied real-world conditions.

## License
This project is for educational purposes and does not include a specific license. Use and modify the code as needed for your own projects.
