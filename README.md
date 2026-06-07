This project implements a Convolutional Neural Network (CNN) to classify chest X-ray images into four categories:

1. NORMAL
2. PNEUMONIA
3. COVID19
4. TURBERCULOSIS

The model is trained using TensorFlow/Keras and exported to TensorFlow Lite (TFLite) and TensorFlow.js formats for deployment on mobile and web platforms.

## Dataset

Dataset source:
https://www.kaggle.com/datasets/jtiptj/chest-xray-pneumoniacovid19tuberculosis

## Requirements

The project requires the following Python libraries:

- tensorflow
- tensorflowjs
- numpy
- pandas
- matplotlib
- seaborn
- scikit-learn
- opencv-python
- scikit-image
- Pillow
- tqdm
- kaggle

Install dependencies using:

pip install -r requirements.txt

## Generated Files

### Model Files

- best_model.h5
  - Best trained Keras model saved during training.

- model.tflite
  - TensorFlow Lite version of the trained model for mobile deployment.

### TensorFlow SavedModel

- saved_model/
  - TensorFlow SavedModel directory used for TFLite conversion.

### TensorFlow.js Model

- tfjs_model/
  - TensorFlow.js version of the trained model.

Contents:

- model.json
- group1-shard1of1.bin

### Labels File

- labels.txt
  - Contains class labels used by the model.

Example:

NORMAL
PNEUMONIA
COVID19
TURBERCULOSIS

## Project Structure

submission/
│
├── best_model.h5
├── model.tflite
├── labels.txt
├── requirements.txt
├── README.txt
│
├── saved_model/
│ ├── assets/
│ ├── variables/
│ ├── saved_model.pb
│ └── fingerprint.pb
│
└── tfjs_model/
├── model.json
└── group1-shard1of1.bin

## Model Input Specification

Input Shape:
(150, 150, 1)

Input Type:
Grayscale image

Preprocessing:

- Resize image to 150 × 150 pixels
- Convert to grayscale
- Normalize pixel values by dividing by 255

## Model Output

Output Shape:
(4,)

Class Order:

0 → COVID19
1 → NORMAL
2 → PNEUMONIA
3 → TURBERCULOSIS

Note:
The class order may vary depending on the directory structure used during training. Always verify using the generated labels.txt file.

## Author

Developed as an image classification project using TensorFlow and Keras.
