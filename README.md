# ResNet-18 Image Classifier

An image classification application based on a fine-tuned ResNet-18 convolutional neural network, developed using PyTorch and Streamlit.

**Live application:** [Open the Image Classifier](https://resnet18-image-classifier.streamlit.app/)

## 1. Project Overview

This project implements a deep learning application capable of classifying images into six categories:

- Buildings
- Forest
- Glacier
- Mountain
- Sea
- Street

The application uses a ResNet-18 model fine-tuned on the Intel Image Classification dataset. Users can upload an image through a web interface and obtain the predicted category together with the classification scores for all six classes.

## 2. Dataset

The project uses the [Intel Image Classification dataset](https://www.kaggle.com/datasets/puneet6060/intel-image-classification), available on Kaggle.

The dataset contains images belonging to six different environmental and urban scene categories.

The original training dataset was divided into training and validation subsets using stratified sampling:

- Training set: 9,823 images
- Validation set: 4,211 images
- Independent test set: 3,000 images

Images were resized to 150 × 150 pixels and converted to RGB format.

## 3. Development Process and Model Comparison

Several machine learning and deep learning approaches were implemented and evaluated during the project.

The evaluated models included:

- Multilayer Perceptron (MLP)
- Convolutional Neural Network (CNN)
- Random Forest using raw image pixels
- Random Forest using autoencoder embeddings
- ResNet-18

The autoencoder was used to compress the original image representation into a 256-dimensional latent vector before applying Random Forest classification.

The following table summarizes the test accuracy obtained by each approach:

| Model | Test Accuracy |
|---|---:|
| MLP | 39.87% |
| CNN | 80.43% |
| Random Forest (raw pixels) | 57.30% |
| Random Forest (AE embeddings) | 57.20% |
| **ResNet-18** | **92.67%** |

ResNet-18 achieved the highest classification accuracy and was therefore selected for the final application.

## 4. ResNet-18 Architecture and Training

The main classification model is ResNet-18, a convolutional neural network based on residual learning.

The model was initialized with ImageNet-pretrained weights and fine-tuned for the six target categories.

Main training configuration:

- Architecture: ResNet-18
- Loss function: Cross-Entropy Loss
- Optimizer: Adam
- Learning rate: 0.0001
- Batch size: 32
- Number of epochs: 10
- Number of output classes: 6
- Number of parameters: 11,179,590

The final fully connected layer was adapted to produce six classification outputs.

The selected checkpoint was determined based on the lowest validation loss.

## 5. Model Performance

The selected model was evaluated using the independent test dataset.

| Metric | Result |
|---|---:|
| Accuracy | 92.67% |
| Macro Precision | 92.91% |
| Macro Recall | 92.78% |
| Macro F1-score | 92.81% |

The results indicate that ResNet-18 achieved strong classification performance across the six categories.

## 6. Application Development

After training and evaluating the model, a separate web application was developed using Streamlit and PyTorch.

The trained model parameters were saved in the `resnet_best_model.pth` checkpoint and subsequently loaded by the application.

The application includes the following features:

- Image upload through a web interface.
- Automatic image preprocessing.
- Classification using the trained ResNet-18 model.
- Display of the predicted category.
- Display of classification scores for all six categories.
- Warning when the two highest classification scores are close.

The application uses the same image size and ImageNet normalization applied during model training.

The displayed scores are softmax outputs and should not be interpreted as calibrated probabilities.

## 7. Technologies

- Python
- PyTorch
- Torchvision
- Streamlit
- Pandas
- Pillow
- Scikit-learn

## 8. Running the Application Locally

Clone this repository or download its files.

Install the required dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
streamlit run app.py
```

The application will become available in your browser, normally at:

`http://localhost:8501`

The trained model checkpoint, `resnet_best_model.pth`, must be located in the same directory as `app.py`.

CPU inference is supported; a GPU is not required.

## 9. Repository Structure

```text
resnet18-image-classifier/
│
├── README.md
├── app.py
├── requirements.txt
└── resnet_best_model.pth
```

## 10. References

1. Intel Image Classification Dataset. Kaggle.  
   https://www.kaggle.com/datasets/puneet6060/intel-image-classification

2. He, K., Zhang, X., Ren, S., and Sun, J. (2016). Deep Residual Learning for Image Recognition. *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, 770–778.  
   https://doi.org/10.1109/CVPR.2016.90

---

**Project:** ResNet-18 Image Classifier  
**Application framework:** Streamlit  
**Deep learning framework:** PyTorch
