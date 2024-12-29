# INM702 Coursework: Neural Network Implementation

This repository contains the implementation of two tasks for the INM702 coursework:  
1. **Customer Airline Satisfaction Prediction**: Neural network built from scratch using NumPy.  
2. **Animal Image Classification**: Convolutional neural networks implemented using PyTorch.

---

## **Requirements**

### **Prerequisites**
- Python 3.8 or later  
- A Kaggle account to access datasets  
- Libraries and tools:
  - `numpy`
  - `pandas`
  - `matplotlib`
  - `opendatasets`
  - `torch`
  - `torchvision`
  - `scikit-learn`

---

## **Task 1: Customer Airline Satisfaction Prediction**

### **Description**  
This task involves predicting customer satisfaction with airline services (satisfied or neutral/dissatisfied) using a fully connected neural network built from scratch with NumPy.  

The dataset contains 23 features like flight class, customer type, seat comfort, online boarding, and food service.  

**Source**: [Airline Passenger Satisfaction Dataset](https://www.kaggle.com/datasets/teejmahal20/airline-passenger-satisfaction)

---

### **Setup Instructions**

1. **Install Required Libraries**  
Run the following command to install necessary Python libraries:  
```bash
pip install numpy pandas matplotlib scikit-learn
```

2. **Download the Dataset**  
   - **Option 1: Manual Download**
     1. Download the dataset from [Kaggle](https://www.kaggle.com/datasets/teejmahal20/airline-passenger-satisfaction).  
     2. Place the files in a directory named `airline_data/`.
   - **Option 2: Use opendatasets**  
     ```python
     import opendatasets as od
     od.download("https://www.kaggle.com/datasets/teejmahal20/airline-passenger-satisfaction")
     ```
     The dataset will be saved in the `airline_data/` directory.

3. **Run the Script**  
   Execute the script for training and evaluation:  
   ```bash
   python nn_from_scratch.py
   ```

---

### **Features and Functionality**
- **Activation Functions**: Sigmoid, ReLU, Softmax (with forward and backward passes).  
- **Dropout**: Inverted dropout implemented to reduce overfitting.  
- **Optimizers**: Stochastic Gradient Descent (SGD) and SGD Mini Batch.  
- **Regularization**: L1 and L2 regularization techniques.  
- **Customizable Architecture**: Flexible design with adjustable layers, neurons, and activation functions.

---

### **Output**
- Accuracy, precision, and confusion matrices for various architectures.
- Plots for loss and accuracy trends over epochs.  

---

## **Task 2: Animal Image Classification**

### **Description**  
This task classifies images of cats, dogs, and snakes using CNNs and transfer learning models in PyTorch.  

The dataset contains 3,000 images, evenly distributed among the three classes.  

**Source**: [Animal Image Classification Dataset](https://kaggle.com/datasets/borhanitrash/animal-image-classification-dataset/data)

---

### **Setup Instructions**

1. **Install Required Libraries**  
Run the following command to install necessary Python libraries:  
```bash
pip install torch torchvision opendatasets numpy pandas matplotlib scikit-learn
```

2. **Download the Dataset**  
   - **Option 1: Manual Download**
     1. Download the dataset from [Kaggle](https://kaggle.com/datasets/borhanitrash/animal-image-classification-dataset/data).  
     2. Extract the dataset into a directory named `data/`.
   - **Option 2: Use opendatasets**  
     ```python
     import opendatasets as od
     od.download("https://kaggle.com/datasets/borhanitrash/animal-image-classification-dataset/data")
     ```
     The dataset will be saved in the `data/` directory.

3. **Run the Script**  
   Execute the script for training and evaluation:  
   ```bash
   python animal_classifier.py
   ```

---

### **Features and Functionality**
- **Custom CNN Model**: Built with 3 convolutional layers, ReLU activations, and max pooling.  
- **Pre-trained Models**: Transfer learning with AlexNet and VGG16, fine-tuned for this task.  
- **Hyperparameter Optimization**: Adjustments for learning rate, batch size, and dropout.  

---

### **Output**
- Training and validation metrics for all models.
- Confusion matrices and classification metrics (accuracy, precision, sensitivity, specificity).  
- Performance comparison between custom and pre-trained models.

---

## **Notes**
- GPU support is recommended for Task 2 to speed up training.  
- Ensure datasets are placed in the correct directories for both tasks.  
- The Jupyter notebook contains additional derivations and visualizations for reference.

---

## **References**
- [Airline Passenger Satisfaction Dataset](https://www.kaggle.com/datasets/teejmahal20/airline-passenger-satisfaction)  
- [Animal Image Classification Dataset](https://kaggle.com/datasets/borhanitrash/animal-image-classification-dataset/data)  
- PyTorch Documentation: [PyTorch.org](https://pytorch.org/docs/)  

---
