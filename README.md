# 🐶🐱 Cat vs Dog Image Classifier using CNN

### 📌 Overview
This project implements a **CNN-based binary classifier** using **TensorFlow and Keras** to distinguish between images of cats and dogs. The model is trained on the **FreeCodeCamp Cats & Dogs Dataset** and utilizes **data augmentation techniques** for better generalization. The trained model can then predict whether an image contains a cat or a dog.

## Project Structure
```
├── cats_and_dogs.zip  # Dataset (downloaded)
├── train              # Training images (cats and dogs)
├── validation         # Validation images (cats and dogs)
├── test               # Test images
└── model_training.py  # Main script
```

### 📊 Dataset
* **Source**: FreeCodeCamp Cats & Dogs Dataset
* **Data Structure**:
  * `train/`: Training images (contains `cats/` and `dogs/`)
  * `validation/`: Validation images (contains `cats/` and `dogs/`)
  * `test/`: Test images without labels

### ⚙️ Technologies & Skills Used
* **Deep Learning**: Convolutional Neural Networks (CNN)
* **Machine Learning**: Image Classification
* **TensorFlow & Keras**: Model development
* **Python Libraries**: NumPy, Pandas, Matplotlib
* **Data Augmentation**: Rotation, flipping, shifting, zooming
* **Model Evaluation**: Accuracy, Loss, Training & Validation plots

### 🚀 Project Workflow
#### 1️⃣ Data Preprocessing
* Rescale images (`1./255`)
* Convert categorical labels into binary format
* Split into **train**, **validation**, and **test** sets
* Perform **data augmentation** for robust model training

#### 2️⃣ Building the CNN Model
* **Conv2D** and **MaxPooling2D** layers for feature extraction
* **Flatten** to transform 2D features into a 1D vector
* **Dense layers** with ReLU activation
* **Sigmoid activation** for binary classification

#### 3️⃣ Model Compilation & Training
* **Optimizer**: Adam
* **Loss Function**: Binary Crossentropy
* **Metrics**: Accuracy
* **Epochs**: 15
* **Batch Size**: 128

#### 4️⃣ Evaluation & Testing
* Train vs Validation Accuracy/Loss plots
* Model prediction on **unlabeled test images**
* Visualization of test image classifications

### 🔧 How to Run
#### 1️⃣ Install dependencies
```bash
pip install tensorflow numpy pandas matplotlib
```
#### 2️⃣ Run the script
```python
[python cat_vs_dog_classifier.py](https://colab.research.google.com/drive/1nCKZIwqawh2L2gb3KOKbhgl4EBMdLH0U?usp=sharing)
```
#### 3️⃣ View the predictions
* The model predicts whether an image contains a **cat or dog** based on its features.

### 📈 Results & Future Improvements
✅ **Achieved over 69% accuracy** on validation data. The training and validation accuracy and loss are plotted to analyze model performance. The model makes predictions on test images, displaying probabilities of being a cat or a dog.

## Skills Used
- TensorFlow/Keras
- Convolutional Neural Networks (CNN)
- Image Preprocessing & Augmentation
- Data Visualization with Matplotlib
- Python Programming
- NumPy for Array Manipulations


🔹 **Possible Enhancements**:
* Try **deeper architectures** (VGG16, ResNet, etc.)
* Implement **Transfer Learning**
* Optimize **hyperparameters** (learning rate, batch size, etc.)

### 🏆 Acknowledgments
* **FreeCodeCamp** for the dataset
* **TensorFlow & Keras** for the framework

## License
This project is open-source and available under the MIT License.

📌 **Want to contribute?** Fork the repo and submit a PR! 🎯

## Author
[Hamna Jamila](https://www.linkedin.com/in/hamna-jamila-58b478270/)



