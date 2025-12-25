# Image Classification Pipeline with PyTorch Lightning (CIFAR-10)

## 📌 Overview
This project implements an **end-to-end image classification pipeline** using **PyTorch Lightning** on the **CIFAR-10** dataset.  
It includes data augmentation, training, validation, testing, early stopping, model checkpointing, and TensorBoard logging.

The model was trained on **Google Colab with GPU acceleration** and achieved **~81% validation accuracy** and **~80% test accuracy**, exceeding the required 70% accuracy.

---

## 🧠 Dataset
- **Dataset:** CIFAR-10  
- **Total Images:** 60,000  
  - Training: 50,000  
  - Testing: 10,000  
- **Image Size:** 32×32 RGB  
- **Classes:** 10  

---

## 🛠️ Technologies Used
- Python  
- PyTorch  
- PyTorch Lightning  
- Torchvision  
- TorchMetrics  
- TensorBoard  
- YAML (configuration file)  
- Google Colab (GPU)

---

## 📂 Project Structure
```
.
├── config.yaml        # Training configuration
├── model.py           # LightningModule (model + training logic)
├── train.py           # Training, validation, and testing script
├── logs/              # TensorBoard logs and checkpoints
├── data/              # CIFAR-10 dataset
└── README.md
```

---

## ⚙️ Model Architecture
- **Backbone:** ResNet-18  
- **Loss Function:** Cross Entropy Loss  
- **Optimizer:** Adam  
- **Metric:** Accuracy  

---

## 🔄 Data Augmentation
Applied only to training data:
- Random Crop  
- Random Horizontal Flip  
- Normalization  

---

## 🚀 Training Features
- PyTorch Lightning `LightningModule`
- Automatic training, validation, and testing loops
- Early stopping
- Model checkpointing (best validation accuracy)
- TensorBoard experiment tracking
- Configuration-driven training using `config.yaml`

---

## 📊 Results
- **Training Accuracy:** ~79%  
- **Validation Accuracy:** ~81%  
- **Test Accuracy:** ~80%  

The model generalizes well and meets all task requirements.

---

## ▶️ How to Run (Google Colab)

1. Open a new notebook in **Google Colab**
2. Enable GPU:  
   `Runtime → Change runtime type → GPU`
3. Install dependencies in a code cell:
   ```python
   !pip install pytorch-lightning torchmetrics pyyaml tensorboard
   ```
4. Create the project files (`config.yaml`, `model.py`, `train.py`)
5. Run training and testing:
   ```python
   !python train.py
   ```
6. View TensorBoard logs:
   ```python
   %load_ext tensorboard
   %tensorboard --logdir logs
   ```

Training, validation, and testing are handled automatically.

---

## 🧪 Model Testing
The trained model is evaluated on the CIFAR-10 **test dataset** using PyTorch Lightning’s built-in testing loop.  
The final test accuracy is reported after training completes.

---

## 🖼️ Inference on Custom Images
The trained model can be used to predict uploaded images by:
- Resizing images to 32×32
- Applying the same normalization as CIFAR-10
- Running inference using the saved model checkpoint

---

## 📦 Reproducibility
- Hyperparameters stored in `config.yaml`
- Best model checkpoint saved automatically
- Training logs available via TensorBoard
- Project can be reproduced locally or containerized using Docker (optional)

