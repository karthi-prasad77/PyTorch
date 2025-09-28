# 🔥 PyTorch Learning Roadmap

This repository is a guide for learning **PyTorch** from scratch to mastery. It contains explanations, examples, and a roadmap to help you become confident in building deep learning models using PyTorch.

---

## 📌 What is PyTorch?

PyTorch is a **deep learning framework** that provides:

* **Tensors** → multidimensional arrays (like NumPy but faster, GPU-ready).
* **Autograd** → automatic differentiation for training neural networks.
* **nn.Module** → building blocks for creating models.
* **Optimizers** → gradient descent implementations (SGD, Adam, etc).
* **Data utilities** → dataset loaders and transforms.
* **Ecosystem** → vision (torchvision), text (torchtext), RL (torchrl), etc.

---

## 🚀 Roadmap to Master PyTorch

### 🟢 Stage 1: PyTorch Fundamentals

1. Install PyTorch (`pip install torch torchvision torchaudio`)
2. Learn **Tensors**

   * Creating tensors (`torch.tensor`, `torch.rand`, `torch.zeros`)
   * Tensor shapes & indexing
   * GPU usage (`.to("cuda")`)
3. Learn **Autograd**

   * `requires_grad=True`
   * `.backward()`
   * `.grad`
   * Building simple functions and computing gradients

📘 Example:

```python
import torch

x = torch.tensor(2.0, requires_grad=True)
y = x**2 + 3*x
y.backward()
print(x.grad)  # dy/dx = 2x + 3 = 7
```

---

### 🟡 Stage 2: Neural Networks

1. Learn `nn.Module`

   * Define simple models using `nn.Linear`, `nn.ReLU`
   * Custom models with `forward()`
2. Loss functions (`nn.MSELoss`, `nn.CrossEntropyLoss`)
3. Optimizers (`torch.optim.SGD`, `torch.optim.Adam`)
4. Training Loop

   * forward → loss → backward → update

📘 Example (Simple MLP):

```python
import torch.nn as nn

class SimpleNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2, 4)
        self.fc2 = nn.Linear(4, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)
```

---

### 🟠 Stage 3: Working with Data

1. Datasets and Dataloaders

   * `torch.utils.data.Dataset`
   * `DataLoader` (batching, shuffling)
2. Torchvision for images (`datasets.MNIST`, `transforms`)
3. Torchtext for NLP

📘 Example:

```python
from torch.utils.data import DataLoader, TensorDataset

X = torch.rand(100, 2)
y = (X[:,0] + X[:,1]).unsqueeze(1)
dataset = TensorDataset(X, y)
loader = DataLoader(dataset, batch_size=16, shuffle=True)
```

---

### 🔴 Stage 4: Deep Learning Architectures

1. Convolutional Neural Networks (CNNs) → Image classification
2. Recurrent Neural Networks (RNNs, LSTMs, GRUs) → Text/NLP
3. Transformers → Modern NLP and vision
4. GANs (Generative Adversarial Networks) → Image generation

---

### 🔵 Stage 5: Advanced Topics

1. Transfer Learning & Fine-tuning
2. Distributed Training (`torch.distributed`, `DataParallel`)
3. Mixed Precision Training (`torch.cuda.amp`)
4. Quantization & Model Optimization
5. Exporting to ONNX & deploying models

---

### 🟣 Stage 6: Projects to Build

* ✅ Linear Regression (y = mx + b)
* ✅ XOR Neural Network
* ✅ MNIST Digit Classifier (CNN)
* ✅ Sentiment Analysis (RNN/Transformer)
* ✅ Image Generator (GAN)
* ✅ Deploy a model with FastAPI / Flask
* ✅ Fine-tune a pre-trained model (ResNet, BERT)

---

## 🎯 Resources

* [PyTorch Official Docs](https://pytorch.org/docs/stable/index.html)
* [Deep Learning with PyTorch (book)](https://pytorch.org/deep-learning-with-pytorch)
* [Stanford CS231n](http://cs231n.stanford.edu/)
* [Fast.ai course](https://course.fast.ai/)

---

## ✅ Tips for Mastery

1. **Learn by coding small examples** (don’t just read).
2. **Inspect tensors and gradients** using `.shape`, `.grad`, `.requires_grad`.
3. **Debug with print statements** (or hooks) to see activations.
4. **Start small** (linear regression, XOR) before CNNs/Transformers.
5. **Do projects** → real datasets, competitions (Kaggle), open-source.

---

📌 **Goal:** By following this roadmap, you’ll go from **PyTorch beginner** → **confident practitioner** → **ready for research/production** 🚀
