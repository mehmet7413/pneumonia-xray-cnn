# pneumonia-xray-cnn

This project uses a Convolutional Neural Network (CNN) to detect pneumonia from chest X-ray images. The model is trained and evaluated on the **Chest X-Ray Images (Pneumonia)** dataset obtained from [Kaggle](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia).

---

## 🧠 Model Architecture

- **Input shape**: 150x150 grayscale images
- **Layers**:
  - Conv2D + BatchNormalization + MaxPooling
  - Conv2D + Dropout + BatchNormalization + MaxPooling
  - Conv2D + BatchNormalization + MaxPooling
  - Flatten → Dense → Dropout → Dense(sigmoid)
- **Optimizer**: RMSProp
- **Loss**: Binary Crossentropy
- **Metrics**: Accuracy

---

## 🔄 Data Preprocessing

- **Normalization**: Pixel values scaled to [0, 1]
- **Reshaped**: (150, 150, 1)
- **Data Augmentation**:
  - Rotation, Zoom
  - Horizontal shift & flip

---

## 📊 Results

- **Test Accuracy**: ~XX% *(Update after training)*
- **Evaluation**: True label vs predicted label visualization
- **Graph**: Accuracy & Loss vs Epochs

---

## 📁 Dataset Structure

```
chest_xray/
    train/
        PNEUMONIA/
        NORMAL/
    test/
        PNEUMONIA/
        NORMAL/
    val/
        PNEUMONIA/
        NORMAL/
```

---

## 💻 How to Run

```bash
pip install -r requirements.txt
python Akciger_Kanser_Tespiti.py
```

---

## 📌 License

This project is licensed under the MIT License.

---

## 🙋‍♂️ Author

- GitHub: [@mehmet7413](https://github.com/mehmet7413)
