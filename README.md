# American Sign Language Detection 🤟

This project uses a Convolutional Neural Network (CNN) to detect American Sign Language (ASL) signs from images. The system classifies 29 different classes — 26 alphabet signs (A-Z) and 3 control signs (`space`, `del`, `nothing`).

---

## 📌 Dataset

- **Source**: [ASL Alphabet Dataset on Kaggle](https://www.kaggle.com/datasets/grassknoted/asl-alphabet)
- **Classes**: 29 (A-Z, space, del, nothing)

⚠️ The dataset is not included in this repo due to size.  
Please download it manually and place it inside the `data/` folder like so:

ASL_Detection_Project/
└── data/
└── asl_alphabet_train/
├── A/
├── B/
└── ...



---

## 🧠 Project Workflow

1. **Visualize Dataset** → `scripts/visualize_dataset.py`  
2. **Preprocess & Split Data** → `scripts/preprocess_data.py`  
3. **Train CNN Model** → `scripts/train_model.py`  
4. **Evaluate Accuracy & Predictions** → `scripts/evaluate_model.py`

---

## 🏗️ Model

- CNN with 3 Conv layers + Dropout + Dense output
- Output: 29-class softmax
- Trained using TensorFlow/Keras

---

## 🧪 Results

- Accuracy: *(add later after training)*
- Sample predictions & confusion matrix plotted

---

## 🚀 To Run This Project

1. Clone this repo  
2. Create and activate a virtual environment  
3. Install requirements:

```bash
pip install -r requirements.txt
Run the scripts in order:

preprocess_data.py

train_model.py

evaluate_model.py
```
Author
Shravani Bande