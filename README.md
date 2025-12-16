# 🚆 Rail Component Classification Model

An AI-powered railway component classification system developed to support **SIH Project – Vimarsha**.  
This model identifies **ERCs (Elastic Rail Clips), Liners, and Clips** from images and provides a **confidence score** for each prediction.

---

## 📌 Project Overview

This repository contains a deep learning–based **railway component classification model** deployed as a web application.  
The system analyzes railway track images and predicts whether the detected component is:

- **ERC (Elastic Rail Clip)**
- **Liner**
- **Clip**

Along with the predicted class, the model outputs a **confidence score**, indicating the certainty of the prediction.

This solution is an integral AI module of **Vimarsha – Track Fittings Lifecycle Management System**, developed under the **Smart India Hackathon (SIH)**.

---

## 🧠 Model Details

- **Supported Components:**
  - ERC (Elastic Rail Clip)
  - Liner
  - Clip

- **Prediction Output:**
  - Predicted component type
  - Confidence score (probability of match)

- **Model Format:**
  - `best.pt` – PyTorch trained model
  - `best.onnx` – ONNX optimized model for deployment

- **Training Dataset:**
  - **150+ real sample images**
  - Dataset **created entirely by our team**
  - **Manual bounding box annotation** for ERCs, liners, and clips

- **Model Architecture:**
  - YOLO-based object detection and classification
  - Optimized for **real-time inference**

---

## 📂 Repository Structure

```text
RailClassification-main/
│
├── app.py               # Flask backend for model inference
├── index.html           # Frontend interface
├── best.pt              # Trained PyTorch model
├── best.onnx            # ONNX optimized model
├── requirements.txt     # Python dependencies
├── render.yaml          # Render deployment configuration
```

## ⚙️ Tech Stack

- **Programming Language:** Python  
- **Backend Framework:** Flask  
- **AI Model:** YOLO (PyTorch → ONNX)  
- **Frontend:** HTML  
- **Deployment Platform:** Render Cloud  
- **Annotation Tool:** Manual bounding box drawing  

---

## 🚀 Deployment

The application is deployed on **Render Cloud** and accessible publicly.

🌐 **Live URL:**  
👉 https://railclassification-7.onrender.com

---

## 🧪 How It Works

1. A railway track image is uploaded or provided  
2. The image is processed by the trained ONNX model  
3. The model detects and classifies the component  
4. Output includes:
   - Component type (ERC / Liner / Clip)
   - Confidence score  
5. Results are displayed in real time via the web interface  

---

## 🏗️ Role in SIH Project – Vimarsha

This model acts as an **AI-based verification layer** in the **Vimarsha ecosystem**, enabling:

- Automated identification of track fittings  
- Confidence-based verification of ERCs, liners, and clips  
- Reduced dependency on manual inspection  
- Scalable deployment for railway infrastructure monitoring  

---

## 👨‍💻 Contributors

- 👤 [IncharaS06](https://github.com/IncharaS06)  
- 👤 [yashwanths814](https://github.com/yashwanths814)  

---

## 📜 License

This project is developed for **academic and hackathon purposes** under the **Smart India Hackathon (SIH)**.  
Any reuse or extension should include proper attribution to the contributors.
