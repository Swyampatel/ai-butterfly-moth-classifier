# AI Butterfly & Moth Classifier

## 📌 Overview
The **AI Butterfly & Moth Classifier** is a deep learning-powered image classification system that identifies butterfly and moth species from images. It leverages **MobileNetV2**, a lightweight yet powerful deep learning model, to provide **fast and accurate** species identification. The model is deployed using a **Flask API**, enabling users to upload images for real-time classification.

## 🚀 Features
- 🦋 **Classifies butterflies and moths** from uploaded images
- ⚡ **Lightweight & efficient** using MobileNetV2
- 📡 **Flask API for easy integration**
- 🖼️ **Supports image uploads & file path-based classification**
- 📊 **Powered by TensorFlow for high accuracy**

## 🏗️ Tech Stack
- **Deep Learning Model:** MobileNetV2
- **Framework:** TensorFlow/Keras
- **Backend:** Flask (Python)
- **Deployment:** Docker (Optional), Cloud Integration (Future Scope)

## 🔧 Installation & Setup
1. **Clone the Repository**
   ```sh
   git clone https://github.com/Swyampatel/ai-butterfly-moth-classifier.git
   cd ai-butterfly-moth-classifier/backend
   ```

2. **Create a Virtual Environment (Optional but Recommended)**
   ```sh
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Run the Flask Server**
   ```sh
   python app.py
   ```
   The API will be available at **http://127.0.0.1:5000/**.

## 📡 API Usage
### **1. Upload an Image for Classification**
   **Endpoint:** `POST /predict`
   - **Request:** Multipart Form Data with `image` file
   - **Response:** JSON with predicted species label & confidence score

   **Example Request (cURL):**
   ```sh
   curl -X POST -F "image=@butterfly.jpg" http://127.0.0.1:5000/predict
   ```
   **Example Response:**
   ```json
   {
     "species": "Monarch Butterfly",
     "confidence": 0.98
   }
   ```

## 📌 Future Improvements
- 🏗️ **Enhance model accuracy** with more training data
- ☁️ **Deploy on cloud (AWS/GCP/Heroku)**
- 📱 **Create a mobile-friendly frontend UI**

## 🤝 Contributing
Contributions are welcome! If you’d like to contribute, fork the repo and submit a pull request.

## 📝 License
This project is licensed under the **MIT License**.

## 📬 Contact
For any inquiries, reach out to me on:
- **GitHub:** [Swyampatel](https://github.com/Swyampatel)
- **LinkedIn:** [linkedin.com/in/swyampatel](https://linkedin.com/in/swyampatel)

