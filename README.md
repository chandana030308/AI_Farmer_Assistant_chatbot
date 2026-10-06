# 🌾 AI Farmer Assistant

AI Farmer Assistant is a smart agriculture application built using Python and Streamlit. It helps farmers with crop recommendations, farming-related questions, weather information, and crop/soil image analysis.

## 🚀 Features

### 💬 Farming Chatbot
Ask farming-related questions about:
- Crops
- Soil
- Irrigation
- Fertilizers
- Pests
- Seasons
  ![AI-Farmer-Assistant-Chatbot](https://github.com/chandana030308/AI_Farmer_Assistant_chatbot/blob/925b612de7a2208ccdb535f0ed4b2f683a208675/Screenshot%20(566).png)

### 🌱 Crop Recommendation
Provides crop recommendations based on:
- Nitrogen (N)
- Phosphorus (P)
- Potassium (K)
- Temperature
- Humidity
- Soil pH
- Rainfall
  ![AI-Farmer-Assistant-Chatbot](https://github.com/chandana030308/AI_Farmer_Assistant_chatbot/blob/75996e1064be6dd956a791871dfa2865ce92c7db/Screenshot%20(567).png)

### 📷 Crop & Soil Scanner
Allows users to:
- Upload a crop or soil image
- Capture an image using the camera
- Enter farm and soil information
- Analyze visible crop/soil conditions
- Get the top crop recommendations

The scanner uses a local Hugging Face model together with the existing machine learning crop prediction model.

> **Note:** Image analysis cannot accurately determine laboratory soil properties such as exact pH, nitrogen, phosphorus, or potassium values. Soil testing is recommended for accurate measurements.
![AI-Farmer-Assistant-Chatbot](https://github.com/chandana030308/AI_Farmer_Assistant_chatbot/blob/0fe72bf62c7fc03b9f665178fc86579627770ceb/Screenshot%20(568).png)

### ⛅ Weather Information
Provides:
- Current temperature
- Humidity
- Wind speed
- Rainfall
- 5-day weather forecast

Weather data is obtained using Open-Meteo.

### 📊 Dataset Explorer
Allows users to:
- View the agriculture dataset
- Check the number of rows and columns
- View summary statistics
- Explore crop distribution

---

## 🛠️ Technologies Used

- Python
- Streamlit
- Scikit-learn
- Pandas
- NumPy
- Hugging Face Transformers
- PyTorch
- OpenAI API
- Open-Meteo API

---

## 📁 Project Structure

```text
AI-Farmer-Assistant/
│
├── data/
│   └── agriculture_data.csv
│
├── models/
│   └── crop_model.pkl
│
├── src/
│   ├── chatbot.py
│   ├── data_processing.py
│   ├── prediction.py
│   ├── weather.py
│   └── image_analysis.py
│
├── screenshots/
│   ├── chatbot.png
│   ├── crop-recommendation.png
│   ├── crop-soil-scanner.png
│   ├── weather.png
│   └── dataset.png
│
├── app.py
├── requirements.txt
├── .env
└── README.md
