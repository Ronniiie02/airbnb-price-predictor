# Airbnb Price Prediction – FastAPI App

This project is a full-stack data application that predicts Airbnb listing prices in New York City using machine learning.  
It provides an interactive web interface and a FastAPI backend for filtering listings and generating price predictions.

The project is designed to be **fully reproducible**, **team-friendly**, and **easy to run locally**.

---

## 🚀 Features

- Machine learning–based Airbnb price prediction  
- FastAPI backend with RESTful APIs  
- Interactive frontend (HTML + JavaScript)  
- Dynamic filtering and real-time inference  
- Reproducible Conda environment for team collaboration  

---

## 🗂 Project Structure

```text
airbnb-price-predictor/
│
├── main.py                # FastAPI backend application
├── requirements.txt       # Python dependencies (reference only)
├── environment.yml        # Conda environment (recommended)
├── AB_NYC_2019.csv        # NYC Airbnb dataset
├── static/
│   ├── index.html         # Frontend UI
│   └── main.js            # Frontend logic
└── README.md

🧪 Environment Setup (Recommended)

⚠️ Important
Do NOT use Python venv or the Conda base environment.
Always use the provided Conda environment for consistency.

1️⃣ Create the Conda environment
conda env create -f environment.yml

2️⃣ Activate the environment
conda activate airbnb_env

3️⃣ Install dependencies (if needed)
pip install -r requirements.txt

▶️ Run the Application

Start the FastAPI server using:

python -m uvicorn main:app --reload --port 8000

🌐 Access the App

Web Application:
http://127.0.0.1:8000

API Documentation (Swagger UI):
http://127.0.0.1:8000/docs

🧠 Data & Model Overview

Dataset: AB_NYC_2019.csv

Key features:

Latitude / Longitude

Room type

Availability

Number of reviews

Host listing count

Target variable:

Log-transformed nightly price

The machine learning model is trained offline and loaded at runtime for prediction.
