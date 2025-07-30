# Employee Monthly Income Prediction

This project trains and deploys a machine learning model to predict an employee’s **Monthly Income** using key features from the IBM HR Analytics Attrition dataset. A web interface is provided via Streamlit for interactive predictions.

## Repository Structure

employee_salary_prediction/
├── models/ # Saved model and preprocessing artifacts
│ ├── income_rf_model.pkl
│ ├── income_scaler.pkl
│ └── income_label_encoders.pkl
├── train_income_model.py # Training script for income prediction
├── streamlit_income_app.py # Streamlit web application
├── IBM_Attrition.csv # Raw dataset (placed at project root)
├── requirements.txt # Python dependencies
└── README.md # This file

text

## Prerequisites

- Python 3.8 or higher  
- pip  
- (Optional) Git to clone the repository

## Setup Instructions

1. **Clone the repository**  
git clone <repository-url>
cd employee_salary_prediction

text

2. **Create & activate a virtual environment**  
Windows
python -m venv .venv
..venv\Scripts\activate

macOS / Linux
python3 -m venv .venv
source .venv/bin/activate

text

3. **Install dependencies**  
pip install --upgrade pip
pip install -r requirements.txt

text

4. **Ensure dataset is present**  
Place `IBM_Attrition.csv` in the project root. If your file has a different name, update the filename in `train_income_model.py` and `streamlit_income_app.py` accordingly.

## Training the Model

Run the training script to:

- Load and preprocess the data  
- Encode categorical features and scale numeric features  
- Train a Random Forest Regressor to predict monthly income  
- Evaluate performance (MAE, R²)  
- Save model and preprocessing objects to `models/`

python train_income_model.py

text

**Expected output**:
Test MAE: <value>
Test R² : <value>
Models and preprocessors saved to ./models/

text

## Running the Web Interface

Launch the Streamlit app for interactive predictions:

streamlit run streamlit_income_app.py

text

A browser window will open at `http://localhost:8501` where you can:

1. Select **Over Time?** (Yes/No)  
2. Input **Age** (18–60)  
3. Enter **Total Working Years** (0–40)  
4. Choose **Stock Option Level** (0–3)  
5. Select **Work–Life Balance Rating** (1–4)  
6. Pick **Job Role** from the dropdown  
7. Pick **Education Field** from the dropdown  

Click **Predict Monthly Income** to view the estimated salary.

## Notes

- To retrain with updated data, replace `IBM_Attrition.csv` and rerun `python train_income_model.py`.  
- The virtual environment isolates dependencies; deactivate it with `deactivate` when finished.  
- On Python 3.12+, if you encounter pip build errors, ensure your `pyproject.toml` includes:
