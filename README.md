# Employee Monthly Income Prediction

This project trains and deploys a machine learning model to predict an employee's **Monthly Income** using key features from the IBM HR Analytics Attrition dataset. A web interface is provided via Streamlit for interactive predictions.

## 🎯 Project Overview

The system uses a **Random Forest Regressor** to predict monthly salary based on seven key factors:
- OverTime (Yes/No)
- Age (18-60 years)
- Total Working Years (0-40 years)
- Stock Option Level (0-3)
- Work-Life Balance Rating (1-4)
- Job Role (various categories)
- Education Field (various categories)

## 📁 Repository Structure

```
employee_salary_prediction/
├── models/                      # Saved model and preprocessing artifacts (created after training)
│   ├── income_rf_model.pkl     # Trained Random Forest model
│   ├── income_scaler.pkl       # Feature scaler
│   └── income_label_encoders.pkl # Categorical encoders
├── train_income_model.py        # Training script for income prediction
├── streamlit_income_app.py      # Streamlit web application
├── IBM_Attrition.csv           # Raw dataset (place at project root)
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

## 🔧 Prerequisites

- **Python 3.8** or higher
- **pip** package manager
- **Git** (optional, for cloning)
- Internet access for package installation

## 🚀 Quick Start

### 1. Clone the Repository
```bash
git clone <repository-url>
cd employee_salary_prediction
```

### 2. Create Virtual Environment
```bash
# Windows
python -m venv .venv
.\.venv\Scripts\activate

# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

**Required packages:**
- pandas
- numpy
- scikit-learn
- streamlit
- joblib

### 4. Prepare Dataset
- Place `IBM_Attrition.csv` in the project root directory
- If using a different filename, update references in both Python scripts

### 5. Train the Model
```bash
python train_income_model.py
```

**Expected output:**
```
Test MAE: 1365.95
Test R² : 0.852
Models and preprocessors saved to ./models/
```

### 6. Launch Web Interface
```bash
streamlit run streamlit_income_app.py
```

The app will open in your browser at `http://localhost:8501`

## 💻 Using the Web Interface

1. **Navigate to the form** in your browser
2. **Fill in employee details:**
   - Over Time? (Yes/No dropdown)
   - Age (slider: 18-60)
   - Total Working Years (number input: 0-40)
   - Stock Option Level (dropdown: 0-3)
   - Work-Life Balance Rating (dropdown: 1-4)
   - Job Role (dropdown with available roles)
   - Education Field (dropdown with available fields)
3. **Click "🔮 Predict Monthly Income"**
4. **View the estimated salary** displayed on screen

## 📊 Model Performance

- **Algorithm:** Random Forest Regressor
- **Mean Absolute Error (MAE):** ~$1,366
- **R² Score:** ~0.852 (85.2% variance explained)
- **Training Features:** 7 key employee characteristics
- **Target Variable:** Monthly Income in USD

## 🔄 Retraining the Model

To retrain with new data:

1. Replace `IBM_Attrition.csv` with your updated dataset
2. Ensure column names match the expected format
3. Run: `python train_income_model.py`
4. Restart the Streamlit app: `streamlit run streamlit_income_app.py`

## 🛠️ Troubleshooting

### Common Issues:

**1. Import Errors**
- Ensure virtual environment is activated
- Run `pip install -r requirements.txt`

**2. File Not Found**
- Check that `IBM_Attrition.csv` is in the project root
- Verify filename spelling and case sensitivity

**3. Model Loading Errors**
- Run training script first: `python train_income_model.py`
- Ensure `models/` directory contains all `.pkl` files

**4. Python 3.12+ Build Issues**
- Create `pyproject.toml` with:
```toml
[build-system]
requires = ["setuptools", "wheel"]
build-backend = "setuptools.build_meta"
```

## 📝 Technical Details

### Data Preprocessing:
- Categorical encoding using LabelEncoder
- Numerical feature scaling with MinMaxScaler
- Feature selection based on importance analysis

### Model Training:
- 80/20 train-test split
- Random Forest with 100 estimators
- Cross-validation for robust evaluation

### Deployment:
- Interactive Streamlit interface
- Real-time predictions
- User-friendly form inputs

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is released under the **MIT License**.

---

**Built with ❤️ using Python, Scikit-learn, and Streamlit**

*Last updated: July 2025*