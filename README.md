# Hybrid AI Model for Student Mental Health Prediction

This project implements a Hybrid AI Model that combines **Machine Learning (Random Forest)** and **Natural Language Processing (NLP)** to predict student mental health risk.

## 📂 Project Structure
- `hybrid_model.py`: The main Python script containing the Step-by-Step implementation.
- `requirements.txt`: List of required Python libraries.

## 🚀 How to Run

1. **Install Dependencies** (if not already installed):
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the Model**:
   ```bash
   python hybrid_model.py
   ```

## 🧠 How it Works
1. **Data Collection**: Uses dummy data (Marks, Attendance, Sleep, Feedback).
2. **Preprocessing**: Cleans text and handles missing values.
3. **Feature Engineering**: 
   - Numeric data -> Random Forest
   - Text data -> TF-IDF -> Logistic Regression
4. **Ensemble Learning**: Combines predictions from both models (70% ML + 30% NLP).
5. **Explainability**: Shows which factors (e.g., Sleep, Attendance) contributed most to the risk.
