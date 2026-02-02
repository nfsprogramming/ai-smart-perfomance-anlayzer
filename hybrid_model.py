import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

class HybridMentalHealthModel:
    def __init__(self):
        self.rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.nlp_model = LogisticRegression()
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.feature_cols = ['marks', 'attendance', 'sleep_hours', 'screen_time', 'assignment_delay']
        self.risk_levels = {0: "LOW RISK 🟢", 1: "MEDIUM RISK �", 2: "HIGH RISK 🔴"}

    def clean_text(self, text):
        text = text.lower()                 # Convert to lowercase
        text = re.sub(r'[^\w\s]', '', text) # Remove punctuation
        return text

    def train(self, data):
        df = pd.DataFrame(data)
        
        # Text Preprocessing
        df['clean_feedback'] = df['feedback'].apply(self.clean_text)
        
        # Features
        X_tabular = df[self.feature_cols]
        y = df['risk_label']
        
        # Text Features
        X_text = self.vectorizer.fit_transform(df['clean_feedback']).toarray()
        
        # Train Models
        self.rf_model.fit(X_tabular, y)
        self.nlp_model.fit(X_text, y)
        
        return "Models Trained Successfully"

    def predict(self, student_data):
        # Prepare Tabular Input
        new_tab_df = pd.DataFrame([[
            student_data['marks'], 
            student_data['attendance'], 
            student_data['sleep_hours'], 
            student_data['screen_time'], 
            student_data['assignment_delay']
        ]], columns=self.feature_cols)

        # Prepare Text Input
        clean_feedback = self.clean_text(student_data['feedback'])
        new_text_vec = self.vectorizer.transform([clean_feedback]).toarray()

        # Get Probabilities
        rf_probs = self.rf_model.predict_proba(new_tab_df)[0]
        nlp_probs = self.nlp_model.predict_proba(new_text_vec)[0]

        # Weighted Ensemble
        weight_rf = 0.7
        weight_nlp = 0.3
        final_probs = (weight_rf * rf_probs) + (weight_nlp * nlp_probs)
        final_prediction_index = np.argmax(final_probs)
        
        predicted_risk = self.risk_levels[final_prediction_index]

        return {
            "rf_probs": rf_probs.tolist(),
            "nlp_probs": nlp_probs.tolist(),
            "final_probs": final_probs.tolist(),
            "risk_index": int(final_prediction_index),
            "risk_label": predicted_risk
        }

    def get_feature_importance(self):
        importances = self.rf_model.feature_importances_
        indices = np.argsort(importances)[::-1]
        feature_importance = {}
        for i in range(len(self.feature_cols)):
            feature_importance[self.feature_cols[indices[i]]] = importances[indices[i]]
        return feature_importance

    def get_recommendations(self, risk_index, student_data):
        recommendations = []
        if risk_index == 2: # High Risk
            recommendations.append("Risk Level is HIGH. Immediate attention needed.")
            if student_data['sleep_hours'] < 6:
                recommendations.append("Increase sleep hours (target: >7h)")
            if student_data['screen_time'] > 5:
                recommendations.append("Reduce screen time (target: <4h)")
            if "stress" in student_data['feedback'].lower():
                recommendations.append("Practice stress management or meditation.")
            recommendations.append("Schedule a meeting with the mentor.")
        elif risk_index == 1: # Medium Risk
            recommendations.append("Risk Level is MEDIUM. Keep an eye on health.")
            recommendations.append("Try to balance study and rest.")
        else:
            recommendations.append("Risk Level is LOW. Keep up the good work!")
        return recommendations

# Default Dummy Data for Training
def get_dummy_data():
    return {
        'marks':            [85, 45, 65, 90, 30, 75, 55, 95, 40, 60],
        'attendance':       [90, 50, 70, 95, 40, 80, 60, 98, 45, 65],
        'sleep_hours':      [7,  4,  6,  8,  3,  7,  5,  8,  4,  6],
        'screen_time':      [3,  9,  6,  2,  10, 4,  7,  2,  9,  6],
        'assignment_delay': [0,  5,  2,  0,  6,  1,  3,  0,  5,  2],
        'feedback': [
            "I feel happy and excited about learning.",
            "I am very stressed and cannot sleep.",
            "I feel okay but a bit tired.",
            "Great motivation and energy!",
            "Depressed and feeling lonely.",
            "Good balance, feeling fine.",
            "Anxious about exams.",
            "Loving the course, feeling great.",
            "Too much pressure, I want to quit.",
            "Confused but trying to cope."
        ],
        'risk_label': [0, 2, 1, 0, 2, 0, 1, 0, 2, 1]
    }


