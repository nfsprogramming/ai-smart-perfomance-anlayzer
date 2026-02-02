from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from hybrid_model import HybridMentalHealthModel, get_dummy_data

app = Flask(__name__, static_folder='static', template_folder='templates')
CORS(app)

# Initialize and train the model on startup
model = HybridMentalHealthModel()
model.train(get_dummy_data())

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    
    # Extract data from request
    try:
        student_data = {
            'marks': float(data.get('marks', 0)),
            'attendance': float(data.get('attendance', 0)),
            'sleep_hours': float(data.get('sleep_hours', 0)),
            'screen_time': float(data.get('screen_time', 0)),
            'assignment_delay': float(data.get('assignment_delay', 0)),
            'feedback': data.get('feedback', '')
        }
    except ValueError:
        return jsonify({"error": "Invalid input data types"}), 400

    # Get Prediction
    result = model.predict(student_data)
    
    # Get Recommendations
    recommendations = model.get_recommendations(result['risk_index'], student_data)
    
    # Get Explanations (Feature Importance)
    feature_importance = model.get_feature_importance()

    return jsonify({
        "prediction": result,
        "recommendations": recommendations,
        "feature_importance": feature_importance
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)
