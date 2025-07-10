from flask import Flask, request, jsonify
import pandas as pd
import joblib
from sklearn.pipeline import Pipeline
from waitress import serve  # For production deployment

app = Flask(__name__)

# Load the trained model
try:
    model = joblib.load('salary_predictor_model.pkl')
    print("✅ Model loaded successfully")
except Exception as e:
    print(f"❌ Error loading model: {str(e)}")
    raise e

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input data from request
        input_data = request.json
        
        # Validate input
        required_fields = ['job_title', 'category', 'role', 'location', 'type']
        if not all(field in input_data for field in required_fields):
            return jsonify({'error': 'Missing required fields'}), 400
        
        # Create DataFrame for prediction
        sample_input = pd.DataFrame([{
            'job_title': input_data['job_title'],
            'category': input_data['category'],
            'role': input_data['role'],
            'location': input_data['location'],
            'type': input_data['type']
        }])
        
        # Make prediction
        predicted_avg = model.predict(sample_input)[0]
        
        # Create response (you can adjust the multipliers based on your data distribution)
        response = {
            'min_salary': predicted_avg * 0.85,  # Example: 15% below average
            'mean_salary': predicted_avg,
            'max_salary': predicted_avg * 1.15   # Example: 15% above average
        }
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy'})

if __name__ == '__main__':
    # For development
    app.run(host='0.0.0.0', port=5000, debug=True)
    
    # For production (recommended):
    # serve(app, host='0.0.0.0', port=5000)
