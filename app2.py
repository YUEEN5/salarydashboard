from flask import Flask, request, jsonify
import pandas as pd
from catboost import CatBoostRegressor
from waitress import serve  # For production deployment

app = Flask(__name__)

# Load CatBoost model
try:
    model = CatBoostRegressor()
    model.load_model('model/catboost_salary_model2.cbm')
    print("✅ Model loaded successfully")
except Exception as e:
    print(f"❌ Error loading model: {str(e)}")
    raise e

@app.route('/predict', methods=['POST'])
def predict():
    try:
        input_data = request.json

        required_fields = ['category', 'role', 'location', 'type']
        if not all(field in input_data for field in required_fields):
            return jsonify({'error': 'Missing required fields'}), 400

        # Prepare input
        sample_input = pd.DataFrame([{
            'category': input_data['category'], 
            'role': input_data['role'],
            'location': input_data['location'],
            'type': input_data['type']
        }])

        # Predict
        predicted_avg = model.predict(sample_input)[0]

        response = {
            'min_salary': round(predicted_avg * 0.85, 2),
            'mean_salary': round(predicted_avg, 2),
            'max_salary': round(predicted_avg * 1.15, 2)
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

    # For production (uncomment this)
    # serve(app, host='0.0.0.0', port=5000)
