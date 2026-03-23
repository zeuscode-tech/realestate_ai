from flask import Flask, request, jsonify, send_from_directory
import pickle
import os
import numpy as np

app = Flask(__name__, static_folder='templates')

MODEL_PATH = 're_model.pkl'

try:
    with open(MODEL_PATH, 'rb') as f:
        data = pickle.load(f)
    model = data['model']
    area_map = data.get('area_map', {})
    area_mean = np.mean(list(area_map.values())) if area_map else 50.0
    print(f"✅ Model loaded successfully from {MODEL_PATH}")
    print(f"   Average area price: {area_mean:.2f}")
except FileNotFoundError:
    print(f"❌ Error: {MODEL_PATH} not found!")
    model = None
    area_mean = 50.0
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None
    area_mean = 50.0


@app.route('/')
def index():
    return send_from_directory('templates', 'index.html')


@app.route('/api/predict', methods=['POST'])
def predict():
    """
    Predict house price based on input parameters.
    
    Expected JSON:
    {
        "age": float,
        "mrt": float,
        "stores": float
    }
    """
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    try:
        data = request.get_json()
        
        age = float(data.get('age', 0))
        mrt = float(data.get('mrt', 0))
        stores = float(data.get('stores', 0))
        
        
        mrt_log = np.log1p(mrt)
        
        # Use global area_mean calculated from training data
        features = np.array([[age, mrt_log, stores, area_mean]])
        
        
        prediction = model.predict(features)[0]
        price = prediction
        
        return jsonify({
            'price': round(price, 2),
            'age': age,
            'mrt': mrt,
            'stores': stores
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400


@app.route('/api/chart-data', methods=['POST'])
def chart_data():
    """
    Generate chart data showing price variation with MRT distance.
    
    Expected JSON:
    {
        "age": float,
        "stores": float
    }
    """
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    try:
        data = request.get_json()
        
        age = float(data.get('age', 0))
        stores = float(data.get('stores', 0))
        
        distances = list(range(100, 3100, 200))
        prices = []
        
        for mrt in distances:
            mrt_log = np.log1p(mrt)
            features = np.array([[age, mrt_log, stores, area_mean]])
            prediction = model.predict(features)[0]
            price = prediction
            prices.append(round(price, 2))
        
        return jsonify({
            'distances': distances,
            'prices': prices
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400


if __name__ == '__main__':
    print("🚀 Starting Smart Realty API...")
    app.run(debug=True, host='0.0.0.0', port=5000)
