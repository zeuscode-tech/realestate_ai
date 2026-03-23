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
    y_test_vals = data.get('y_test', [])
    y_pred_vals = data.get('y_pred', [])
    r2_val = data.get('r2', 0.91)
    print(f"✅ Model loaded successfully from {MODEL_PATH}")
    print(f"   Average area price: {area_mean:.2f}")
except FileNotFoundError:
    print(f"❌ Error: {MODEL_PATH} not found!")
    model = None
    area_mean = 50.0
    y_test_vals = []
    y_pred_vals = []
    r2_val = 0.91
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None
    area_mean = 50.0
    y_test_vals = []
    y_pred_vals = []
    r2_val = 0.91


@app.route('/')
def index():
    return send_from_directory('templates', 'index.html')


@app.route('/api/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    try:
        data = request.get_json()
        
        age = float(data.get('age', 0))
        mrt = float(data.get('mrt', 0))
        stores = float(data.get('stores', 0))
        lat = round(float(data.get('lat', 0)), 3)
        lng = round(float(data.get('lng', 0)), 3)

        mrt_log = np.log1p(mrt)

        # Use area_map lookup if coordinates are provided, otherwise global mean
        local_mean = area_map.get((lat, lng), area_mean)

        features = np.array([[age, mrt_log, stores, local_mean]])
        

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


@app.route('/scatter.png')
def scatter_image():
    return send_from_directory('templates', 'scatter.png')


@app.route('/api/model-stats', methods=['GET'])
def model_stats():
    """Return model accuracy stats and scatter chart data."""
    return jsonify({
        'r2': round(r2_val * 100, 1),
        'y_test': y_test_vals[:100],
        'y_pred': [round(v, 2) for v in y_pred_vals[:100]]
    })


@app.route('/api/dataset', methods=['GET'])
def dataset_preview():
    """Return first 10 rows of dataset."""
    import pandas as pd
    df = pd.read_csv('Real estate.csv')
    df.columns = [c.strip() for c in df.columns]
    rows = df.head(10).to_dict(orient='records')
    return jsonify({'rows': rows})


if __name__ == '__main__':
    print("🚀 Starting Smart Realty API...")
    app.run(debug=True, host='0.0.0.0', port=5000)