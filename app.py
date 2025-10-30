from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)

model = joblib.load('models/airquality_model.pkl')
scaler = joblib.load('models/scaler.pkl')
features = joblib.load('models/features.pkl')

def classify_aqi(aqi_value):
    if aqi_value <= 50:
        return {'category': 'Good', 'color': '#00e400',
                'description': 'Air quality is satisfactory.',
                'recommendation': 'Enjoy outdoor activities!'}
    elif aqi_value <= 100:
        return {'category': 'Moderate', 'color': '#ffff00',
                'description': 'Acceptable air quality.',
                'recommendation': 'Sensitive people should limit prolonged outdoor exertion.'}
    elif aqi_value <= 150:
        return {'category': 'Unhealthy for Sensitive Groups', 'color': '#ff7e00',
                'description': 'Sensitive groups may experience health effects.',
                'recommendation': 'Children and elderly should reduce outdoor activities.'}
    elif aqi_value <= 200:
        return {'category': 'Unhealthy', 'color': '#ff0000',
                'description': 'Everyone may experience health effects.',
                'recommendation': 'Everyone should reduce outdoor exertion.'}
    elif aqi_value <= 300:
        return {'category': 'Very Unhealthy', 'color': '#8f3f97',
                'description': 'Health alert for everyone.',
                'recommendation': 'Avoid outdoor activities. Wear masks.'}
    else:
        return {'category': 'Hazardous', 'color': '#7e0023',
                'description': 'Emergency conditions.',
                'recommendation': 'Stay indoors. Use air purifiers.'}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        input_features = [float(request.form[f]) for f in [
            'CO', 'PT08_S1', 'NMHC', 'C6H6', 'PT08_S2', 'NOx', 'PT08_S3',
            'NO2', 'PT08_S4', 'PT08_S5', 'T', 'RH', 'AH', 'Hour', 'Day', 'Month'
        ]]
        
        input_df = pd.DataFrame([input_features], columns=features)
        input_scaled = scaler.transform(input_df)
        prediction = model.predict(input_scaled)[0]
        aqi_info = classify_aqi(prediction)
        
        return render_template('result.html', aqi=round(prediction, 2),
                             category=aqi_info['category'], color=aqi_info['color'],
                             description=aqi_info['description'],
                             recommendation=aqi_info['recommendation'])
    except Exception as e:
        return render_template('index.html', error=str(e))

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
