from flask import Flask, request, jsonify, render_template
import joblib

app = Flask(__name__)
model = joblib.load('model.pkl')  # Updated path to the model file

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    try:
        features = [
            float(data['feature1']),
            float(data['feature2']),
            float(data['feature3']),
            float(data['feature4']),
            float(data['feature5']),
            float(data['feature6']),
            float(data['feature7']),
            float(data['feature8']),
            float(data['feature9']),
            float(data['feature10']),
            float(data['feature11']),
            float(data['feature12']),
            float(data['feature13']),
            float(data['feature14']),
            float(data['feature15']),
            float(data['feature16']),
            float(data['feature17']),
            float(data['feature18']),
            float(data['feature19']),
            float(data['feature20']),
            float(data['feature21']),
            float(data['feature22']),
            float(data['feature23']),
            float(data['feature24']),
            float(data['feature25']),
            float(data['feature26']),
            float(data['feature27']),
            float(data['feature28']),
            float(data['feature29']),
            float(data['feature30'])
        ]  # Convert to float

        print(f"Received features: {features}")
        prediction = model.predict([features])
        print(f"Model prediction: {prediction}")
        if prediction[0] == 1:
            return jsonify({"result": "fraudulent"})
        else:
            return jsonify({"result": "safe"})

    except (ValueError, KeyError) as e:
        print(f"Error: {str(e)}")  # Debugging error
        return jsonify({"alert": "Invalid input. Please ensure all inputs are numerical."})

if __name__ == '__main__':
    app.run(debug=True)
