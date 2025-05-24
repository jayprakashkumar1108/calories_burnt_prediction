from flask import Flask, render_template, request
from datetime import datetime
import pickle
import pandas as pd
from calories_prediction import predict_calories

app = Flask(__name__)

def calculate_health_data(age, gender, height, weight, heart_rate, body_temp, exercise):
    try:
        if gender == 'male':
            bmr = 88.362 + (13.397 * weight) + (4.799 * height) - (5.677 * age)
            ideal_calories = (10 * weight + 6.25 * height - 5 * age + 5) * 1.55
        else:
            bmr = 447.593 + (9.247 * weight) + (3.098 * height) - (4.330 * age)
            ideal_calories = (10 * weight + 6.25 * height - 5 * age - 161) * 1.55
        
        calories = predict_calories(gender, age, weight, height, exercise, heart_rate, body_temp)
        ideal_calories = int(ideal_calories)
        
        calories_status = "good" if 0.9 * ideal_calories <= calories <= 1.1 * ideal_calories else "warning"
        heart_status = "good" if 60 <= heart_rate <= 100 else "warning"
        temp_status = "good" if 36.1 <= body_temp <= 37.2 else "warning"
        
        score = 80  
        if calories_status == "good": score += 10
        if heart_status == "good": score += 5
        if temp_status == "good": score += 5
        score = min(100, score)
        
        return {
            'calories': calories,
            'ideal_calories': ideal_calories,
            'score': score,
            'calories_status': calories_status,
            'heart_status': heart_status,
            'temp_status': temp_status,
            'status': 'success'
        }
    except Exception as e:
        return {'status': 'error', 'message': str(e)}

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        try:
            data = {
                'name': request.form.get('name', 'User'),
                'age': int(request.form.get('age', 25)),
                'gender': request.form.get('gender', 'male'),
                'height': float(request.form.get('height', 170)),
                'weight': float(request.form.get('weight', 70)),
                'heart_rate': int(request.form.get('heart_rate', 72)),
                'body_temp': float(request.form.get('body_temp', 36.6)),
                'exercise': int(request.form.get('exercise', 30))
            }
            
            health_data = calculate_health_data(
                data['age'], data['gender'], data['height'], 
                data['weight'], data['heart_rate'], 
                data['body_temp'], data['exercise']
            )
            
            if health_data['status'] == 'error':
                return render_template('predict.html', error=health_data['message'])
            
            report_data = {
                **data,
                **health_data,
                'date': datetime.now().strftime("%B %d, %Y"),
                'ideal_calories_min': int(health_data['ideal_calories'] * 0.9),
                'ideal_calories_max': int(health_data['ideal_calories'] * 1.1),
                'calories_percent': min(100, (health_data['calories'] / (health_data['ideal_calories'] * 1.5)) * 100),
                'heart_percent': min(100, (data['heart_rate'] / 120) * 100),
                'temp_percent': min(100, ((data['body_temp'] - 35) / (42 - 35)) * 100),
                'ideal_calories_percent': 50  
            }
            
            return render_template('report.html', **report_data)
            
        except Exception as e:
            return render_template('predict.html', error=f"Invalid input: {str(e)}")
    
    return render_template('predict.html')

if __name__ == '__main__':
    app.run(debug=True)