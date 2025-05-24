import os
import pickle
import pandas as pd

def load_model():
    possible_paths = [
    'collegeProject/models/xgb_calorie_model.pkl',
    'xgb_calorie_model.pkl',
    'collegeProject/models/E_xgb_calorie_model.pkl'
    ]

    
    for path in possible_paths:
        if os.path.exists(path):
            with open(path, 'rb') as file:
                return pickle.load(file)
    
    raise FileNotFoundError(
        "Could not find XGBoost model file. Tried:\n" +
        "\n".join(f"- {p}" for p in possible_paths)
    )

def predict_calories(gender, age, weight, height, duration, heart_rate, body_temp):
    model = load_model()
    bmi = weight / ((height / 100) ** 2)
    bmi = round(bmi, 2)
    
    input_data = pd.DataFrame({
        'Gender': [gender],
        'Age': [age],
        'BMI': [bmi],
        'Duration': [duration],
        'Heart_Rate': [heart_rate],
        'Body_Temp': [body_temp]
    })
    
    input_data = pd.get_dummies(input_data, drop_first=True)
    
    expected_columns = ['Age', 'BMI', 'Duration', 'Heart_Rate', 'Body_Temp', 'Gender_male']
    for col in expected_columns:
        if col not in input_data.columns:
            input_data[col] = 0
    
    input_data = input_data[expected_columns]
    
    prediction = model.predict(input_data)
    return int(prediction[0])