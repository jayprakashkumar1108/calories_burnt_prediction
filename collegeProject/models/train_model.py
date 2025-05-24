import xgboost as xgb
import pandas as pd
import pickle
import os
from sklearn.model_selection import train_test_split

def train_and_save_model():
    os.makedirs('models', exist_ok=True)
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    calories_path = os.path.join(base_dir, 'data', 'calories.csv')
    exercise_path = os.path.join(base_dir, 'data', 'exercise.csv')

    try:
        calories = pd.read_csv(calories_path)
        exercise = pd.read_csv(exercise_path)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please make sure:")
        print(f"1. calories.csv exists at {calories_path}")
        print(f"2. exercise.csv exists at {exercise_path}")
        return

    exercise_df = exercise.merge(calories, on='User_ID')
    exercise_train_data, exercise_test_data = train_test_split(exercise_df, test_size=0.2, random_state=1)

    for data in [exercise_train_data, exercise_test_data]:
        data['BMI'] = data['Weight'] / ((data['Height'] / 100) ** 2)
        data['BMI'] = round(data['BMI'], 2)

    exercise_train_data = exercise_train_data[['Gender', 'Age', 'BMI', 'Duration', 'Heart_Rate', 'Body_Temp', 'Calories']]
    exercise_test_data = exercise_test_data[['Gender', 'Age', 'BMI', 'Duration', 'Heart_Rate', 'Body_Temp', 'Calories']]

    exercise_train_data = pd.get_dummies(exercise_train_data, drop_first=True)
    exercise_test_data = pd.get_dummies(exercise_test_data, drop_first=True)

    X_train = exercise_train_data.drop('Calories', axis=1)
    y_train = exercise_train_data['Calories']
    X_test = exercise_test_data.drop('Calories', axis=1)
    y_test = exercise_test_data['Calories']

    xgb_model = xgb.XGBRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        random_state=42
    )

    xgb_model.fit(X_train, y_train)

    model_path = os.path.join(base_dir, 'models', 'xgb_calorie_model.pkl')
    with open(model_path, 'wb') as file:
        pickle.dump(xgb_model, file)
    
    print(f"Model successfully saved to {model_path}")

if __name__ == '__main__':
    train_and_save_model()