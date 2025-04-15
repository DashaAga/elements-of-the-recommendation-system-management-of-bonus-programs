import uuid
import tempfile
import pandas as pd 
from flask import Flask, request, send_file 
from model import Model 
from data_loader import DataLoader 
from preprocessing import DataPreprocessor 
from feature_engineering import FeatureEngineering 
from pathlib import Path 

BASE_DIR = Path(__file__).resolve().parent.parent 
 
data_path = BASE_DIR / "data" / "data.csv" 
model_path = BASE_DIR / "models" / "model_01_04_2025.pkl" 
unique_filename = f"result_{uuid.uuid4().hex}.xlsx"
temp_result_path = tempfile.gettempdir() + '/' + unique_filename
 
app = Flask(__name__) 

loaded_gb = Model.load_model(model_path) 

loaded_model = Model({}) 
loaded_model.model = loaded_gb
 
@app.route('/') 
def index(): 
    return ''' 
        <h1>Загрузите Excel файл с ID пользователей</h1> 
        <form method="POST" enctype="multipart/form-data"> 
            <input type="file" name="file" /> 
            <input type="submit" value="Загрузить" /> 
        </form> 
    ''' 
 
@app.route('/', methods=['POST']) 
def upload_file(): 
    if 'file' not in request.files: 
        return "Нет файла для загрузки", 400
    file = request.files['file'] 
    if file.filename == '': 
        return "Нет выбранного файла", 400

    try: 
        df = pd.read_excel(file) 
    except Exception as e: 
        return f"Ошибка при загрузке файла: {e}" 

    if 'user_id' not in df.columns: 
        return "Нет столбца с ID пользователей (user_id)" , 400

    for column in df.columns:
        if not pd.api.types.is_numeric_dtype(df[column]):
            try:
                df[column] = pd.to_numeric(df[column], errors='coerce')  # Преобразует в NaN, если не число
            except ValueError:
                return f"Невозможно преобразовать столбец {column} в числовой формат", 400

    user_ids = df['user_id'] 
 
    # если фичи получаем из хранилища:
 
    # data_loader = DataLoader() 
 
    # Format the ids as a comma-separated string for the SQL query 
    # ids_str = ",".join([f"{id_}" for id_ in user_ids]) 
 
    # sql_customers_features = f""" 
    # SELECT * 
    # FROM dbo_mb.feature_store 
    # WHERE customer_mindbox_id IN ({ids_str}) 
    # """ 
 
    all_customers = pd.read_csv(data_path) 
    all_customers = all_customers.iloc[:,1:] 
    customers_features = all_customers[all_customers['customer_mindbox_id'].isin(user_ids)] 

    feature_engineer = FeatureEngineering(customers_features)

    df_with_features = feature_engineer.generate_features()

    columns_to_drop = [
    "bonus_accrual", 
    "purchase_sum_restore",
    "days_since_last_redemption",
    "purchase_frequency_last_year",
    "purchase_iphone",
     "purchase_count_iphone"
]

    df_with_features = feature_engineer.clean_data(columns_to_drop=columns_to_drop)

    preprocessor = DataPreprocessor(fill_strategy="zero", scale_method="standard")

    df_cleaned = preprocessor.preprocess(df_with_features)

    df_cleaned = df_cleaned.drop(columns=["customer_mindbox_id", "target"], errors="ignore") 
 
    predictions = loaded_model.predict(df_cleaned) 
 
    predictions.to_excel(temp_result_path, index=True)
 
    return send_file(temp_result_path, as_attachment=True), 200
 
if __name__ == '__main__': 
    app.run(debug=True)