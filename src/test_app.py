import io
import pandas as pd
import pytest
from app import app

@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def create_excel_file(df):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False)
    output.seek(0)
    return output

def test_index_page(client):
    response = client.get('/')
    assert response.status_code == 200

def test_upload_no_file(client):
    response = client.post('/', data={})
    assert response.status_code == 400

def test_upload_empty_filename(client):
    data = {'file': (io.BytesIO(), '')}
    response = client.post('/', data=data, content_type='multipart/form-data')
    assert response.status_code == 400

def test_upload_no_user_id_column(client):
    df = pd.DataFrame({'wrong_column': [1, 2, 3]})
    excel_file = create_excel_file(df)
    data = {'file': (excel_file, 'test.xlsx')}
    response = client.post('/', data=data, content_type='multipart/form-data')
    assert response.status_code == 400

def test_valid_file_prediction(client):
    df = pd.DataFrame({'user_id': [132, 364]})
    excel_file = create_excel_file(df)
    data = {'file': (excel_file, 'valid.xlsx')}
    response = client.post('/', data=data, content_type='multipart/form-data')
    assert response.status_code == 200
    assert response.headers['Content-Disposition'].startswith('attachment;')
    assert response.mimetype == 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'