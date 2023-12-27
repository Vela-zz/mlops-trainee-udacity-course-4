import json

from fastapi.testclient import TestClient

from main import app

client = TestClient(app)


def test_read_main():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Welcome to the Udacity MLops L4 model inference API!"}  # NOQA: E501


def test_predict():
    response = client.post("/predict",
                           json={
                                "age": 39,
                                "workclass": "State-gov",
                                "fnlgt": 77516,
                                "education": "Bachelors",
                                "education-num": 13,
                                "marital-status": "Never-married",
                                "occupation": "Adm-clerical",
                                "relationship": "Not-in-family",
                                "race": "White",
                                "sex": "Male",
                                "capital-gain": 2174,
                                "hours-per-week": 40,
                                "native-country": "United-States"
                            })
    assert response.status_code == 200
    assert json.loads(response.content.decode())['pred'] in ['<=50K', '>50K']


def test_predict_with_invalid_data():
    invalid_example = {
                "age": 40,
                "workclass": "Nothing",
                "fnlgt": 77516,
                "education": "Bachelors",
                "education-num": 13,
                "marital-status": "Never-married",
                "occupation": "Adm-clerical",
                "relationship": "Not-in-family",
                "race": "Colorful",
                "sex": "Helicopter",
                "capital-gain": 2174,
                "hours-per-week": 40,
                "native-country": "United-States"
            }
    response = client.post("/predict", json=invalid_example)
    assert response.status_code != 200
