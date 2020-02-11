from predictor import Predictor
import logging

logging.basicConfig(level=logging.INFO)

predictor = Predictor()

while True:
    string = input("입력: ")
    response = predictor.predict(string)
    print(response)