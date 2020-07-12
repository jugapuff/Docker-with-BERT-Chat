from grpc_wrapper.server import create_server, BaseModel
import time
from predictor import Predictor
import json


class YourModel(BaseModel):
    def __init__(self):
        print("init")
        self.predictor = Predictor()

    def send(self, input):
        result = self.predictor.predict(input["query"])
        return output


def run():

    model = YourModel()
    server = create_server(model, ip="[::]", port=50051, max_workers=5)
    server.start()
    try:
        while True:
            time.sleep(60 * 60 * 24)
    except KeyboardInterrupt:
        server.stop(0)


if __name__ == "__main__":

    run()
