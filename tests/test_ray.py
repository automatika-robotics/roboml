import logging
import time
from multiprocessing import Process
from websockets.sync.client import connect
import msgpack

import httpx
import pytest
from roboml.main import ray

HOST = "localhost"
PORT = 8000
MODEL_NAME = "test"
MODEL_TYPE = "TransformersLLM"


@pytest.fixture(scope="module", autouse=True)
def run_before_and_after_tests():
    """Fixture to run ray before tests are run"""

    # start server
    p = Process(target=ray)
    p.start()

    # give it 20 seconds to start before sending request
    time.sleep(20)

    yield

    # terminate server process - kill to remove ray monitoring child
    p.kill()


def test_ray_connect():
    """
    Test ray app factory connection
    """
    response = httpx.get(f"http://{HOST}:{PORT}/")
    assert response.status_code == 200


def test_add_node():
    """
    Test adding model node
    """
    node_params = {
        "node_name": MODEL_NAME,
        "node_model": MODEL_TYPE,
    }

    # create node
    response = httpx.post(
        f"http://{HOST}:{PORT}/add_node", params=node_params, timeout=100
    )
    logging.info(response.status_code)
    assert response.status_code == 201


def test_model_init():
    """
    Test initializing model
    """
    # init model with default params
    response = httpx.post(f"http://{HOST}:{PORT}/{MODEL_NAME}/initialize", timeout=600)
    logging.info(response.status_code)
    assert response.status_code == 200


def test_model_inference():
    """
    Test model inference over http endpoint
    """

    # call model inference
    body = {"query": [{"role": "user", "content": "Whats up?"}]}
    response = httpx.post(
        f"http://{HOST}:{PORT}/{MODEL_NAME}/inference", json=body, timeout=30
    )
    for chunk in response.iter_text(chunk_size=None):
        logging.info(chunk)
    assert response.status_code == 200


def test_ws_model_inference():
    """
    Test model inference over websocket endpoint
    """

    # call model inference
    with connect(f"ws://{HOST}:{PORT}/{MODEL_NAME}/ws_inference") as websocket:
        message = msgpack.packb({
            "query": [{"role": "user", "content": "Space the final"}],
            "stream": True,
        })
        websocket.send(message)
        while True:
            received = websocket.recv()
            if received == "<<Response Ended>>":
                break
            logging.info(received)

    assert received == "<<Response Ended>>"


def test_remove_node():
    """
    Test removing model node
    """
    node_params = {"node_name": MODEL_NAME}

    # remove node
    response = httpx.post(
        f"http://{HOST}:{PORT}/remove_node", params=node_params, timeout=30
    )
    logging.info(response.status_code)
    assert response.status_code == 202
