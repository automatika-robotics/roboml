import base64
import logging
import time
from multiprocessing import Process

import cv2
import httpx
import pytest
from roboml.main import ray

HOST = "http://localhost"
PORT = 8000
MODEL_NAME = "idefics"
MODEL_TYPE = "Idefics2"


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
    response = httpx.get(f"{HOST}:{PORT}/")
    assert response.status_code == 200


def test_add_node():
    """
    Test adding model node
    """
    node_params = {"node_name": MODEL_NAME, "node_type": MODEL_TYPE}

    # create node
    response = httpx.post(f"{HOST}:{PORT}/add_node", params=node_params, timeout=30)
    logging.info(response.status_code)
    assert response.status_code == 201


def test_model_init():
    """
    Test initializing model
    """
    # init model with default params
    response = httpx.post(f"{HOST}:{PORT}/{MODEL_NAME}/initialize", timeout=600)
    logging.info(response.status_code)
    assert response.status_code == 200


def test_model_inference():
    """
    Test initializing model
    """

    img = cv2.imread("tests/resources/test.jpeg", cv2.COLOR_BGR2RGB)

    encode_params = [int(cv2.IMWRITE_PNG_COMPRESSION), 9]
    _, buffer = cv2.imencode(".png", img, encode_params)
    img_str = base64.b64encode(buffer).decode("utf-8")

    # call model inference
    body = {"query": "What do you see?", "images": [img_str]}
    response = httpx.post(
        f"{HOST}:{PORT}/{MODEL_NAME}/inference", json=body, timeout=30
    )
    logging.info(response.json())
    assert response.status_code == 200
    assert "output" in response.json()


def test_remove_node():
    """
    Test removing model node
    """
    node_params = {"node_name": MODEL_NAME}

    # remove node
    response = httpx.post(f"{HOST}:{PORT}/remove_node", params=node_params, timeout=30)
    logging.info(response.status_code)
    assert response.status_code == 202
