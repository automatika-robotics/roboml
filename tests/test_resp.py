import msgpack_numpy as m_pack
import msgpack
import pytest
from multiprocessing import Process, set_start_method
from redis import Redis
import time
import logging

from roboml.main import resp

# patch msgpack for numpy
m_pack.patch()


MODEL_TYPE = "SpeechT5"
MODEL_NAME = "test"


@pytest.fixture(scope="module", autouse=True)
def run_before_and_after_tests():
    """Fixture to run resp server before tests are run"""

    # start server
    set_start_method("spawn", force=True)
    p = Process(target=resp)
    p.start()

    # give it 10 seconds to start before sending request
    time.sleep(10)

    yield

    # terminate server process
    p.terminate()


def test_resp_connect():
    """
    Test resp server connection
    """
    r = Redis("localhost", port=6379)
    response = r.execute_command(b"PING")
    assert response == b"PONG"


def test_add_node():
    """
    Test adding model node
    """
    node_params = {"node_name": MODEL_NAME, "node_model": MODEL_TYPE}

    # create node
    params = msgpack.packb(node_params)
    r = Redis("localhost", port=6379)
    response = r.execute_command("add_node", params)
    response = msgpack.unpackb(response)
    logging.info(response)
    assert f"{MODEL_NAME}.initialize" in response


def test_model_init():
    """
    Test initializing model
    """
    # init model with default params
    r = Redis("localhost", port=6379)
    response = r.execute_command(f"{MODEL_NAME}.initialize")
    logging.info(response)
    assert response == b"OK"


def test_model_inference():
    """
    Test initializing model
    """
    # call model inference
    r = Redis("localhost", port=6379)
    body = {"query": "This should be spoken out loud"}
    body = msgpack.packb(body)
    response = r.execute_command(f"{MODEL_NAME}.inference", body)
    response = msgpack.unpackb(response)
    assert "output" in response


def test_remove_node():
    """
    Test removing model node
    """
    node_params = {"node_name": MODEL_NAME}

    # remove node
    params = msgpack.packb(node_params)
    r = Redis("localhost", port=6379)
    response = r.execute_command("remove_node", params)
    response = msgpack.unpackb(response)
    logging.info(response)
    assert f"{MODEL_NAME}.initialize" not in response
