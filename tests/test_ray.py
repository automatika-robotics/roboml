import logging
import sys
import time
from multiprocessing import Process
from websockets.sync.client import connect
import msgpack

import httpx
import pytest

HOST = "localhost"
PORT = 8000
MODEL_NAME = "test"
MODEL_TYPE = "TransformersLLM"


def _start_ray_server():
    """Start ray server in a subprocess with clean sys.argv."""
    sys.argv = ["roboml"]
    from roboml.main import ray

    ray()


@pytest.fixture(scope="module", autouse=True)
def run_before_and_after_tests():
    """Fixture to run ray before tests are run"""

    # start server
    p = Process(target=_start_ray_server)
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


def test_list_models():
    """
    Test OpenAI-compatible /v1/models endpoint
    """
    response = httpx.get(f"http://{HOST}:{PORT}/v1/models", timeout=10)
    assert response.status_code == 200
    data = response.json()
    assert data["object"] == "list"
    assert any(m["id"] == MODEL_NAME for m in data["data"])


def test_chat_completions():
    """
    Test OpenAI-compatible /v1/chat/completions endpoint
    """
    body = {
        "messages": [{"role": "user", "content": "Say hello in one word."}],
        "max_tokens": 20,
        "temperature": 0.7,
    }
    response = httpx.post(
        f"http://{HOST}:{PORT}/{MODEL_NAME}/v1/chat/completions",
        json=body,
        timeout=30,
    )
    assert response.status_code == 200
    data = response.json()
    assert data["object"] == "chat.completion"
    assert len(data["choices"]) == 1
    assert data["choices"][0]["message"]["role"] == "assistant"
    assert len(data["choices"][0]["message"]["content"]) > 0
    logging.info(data["choices"][0]["message"]["content"])


def test_chat_completions_streaming():
    """
    Test OpenAI-compatible streaming via SSE
    """
    body = {
        "messages": [{"role": "user", "content": "Count to 3."}],
        "max_tokens": 30,
        "stream": True,
    }
    with httpx.stream(
        "POST",
        f"http://{HOST}:{PORT}/{MODEL_NAME}/v1/chat/completions",
        json=body,
        timeout=30,
    ) as response:
        assert response.status_code == 200
        chunks = []
        for line in response.iter_lines():
            if line.startswith("data: "):
                payload = line[6:]
                if payload == "[DONE]":
                    break
                chunks.append(payload)

    assert len(chunks) > 1  # at least role chunk + content chunks
    logging.info(f"Received {len(chunks)} SSE chunks")


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
