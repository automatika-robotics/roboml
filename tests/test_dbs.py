import inspect
import shutil
import logging

import pytest
from roboml import databases
from roboml.databases._base import VectorDBTemplate
from roboml.interfaces import DBAdd, DBQuery, DBMetadataQuery


dbs = []


@pytest.fixture(scope="module", autouse=True)
def run_before_and_after_tests():
    """Fixture to run delete test db data"""

    global dbs
    dbs = [
        db_class(name="test")
        for _, db_class in inspect.getmembers(databases, predicate=inspect.isclass)
        if issubclass(db_class, VectorDBTemplate) and db_class is not VectorDBTemplate
    ]
    # initialize all dbs
    for db in dbs:
        db.initialize(db_location="tests/db_data")

    yield

    # delete data folder
    shutil.rmtree("tests/db_data")
    dbs = []


@pytest.fixture
def data():
    return {
        "ids": ["a"],
        "metadatas": [{"something": "about a"}],
        "documents": ["description of a"],
        "collection_name": "alphabets",
    }


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_add_data(data):
    """
    Test add
    """
    global dbs
    data = DBAdd(**data, reset_collection=True)
    for db in dbs:
        db._add(data)


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_conditional_add(data):
    """
    Test conditional add
    """
    global dbs
    data = DBAdd(**data)
    for db in dbs:
        db._conditional_add(data)


def test_metadata_query(data):
    """
    Test conditional add
    """
    global dbs
    data = DBMetadataQuery(
        metadatas=data["metadatas"], collection_name=data["collection_name"]
    )
    for db in dbs:
        result = db._metadata_query(data)
        logging.info(result)
        assert result["output"]["documents"][0] == "description of a"


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_query(data):
    """
    Test conditional add
    """
    global dbs
    data = DBQuery(collection_name=data["collection_name"], query="what is a")
    for db in dbs:
        result = db._query(data)
        logging.info(result)
        assert result["output"]["documents"][0][0] == "description of a"
