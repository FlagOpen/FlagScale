from configparser import ConfigParser

import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--test_path",
        action="store",
        default="none",
        help="Base path for the test cases",
    )
    parser.addoption(
        "--test_type",
        action="store",
        default="none",
        help="Different Types of Testing (train/inference/....)",
    )
    parser.addoption(
        "--test_model",
        action="store",
        default="none",
        help="Model name for the test cases",
    )
    parser.addoption(
        "--test_case", action="store", default="none", help="Specific test case to run"
    )


@pytest.fixture
def test_path(request):
    return request.config.getoption("--test_path")


@pytest.fixture
def test_type(request):
    return request.config.getoption("--test_type")


@pytest.fixture
def test_model(request):
    return request.config.getoption("--test_model")


@pytest.fixture
def test_case(request):
    return request.config.getoption("--test_case")
