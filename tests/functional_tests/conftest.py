import pytest
from configparser import ConfigParser

def pytest_addoption(parser):
    parser.addoption(
        "--test_path", action="store", default="none", help="test path"
    )

@pytest.fixture
def test_path(request):
    return request.config.getoption("--test_path")