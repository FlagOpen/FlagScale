import pytest
from configparser import ConfigParser

def pytest_addoption(parser):
    parser.addoption(
        "--test_reaults_path", action="store", default="none", help="test result path"
    )

@pytest.fixture
def test_reaults_path(request):
    return request.config.getoption("--test_reaults_path")