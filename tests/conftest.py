import pytest


@pytest.fixture(autouse=True)
def before_each_test():
    pass


@pytest.fixture(scope="session", autouse=True)
def after_all_tests():
    yield
    print('\n\nAll tests done.')
