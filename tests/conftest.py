import pytest

from pingouin_app import create_app


@pytest.fixture(scope="session")
def app(request):
    app = create_app()

    ctx = app.app_context()
    ctx.push()

    def teardown():
        ctx.pop()

    request.addfinalizer(teardown)
    return app


@pytest.fixture(scope="function")
def client(app):
    return app.test_client()
