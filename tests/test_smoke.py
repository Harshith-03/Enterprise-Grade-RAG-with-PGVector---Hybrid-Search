from app.main import app


def test_fastapi_app_metadata() -> None:
    assert app.title == "Enterprise RAG"
    assert app.version == "0.1.0"
