"""Shared pytest fixtures for the Flask image recognition tests."""
import io
import os

import pytest
from PIL import Image

from app import app as flask_app

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAMPLE_IMAGE = os.path.join(PROJECT_ROOT, "test_images", "3", "Sign 3 (122).jpeg")


@pytest.fixture
def client():
    """Give each test a Flask test client that sends fake browser requests."""
    flask_app.config["TESTING"] = True
    with flask_app.test_client() as test_client:
        yield test_client


@pytest.fixture
def sample_image_path():
    """Path to a real hand-sign photo from the project's test_images folder."""
    return SAMPLE_IMAGE


@pytest.fixture
def make_image():
    """Return a helper that creates a small in-memory image file.

    Building images in memory means the tests do not depend on extra files.
    """
    def _make(size=(64, 64), mode="RGB"):
        buffer = io.BytesIO()
        colour = 128 if mode == "L" else (128, 64, 32)
        Image.new(mode, size, color=colour).save(buffer, format="PNG")
        buffer.seek(0)
        return buffer
    return _make
