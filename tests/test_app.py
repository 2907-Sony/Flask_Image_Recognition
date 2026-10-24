"""Tests for the Flask routes in app.py."""
import io
from unittest.mock import patch

ERROR_MESSAGE = b"File cannot be processed."


# Normal cases

def test_home_page_loads(client):
    """The home page returns 200 OK and shows the upload form."""
    response = client.get("/")
    assert response.status_code == 200
    assert b"Hand Sign Digit Language Detection" in response.data
    assert b'name="file"' in response.data


@patch("app.predict_result", return_value=7)
def test_valid_image_shows_prediction(mock_predict, client, sample_image_path):
    """A valid image upload renders the predicted digit (model mocked for speed)."""
    with open(sample_image_path, "rb") as image_file:
        response = client.post(
            "/prediction",
            data={"file": (image_file, "sign3.jpeg")},
            content_type="multipart/form-data",
        )
    assert response.status_code == 200
    assert b"7" in response.data
    assert ERROR_MESSAGE not in response.data
    # The route must pass a correctly shaped image to the model
    sent_image = mock_predict.call_args[0][0]
    assert sent_image.shape == (1, 224, 224, 3)


def test_real_model_end_to_end(client, sample_image_path):
    """Full flow with the real model: a 'Sign 3' photo is predicted as 3."""
    with open(sample_image_path, "rb") as image_file:
        response = client.post(
            "/prediction",
            data={"file": (image_file, "sign3.jpeg")},
            content_type="multipart/form-data",
        )
    assert response.status_code == 200
    assert b">3<" in response.data


#  Missing file

def test_missing_file_shows_error(client):
    """Submitting the form with no file field shows the error message."""
    response = client.post("/prediction", data={}, content_type="multipart/form-data")
    assert response.status_code == 200
    assert ERROR_MESSAGE in response.data


def test_empty_file_shows_error(client):
    """Clicking Submit without choosing a file (empty upload) shows the error message."""
    response = client.post(
        "/prediction",
        data={"file": (io.BytesIO(b""), "")},
        content_type="multipart/form-data",
    )
    assert response.status_code == 200
    assert ERROR_MESSAGE in response.data


#  Invalid or unexpected input ]

def test_text_file_shows_error(client):
    """Uploading a text file instead of an image shows the error message."""
    response = client.post(
        "/prediction",
        data={"file": (io.BytesIO(b"this is not an image"), "notes.txt")},
        content_type="multipart/form-data",
    )
    assert response.status_code == 200
    assert ERROR_MESSAGE in response.data


def test_grayscale_image_shows_error(client, make_image):
    """A 1-channel grayscale image cannot be reshaped to 3 channels, so it is rejected."""
    response = client.post(
        "/prediction",
        data={"file": (make_image(mode="L"), "gray.png")},
        content_type="multipart/form-data",
    )
    assert response.status_code == 200
    assert ERROR_MESSAGE in response.data


def test_get_request_to_prediction_not_allowed(client):
    """/prediction only accepts POST, so a GET returns 405 Method Not Allowed."""
    response = client.get("/prediction")
    assert response.status_code == 405


def test_unknown_page_returns_404(client):
    """A page that does not exist returns 404 Not Found."""
    response = client.get("/does-not-exist")
    assert response.status_code == 404
