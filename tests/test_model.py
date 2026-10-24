"""Tests for image preprocessing and prediction in model.py."""
import io
from unittest.mock import patch

import numpy as np
import pytest
from PIL import UnidentifiedImageError

from model import predict_result, preprocess_img


#  preprocess_img

def test_preprocess_returns_model_input_shape(sample_image_path):
    """A real photo becomes a (1, 224, 224, 3) array, the shape the model expects."""
    processed = preprocess_img(sample_image_path)
    assert processed.shape == (1, 224, 224, 3)


def test_preprocess_scales_pixels_between_0_and_1(sample_image_path):
    """Pixel values are normalised to the 0-1 range."""
    processed = preprocess_img(sample_image_path)
    assert processed.min() >= 0.0
    assert processed.max() <= 1.0


def test_preprocess_resizes_any_image_size(make_image):
    """Edge case: a small, non-square image is still resized to 224x224."""
    processed = preprocess_img(make_image(size=(50, 80)))
    assert processed.shape == (1, 224, 224, 3)


def test_preprocess_rejects_non_image_data():
    """Edge case: bytes that are not an image raise UnidentifiedImageError."""
    with pytest.raises(UnidentifiedImageError):
        preprocess_img(io.BytesIO(b"not an image"))


def test_preprocess_rejects_grayscale_image(make_image):
    """Edge case: a grayscale (1-channel) image cannot be reshaped to 3 channels."""
    with pytest.raises(ValueError):
        preprocess_img(make_image(mode="L"))


# predict_result

def test_predict_returns_index_of_highest_score():
    """predict_result returns the digit with the highest model score (model mocked)."""
    fake_scores = np.array([[0.01, 0.02, 0.05, 0.10, 0.02, 0.60, 0.05, 0.05, 0.05, 0.05]])
    with patch("model.model.predict", return_value=fake_scores):
        result = predict_result(np.zeros((1, 224, 224, 3)))
    assert result == 5


def test_predict_returns_plain_int():
    """predict_result returns a plain Python int, not a NumPy type."""
    fake_scores = np.array([[0.9, 0.1, 0, 0, 0, 0, 0, 0, 0, 0]])
    with patch("model.model.predict", return_value=fake_scores):
        result = predict_result(np.zeros((1, 224, 224, 3)))
    assert isinstance(result, int)


def test_real_model_predicts_a_digit(sample_image_path):
    """With the real model, the prediction is a digit between 0 and 9."""
    result = predict_result(preprocess_img(sample_image_path))
    assert 0 <= result <= 9
