"""Flask web application for hand-sign digit recognition.

Routes:
    /            shows the upload page.
    /prediction  receives an uploaded image and shows the predicted digit.
"""


# Importing required libs
from flask import Flask, render_template, request
from model import preprocess_img, predict_result

# Instantiating flask app
app = Flask(__name__)


# Home route
@app.route("/")
def main():
    """Render the home page with the image upload form."""
    return render_template("index.html")


# Prediction route
@app.route('/prediction', methods=['POST'])
def predict_image_file():
    """Preprocess the uploaded image, predict the digit and render the result.

    If no file is uploaded, or the file is not a usable image, the result
    page shows an error message instead of crashing.
    """
    try:
        img = preprocess_img(request.files['file'].stream)
        pred = predict_result(img)
        return render_template("result.html", predictions=str(pred))
    # KeyError:   the request has no "file" field.
    # OSError:    the upload is empty or not an image (PIL's UnidentifiedImageError).
    # ValueError: the image cannot be reshaped for the model (e.g. grayscale).
    except (KeyError, OSError, ValueError):
        error = "File cannot be processed."
        return render_template("result.html", err=error)

# Driver code
if __name__ == "__main__":
    app.run(port=9000, debug=True)
