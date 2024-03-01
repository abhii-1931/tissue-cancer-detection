from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
from PIL import Image
from io import BytesIO


# Read image file and perform prediction


app = Flask(__name__)

# Load the pre-trained model using TensorFlow.js
model = tf.keras.models.load_model("model.h5")


@app.route("/")
def index():
    return render_template("templates/index.html", result="")


@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return render_template("templates/index.html", result="Error: No file provided")

    image_file = request.files["image"]

    if image_file.filename == "":
        return render_template("templates/index.html", result="Error: No file selected")

    # Read image file and perform prediction
    # Read image file and perform prediction
    image = Image.open(image_file)
    image = image.resize((96, 96))  # Resize to match the model's expected input size
    image_array = np.array(image) / 255.0  # Normalization

    # Expand the dimensions to match the model's expected shape
    image_array = np.expand_dims(image_array, axis=0)

    # Make the prediction
    prediction = model.predict(image_array)

    # result = f"There is a {percentage}% chance of having cancer."

    if prediction >= 0.5:
        result = "Cancer detected"
    else:
        result = "No cancer detected"

    prediction = prediction * 100

    # Display the result
    output = "Cancer detected" if prediction > 0.5 else "No cancer detected"
    return render_template("templates/index.html", result=result, percentage=prediction)


if __name__ == "__main__":
    app.run(debug=True)
