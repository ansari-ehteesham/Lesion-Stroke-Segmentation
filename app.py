from flask import Flask, request, render_template
from PIL import Image
from io import BytesIO
import base64
import numpy as np
from lesionSeg.Pipeline.prediction import Prediction

app = Flask(__name__)

# Make sure to import or define your Prediction class and any required functions.
# from your_module import Prediction

@app.route("/", methods=['GET', 'POST'])
def hello_world():
    if request.method == 'POST':
        model = Prediction()  # Ensure Prediction is properly imported/defined

        # Retrieve the uploaded image from the form (input name="file")
        img_file = request.files['file']

        # Open the image using PIL and convert to a NumPy array
        img = Image.open(img_file.stream)
        np_img = np.asarray(img)

        # Get the prediction result (e.g., a mask or processed image)
        result = model.model_prediction(np_img)

        # Scale the result image from [0,1] to [0,255] and convert to uint8
        updt_result = (result * 255).astype(np.uint8)

        if updt_result.ndim == 3 and updt_result.shape[2] == 1:
            updt_result = np.squeeze(updt_result, axis=2)

        pred_img = Image.fromarray(updt_result)

        # Save the PIL image to a BytesIO object as JPEG
        buffered = BytesIO()
        pred_img.save(buffered, format="JPEG")

        # Encode the image in base64 to embed in HTML
        img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')

        # Render the template with the image string
        return render_template('index.html', prediction_image=img_str)

    # For GET requests, simply render the template without a prediction image
    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)
