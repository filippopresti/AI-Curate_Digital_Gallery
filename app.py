# style_transfer_app/app.py
from flask import Flask, render_template, request
import os
import io
import base64
import tensorflow as tf
import tensorflow_hub as hub
from PIL import Image
import numpy as np


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.path.join('static', 'uploads')
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load the style transfer model
print("Loading model...")
model_url = "https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2"
model = hub.load(model_url)
print("Model loaded.")

def preprocess_image(image_path, target_shortest_side=512):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    original_shape = tf.cast(tf.shape(img)[:-1], tf.float32)
    scale = target_shortest_side / tf.reduce_min(original_shape)
    new_shape = tf.cast(original_shape * scale, tf.int32)
    img = tf.image.resize(img, new_shape)
    img = img[tf.newaxis, :]
    return img

def tensor_to_image(tensor):
    tensor = tensor * 255
    tensor = np.array(tensor, dtype=np.uint8)
    if np.ndim(tensor) > 3:
        assert tensor.shape[0] == 1
        tensor = tensor[0]
    return Image.fromarray(tensor)

def apply_style_transfer(content_image_path, style_image_path):
    try:
        print("preprocessing images")
        content_image = preprocess_image(content_image_path)
        style_image = preprocess_image(style_image_path)
        print("applying style transfer")
        stylized_image = model(tf.constant(content_image), tf.constant(style_image))[0]
        print("style transfer complete")
        stylized_image = tensor_to_image(stylized_image)
        buffered = io.BytesIO()
        stylized_image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        return f"data:image/png;base64,{img_str}"
    except Exception as e:
        print(f"Error during style transfer {e}")
        return None


@app.route('/', methods=['GET', 'POST'])
def index():
    content_image_path = None
    style_image_path = None
    stylized_image_data = None

    if request.method == 'POST':

        # Save content image
        if 'content-image' in request.files:
            content_image = request.files['content-image']
            if content_image.filename != '':
                content_image_path = os.path.join(app.config['UPLOAD_FOLDER'], content_image.filename)
                content_image.save(content_image_path)

        # Save style image
        if 'style-image' in request.files:
            style_image = request.files['style-image']
            if style_image.filename != '':
                style_image_path = os.path.join(app.config['UPLOAD_FOLDER'], style_image.filename)
                style_image.save(style_image_path)

        if content_image_path and style_image_path:
            stylized_image_data = apply_style_transfer(content_image_path, style_image_path)
            if stylized_image_data:
                return render_template('index.html',
                    content_image_path=content_image_path,
                    style_image_path=style_image_path,
                    stylized_image_data=stylized_image_data)
            else:
                error = "Failed to style transfer."
                return render_template('index.html',
                     content_image_path=content_image_path,
                    style_image_path=style_image_path,
                    error=error)
        else:
             error = "Please upload both content and style images"
             return render_template('index.html', error=error)
    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)