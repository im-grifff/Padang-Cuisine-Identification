import tensorflow as tf
from keras.models import load_model
from PIL import Image
from flask import Flask, render_template, request
import numpy as np

app = Flask(__name__)

# Load model
model = tf.keras.models.load_model('./assets/padangmodel.h5')

class_names = ['ayam_goreng', 'ayam_pop', 'daging_rendang', 'dendeng_batokok', 'gulai_ikan', 'gulai_tunjang', 'telur_balado', 'telur_dadar']

def process_image(image_path):
    img = Image.open(image_path)
    img = img.resize((224, 224))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    return img


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # Mendapatkan file gambar dari form
        file = request.files['file']

        # Menyimpan file sementara
        file_path = './static/images/temp.png'
        file.save(file_path)

        # Memproses gambar
        img = process_image(file_path)

        # Melakukan klasifikasi gambar
        predictions = model.predict(img)
        predicted_class = np.argmax(predictions[0])
        predicted_label = class_names[predicted_class]
        accuracy = predictions[0][predicted_class] * 100
        accuracy_formatted = "{:.2f}".format(accuracy)

        # Mengirimkan hasil klasifikasi ke halaman web
        return render_template('index.html', label=predicted_label, accuracy=accuracy_formatted, image_file=file_path)
    else:
        return render_template('index.html', label=None, accuracy=None, image_file=None)

if __name__ == "__main__":
    app.run(debug=True)
