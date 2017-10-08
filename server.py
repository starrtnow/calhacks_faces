from flask import Flask
from flask import request
from flask import send_file
from networks import VAENetwork
from scipy import misc

import os
import torchvision
import torchvision.transforms as tf
import uuid

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = './uploads/'

params = {
    'learning rate': 0.005,     
    'encoder dropout': 0.65,     
    'decoder dropout': 0.45,     
    'z_channel': 32,     
    'z_dim': 128 
}

dataset = torchvision.datasets.ImageFolder(root="data/faces", transform=tf.ToTensor())
vae = VAENetwork(dataset, hyper_params = params, cuda=False)
vae.load("mc9")

import data

@app.route("/")
def hello():
    return generate_random_face()

def make_uuid():
    return uuid.uuid4().hex

def generate_random_face():
    iid = make_uuid()
    face = vae.sample()
    torchvision.utils.save_image(face[0], "{}.png".format(iid))
    return send_file("{}.png".format(iid))

from scipy import misc
from PIL import Image
import cv2
import torchvision.transforms as tf

def pil_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')

def generate_from_image(filename, uid):
    path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    data.cut_head(path, "faces", (128, 128))
    face = pil_loader(os.path.join("faces", filename))
    face = tf.ToTensor()(face)
    face = face.view(1, 3, 128, 128)
    morphed = vae.sample(face)
    
    morphed_file = os.path.join("faces", "morphed{}.png".format(uid))
    torchvision.utils.save_image(morphed[0], morphed_file)
    img = data.paste_head(morphed_file, path)
    
    wtf = os.path.join("faces", "combined{}.png".format(uid))
    cv2.imwrite(wtf, img)
    return send_file(wtf)

@app.route("/upload", methods=['GET', 'POST'])
def upload_file():
    uid = make_uuid()
    if request.method == 'POST':
        if 'file' not in request.files:
            return "no files"
        f = request.files['file']
        new_filename = uid + f.filename
        if f.filename != '' and f:
            f.save(os.path.join(app.config['UPLOAD_FOLDER'], new_filename))
            return generate_from_image(new_filename, uid)

    return '''
        <!doctype html>
        <title>Upload a file</title>
        <h1>Upload an image</h1>
        <form method=post enctype=multipart/form-data>.
            <p><input type=file name=file>
                <input type=submit value=Upload>
        </form>
        '''




if __name__ == "__main__":
    app.run(host='0.0.0.0')
