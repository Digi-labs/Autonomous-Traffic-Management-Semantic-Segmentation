#Usage: python app.py
import os

from flask import Flask, render_template, request, redirect, url_for
from werkzeug import secure_filename
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from keras.models import Sequential, load_model
import numpy as np
import argparse
import imutils
import cv2
import time
import tensorflow as tf
import uuid
import base64
import sys

sys.path.append("..")

from utils import label_map_util
from utils import visualization_utils as vis_util
from distutils.version import StrictVersion
# module level variables ##############################################################################################



MODEL_PATH  =  "enet-cityscapes/enet-model.net"
CLASSES_PATH = "enet-cityscapes/enet-classes.txt"
COLORS_PATH  =  "enet-cityscapes/enet-colors.txt"

img_width, img_height = 300, 168


UPLOAD_FOLDER = 'uploads'

TEST_FOLDER = 'test'


ALLOWED_EXTENSIONS = set(['jpg', 'jpeg','png','mp4'])

def get_as_base64(url):
    return base64.b64encode(requests.get(url).content)

def mai(PATH_TO_IMAGE):
   CLASSES = open(CLASSES_PATH).read().strip().split("\n")

   # if a colors file was supplied, load it from disk
   if COLORS_PATH:
	    COLORS = open(COLORS_PATH).read().strip().split("\n")
	    COLORS = [np.array(c.split(",")).astype("int") for c in COLORS]
	    COLORS = np.array(COLORS, dtype="uint8")

   # otherwise, we need to randomly generate RGB colors for each class
   # label
   else:
	    # initialize a list of colors to represent each class label in
	    # the mask (starting with 'black' for the background/unlabeled
	    # regions)
	    np.random.seed(42)
	    COLORS = np.random.randint(0, 255, size=(len(CLASSES) - 1, 3),
		    dtype="uint8")
	    COLORS = np.vstack([[0, 0, 0], COLORS]).astype("uint8")

   # initialize the legend visualization
   legend = np.zeros(((len(CLASSES) * 25) + 25, 300, 3), dtype="uint8")

   # loop over the class names + colors
   for (i, (className, color)) in enumerate(zip(CLASSES, COLORS)):
	    # draw the class name + color on the legend
	    color = [int(c) for c in color]
	    cv2.putText(legend, className, (5, (i * 25) + 17),
		    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
	    cv2.rectangle(legend, (100, (i * 25)), (300, (i * 25) + 25),
		    tuple(color), -1)

   # load our serialized model from disk
   print("[INFO] loading model...")
   net = cv2.dnn.readNet(MODEL_PATH)

   # load the input image, resize it, and construct a blob from it,
   # but keeping mind mind that the original input image dimensions
   # ENet was trained on was 1024x512
   image = cv2.imread(PATH_TO_IMAGE)
   image = imutils.resize(image,img_width)
   blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (1024, 512), 0,
	    swapRB=True, crop=False)

   # perform a forward pass using the segmentation model
   net.setInput(blob)
   start = time.time()
   output = net.forward()
   end = time.time()

   # show the amount of time inference took
   print("[INFO] inference took {:.4f} seconds".format(end - start))

   # infer the total number of classes along with the spatial dimensions
   # of the mask image via the shape of the output array
   (numClasses, height, width) = output.shape[1:4]

   # our output class ID map will be num_classes x height x width in
   # size, so we take the argmax to find the class label with the
   # largest probability for each and every (x, y)-coordinate in the
   # image
   classMap = np.argmax(output[0], axis=0)

   # given the class ID map, we can map each of the class IDs to its
   # corresponding color
   mask = COLORS[classMap]

   # resize the mask and class map such that its dimensions match the
   # original size of the input image (we're not using the class map
   # here for anything else but this is how you would resize it just in
   # case you wanted to extract specific pixels/classes)
   mask = cv2.resize(mask, (image.shape[1], image.shape[0]),
	   interpolation=cv2.INTER_NEAREST)
   classMap = cv2.resize(classMap, (image.shape[1], image.shape[0]),
	   interpolation=cv2.INTER_NEAREST)

   # perform a weighted combination of the input image with the mask to
   # form an output visualization
   output = ((0.4 * image) + (0.6 * mask)).astype("uint8")

   # show the input and output images
   cv2.imshow("Legend", legend)
   cv2.imshow("Input", image)
   cv2.imshow("Output", output)
   cv2.imwrite('uploads/Output.png',output)
   cv2.waitKey(0)

# end main

def my_random_string(string_length=10):
    """Returns a random string of length string_length."""
    random = str(uuid.uuid4()) # Convert UUID format to a Python string.
    random = random.upper() # Make all characters uppercase.
    random = random.replace("-","") # Remove the UUID '-'.
    return random[0:string_length] # Return the random string.

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route("/")
def template_test():
    return render_template('template.html', imagesource='../uploads/template.png')


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        import time
        start_time = time.time()
        file = request.files['file']
       

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)

            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            mai(file_path)

            print(file_path)
            filename = my_random_string(6) + filename

            os.rename(file_path, os.path.join(app.config['UPLOAD_FOLDER'], filename))
            print("--- %s seconds ---" % str (time.time() - start_time))
            return render_template('template.html',imagesource='../uploads/' + filename )

from flask import send_from_directory

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'],
                               filename)

from werkzeug import SharedDataMiddleware
app.add_url_rule('/uploads/<filename>', 'uploaded_file',
                 build_only=True)
app.wsgi_app = SharedDataMiddleware(app.wsgi_app, {
    '/uploads':  app.config['UPLOAD_FOLDER']
})

if __name__ == "__main__":
    app.debug=False
    app.run(host='0.0.0.0', port=3000)