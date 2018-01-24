import tensorflow as tf
import numpy as np
import os, glob, cv2
import sys, argparse
from optparse import OptionParser

classes = ['yes', 'no']
result_array = []
# First, pass the path of the image
dir_path = os.path.dirname(os.path.realpath(__file__))

parser = OptionParser(usage='Use -h to see more infomation')
parser.add_option("-m", "--model-path", dest="modelpath",
                  help="model path", type="str", action="store")
parser.add_option("-i", "--index", dest="dataindex",
                  help="data index", type="str", action="store")
(options, args) = parser.parse_args()

model_name = options.modelpath if options.modelpath else 'v3-yes-no-model.meta'
image_index = options.dataindex if options.dataindex else '15'
image_path = 'testing_data_{}'.format(image_index) #sys.argv[1]
path = os.path.join(image_path, '*g')
files = glob.glob(path)


y_test_images = np.zeros((1, 2))

image_size = 15#128
num_channels = 1
images = []

for fl in files:
    image = cv2.imread(fl, 0)
    image = cv2.resize(image, (image_size, image_size), 0, 0, cv2.INTER_LINEAR)
    image = image.astype(np.float32)
    image = np.multiply(image, 1.0 / 255.0)
    image = np.reshape(image, (image_size, image_size, num_channels))
    images.append(image)

images = np.array(images)

## Let us restore the saved model
sess = tf.Session()
# Step-1: Recreate the network graph. At this step only graph is created.
saver = tf.train.import_meta_graph(model_name)
# Step-2: Now let's load the weights saved using the restore method.
saver.restore(sess, tf.train.latest_checkpoint('./'))

# Accessing the default graph which we have restored
graph = tf.get_default_graph()

# Now, let's get hold of the op that we can be processed to get the output.
# In the original network y_pred is the tensor that is the prediction of the network
y_pred = graph.get_tensor_by_name("y_pred:0")

## Let's feed the images to the input placeholders
x = graph.get_tensor_by_name("x:0")
y_true = graph.get_tensor_by_name("y_true:0")

# #The input to the network is of shape [None image_size image_size num_channels]. Hence we reshape.
# x_batch = images.reshape(1, image_size, image_size, num_channels)

### Creating the feed_dict that is required to be fed to calculate y_pred
feed_dict_testing = {x: images, y_true: y_test_images}
result = sess.run(y_pred, feed_dict=feed_dict_testing)
# result is of this format [probabiliy_of_rose probability_of_sunflower]
print(result)
pre_class = sess.run(tf.argmax(result, axis=1))
print(len(pre_class))
print(len(files))

for index in range(len(pre_class)):
    if classes[pre_class[index]] == 'yes':
        result_array.append(files[index])
print(result_array)

import re
pattern_filename = re.compile(r'{}\\(?P<row>\d+)x(?P<col>\d+).png'.format(image_path))

origin_img = cv2.imread('origin_data/{}-02-last_result.png'.format(image_index), 0)
img_zero = np.zeros_like(origin_img)
square_array = []
for element in result_array:
    result_re = re.match(pattern_filename, element)
    row = int(result_re.group('row'))
    col = int(result_re.group('col'))

    # square_array.append([[row, col], [row+15, col+15]])
    cv2.rectangle(img_zero, (col, row), (col+15, row+15), 255, -1)

    print(row, col)

image, contours, hierarchy = cv2.findContours(img_zero, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
cv2.drawContours(origin_img, contours, -1, 255, 1)  # Draw filled contour in mask

cv2.imwrite('{}.png'.format(image_index), origin_img)
