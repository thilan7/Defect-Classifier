from keras.models import load_model
from keras.preprocessing import image
import numpy as np

# dimensions of images
img_width, img_height = 150, 150

# load the saved model
model = load_model('model.h5')
model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

# predicting images
img = image.load_img('test4.jpg', target_size=(img_width, img_height))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)

images = np.vstack([x])
classes = model.predict_classes(images, batch_size=10)
print (classes)

# predicting multiple images at once
img = image.load_img('test5.jpg', target_size=(img_width, img_height))
y = image.img_to_array(img)
y = np.expand_dims(y, axis=0)

# pass the list of multiple images np.vstack()
images = np.vstack([x, y])
classes = model.predict_classes(images, batch_size=10)

# print the classes, that images belongs
print (classes)
print("XXXXXXXXXXX")
print (classes[0])
print("XXXXXXXXXXX")
print (classes[0][0])