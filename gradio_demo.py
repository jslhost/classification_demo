import tensorflow as tf
from tensorflow.keras.utils import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.vgg16 import preprocess_input

import gradio as gr

model = tf.keras.models.load_model('best_model.h5', compile=False)

optimizer = tf.keras.optimizers.Adam(learning_rate=1e-6)
#optimizer = tf.keras.optimizers.RMSprop(learning_rate=1e-6)

model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

def classify_image(input):
    #image = load_img(input, target_size=(224, 224))
    image = img_to_array(input)
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    image = preprocess_input(image)
    prediction = model.predict(image)
    result = 'cat' if prediction[0][0] < 0.5 else 'dog'
    return result

gr.Interface(fn=classify_image,
             inputs=gr.Image(shape=(224, 224)),
             outputs=gr.Label(),
             examples=["test_image.jpg", "test_dog_image.jpg"]).launch()