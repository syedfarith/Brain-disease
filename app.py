# import json
# from keras.models import model_from_json
# from keras.utils import get_custom_objects
# from keras.layers import DepthwiseConv2D

# # Custom DepthwiseConv2D layer without 'groups' parameter
# class CustomDepthwiseConv2D(DepthwiseConv2D):
#     def __init__(self, **kwargs):
#         if 'groups' in kwargs:
#             kwargs.pop('groups')
#         super(CustomDepthwiseConv2D, self).__init__(**kwargs)

# # Register the custom layer
# get_custom_objects().update({'DepthwiseConv2D': CustomDepthwiseConv2D})

# # Load the model JSON
# with open("model.json", "r") as json_file:
#     model_json = json_file.read()

# # Modify the model JSON to remove 'groups' parameter from DepthwiseConv2D layers
# model_config = json.loads(model_json)
# for layer in model_config['config']['layers']:
#     if layer['class_name'] == 'DepthwiseConv2D' and 'groups' in layer['config']:
#         del layer['config']['groups']

# # Recreate the model from the modified JSON
# model = model_from_json(json.dumps(model_config), custom_objects={'DepthwiseConv2D': CustomDepthwiseConv2D})

# # Load weights into the new model
# model.load_weights("model.h5")

# # Now you can use the model as usual

# # Streamlit code for uploading and predicting images
# import streamlit as st
# from PIL import Image, ImageOps
# import numpy as np

# # Disable scientific notation for clarity
# np.set_printoptions(suppress=True)

# # Load the labels
# class_names = open("labels.txt", "r").readlines()

# # Streamlit app title
# st.title("Image Classification with Teachable Machine")

# # Upload image
# uploaded_file = st.file_uploader("Choose an image...", type="jpg")

# if uploaded_file is not None:
#     # Open the image file
#     image = Image.open(uploaded_file).convert("RGB")
    
#     # Display the image
#     st.image(image, caption='Uploaded Image.', use_column_width=True)
#     st.write("")
#     st.write("Classifying...")

#     # Resizing the image to be at least 224x224 and then cropping from the center
#     size = (224, 224)
#     image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)

#     # Turn the image into a numpy array
#     image_array = np.asarray(image)

#     # Normalize the image
#     normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

#     # Create the array of the right shape to feed into the keras model
#     data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

#     # Load the image into the array
#     data[0] = normalized_image_array

#     # Predict the model
#     prediction = model.predict(data)
#     index = np.argmax(prediction)
#     class_name = class_names[index]
#     confidence_score = prediction[0][index]

#     # Print prediction and confidence score
#     st.write("Class:", class_name[2:])
#     st.write("Confidence Score:", confidence_score)



import streamlit as st
from keras.models import load_model  # TensorFlow is required for Keras to work
from PIL import Image, ImageOps  # Install pillow instead of PIL
import numpy as np

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load the model
model = load_model("keras_Model.h5", compile=False)

# Load the labels
class_names = open("labels.txt", "r").readlines()

# Streamlit app title
st.title("Brain Disease Classification")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    # Open the image file
    image = Image.open(uploaded_file).convert("RGB")
    
    # Display the image
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("Classifying...")

    # Resizing the image to be at least 224x224 and then cropping from the center
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)

    # Turn the image into a numpy array
    image_array = np.asarray(image)

    # Normalize the image
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

    # Create the array of the right shape to feed into the keras model
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

    # Load the image into the array
    data[0] = normalized_image_array

    # Predict the model
    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]

    # Print prediction and confidence score
    st.write("Class:", class_name[2:].strip())  # Remove leading/trailing whitespace
    st.write("Confidence Score:", confidence_score*100)
