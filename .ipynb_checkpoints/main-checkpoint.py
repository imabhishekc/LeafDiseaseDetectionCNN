# import streamlit as st
# import tensorflow as tf
# import numpy as np

# #Tensorflow model prediction
# def model_prediction(test_image):
#     model = tf.keras.models.load_model('trained_model.keras')
#     image = tf.keras.preprocessing.image.load_img(test_image, target_size=(128, 128))
#     input_arr = tf.keras.preprocessing.image.img_to_array(image)
#     input_arr = np.array([input_arr]) #convert single image to a batch
#     prediction = model.predict(input_arr)
#     result_index = np.argmax(prediction)
#     return result_index

# #sidebar
# st.sidebar.title("Dashboard")
# app_mode = st.sidebar.selectbox("Select Page", ["Home", "About", "Disease Recognition"])

# #Home Page
# if(app_mode == "Home"):
#     st.header("LEAF DISEASE RECOGNITION SYSTEM")
#     image_path = "background.png"
#     st.image(image_path, use_column_width=True)
#     st.markdown(""""
#    Welcome to the Plant Disease Recognition System! 🌿🔍
    
#     Our mission is to help in identifying plant diseases efficiently. Upload an image of a plant, and our system will analyze it to detect any signs of diseases. Together, let's protect our crops and ensure a healthier harvest!

#     ### How It Works
#     1. **Upload Image:** Go to the **Disease Recognition** page and upload an image of a plant with suspected diseases.
#     2. **Analysis:** Our system will process the image using advanced algorithms to identify potential diseases.
#     3. **Results:** View the results and recommendations for further action.

#     ### Why Choose Us?
#     - **Accuracy:** Our system utilizes state-of-the-art machine learning techniques for accurate disease detection.
#     - **User-Friendly:** Simple and intuitive interface for seamless user experience.
#     - **Fast and Efficient:** Receive results in seconds, allowing for quick decision-making.

#     ### Get Started
#     Click on the **Disease Recognition** page in the sidebar to upload an image and experience the power of our Plant Disease Recognition System!

#     ### About Us
#     Learn more about the project, our team, and our goals on the **About** page.
#     """)

# #About Page
# elif(app_mode == "About"):
#     st.header("About")
#     st.markdown(""""
#     #### About Dataset
#                 This dataset is recreated using offline augmentation from the original dataset.The original dataset can be found on this github repo.
#                 This dataset consists of about 87K rgb images of healthy and diseased crop leaves which is categorized into 38 different classes.The total dataset is divided into 80/20 ratio of training and validation set preserving the directory structure.
#                 A new directory containing 33 test images is created later for prediction purpose.
#                 #### Content
#                 1. train (70295 images)
#                 2. test (33 images)
#                 3. validation (17572 images)
#     """)



import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import tempfile
import cv2

# TensorFlow model prediction
def model_prediction(test_image):
    model = tf.keras.models.load_model('trained_model.keras')
    image = tf.keras.preprocessing.image.img_to_array(test_image)
    image = np.expand_dims(image, axis=0)
    prediction = model.predict(image)
    result_index = np.argmax(prediction)
    return result_index

# Sidebar
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page", ["Home", "About", "Disease Recognition"])

# Home Page
if app_mode == "Home":
    st.header("LEAF DISEASE RECOGNITION SYSTEM")
    image_path = "background.png"
    st.image(image_path, use_column_width=True)
    st.markdown("""
    Welcome to the Plant Disease Recognition System! 🌿🔍

    Our mission is to help in identifying plant diseases efficiently. Upload an image of a plant, and our system will analyze it to detect any signs of diseases. Together, let's protect our crops and ensure a healthier harvest!

    ### How It Works
    1. **Upload Image:** Go to the **Disease Recognition** page and upload an image of a plant with suspected diseases.
    2. **Analysis:** Our system will process the image using advanced algorithms to identify potential diseases.
    3. **Results:** View the results and recommendations for further action.

    ### Why Choose Us?
    - **Accuracy:** Our system utilizes state-of-the-art machine learning techniques for accurate disease detection.
    - **User-Friendly:** Simple and intuitive interface for a seamless user experience.
    - **Fast and Efficient:** Receive results in seconds, allowing for quick decision-making.

    ### Get Started
    Click on the **Disease Recognition** page in the sidebar to upload an image and experience the power of our Plant Disease Recognition System!

    ### About Us
    Learn more about the project, our team, and our goals on the **About** page.
    """)

# About Page
elif app_mode == "About":
    st.header("About")
    st.markdown("""
    #### About Dataset
    This dataset is recreated using offline augmentation from the original dataset. The original dataset can be found on this GitHub repo.
    This dataset consists of about 87K RGB images of healthy and diseased crop leaves categorized into 38 different classes. The total dataset is divided into an 80/20 ratio of training and validation set preserving the directory structure.
    A new directory containing 33 test images is created later for prediction purposes.

    #### Content
    1. train (70295 images)
    2. test (33 images)
    3. validation (17572 images)
    """)

# Disease Recognition Page
elif app_mode == "Disease Recognition":
    st.header("Disease Recognition")
    
    upload_option = st.selectbox("Choose an option", ["Upload Image", "Live Camera"])
    
    if upload_option == "Upload Image":
        uploaded_file = st.file_uploader("Choose an image...", type="jpg")
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption='Uploaded Image', use_column_width=True)
            st.write("")
            st.write("Classifying...")
            image = image.resize((128, 128))
            result_index = model_prediction(image)
            st.success(f"Prediction: {result_index}")
    
    elif upload_option == "Live Camera":
        st.write("Press 'Capture' to capture image.")
        start_camera_button = st.button("Start Camera", key="start_camera")
        captured_image = None
        
        # if start_camera_button:
        #     cap = cv2.VideoCapture(0)
        #     frame_window = st.image([])

        #     while True:
        #         ret, frame = cap.read()
        #         if not ret:
        #             break
        #         frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        #         frame_window.image(frame)
                
        #         if st.button("Capture", key="capture"):
        #             captured_image = frame
        #             break
            
        #     cap.release()
        #     cv2.destroyAllWindows()

        #     if captured_image is not None:
        #         temp_image = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
        #         cv2.imwrite(temp_image.name, cv2.cvtColor(captured_image, cv2.COLOR_RGB2BGR))
        #         image = Image.open(temp_image.name)
        #         st.image(image, caption='Captured Image', use_column_width=True)
        #         st.write("")
        #         st.write("Classifying...")
        #         image = image.resize((128, 128))
        #         result_index = model_prediction(image)
        #         st.success(f"Prediction: {result_index}")



# import streamlit as st
# import tensorflow as tf
# import numpy as np
# from PIL import Image
# import tempfile
# import cv2
# from datetime import datetime

# # TensorFlow model prediction
# def model_prediction(test_image):
#     model = tf.keras.models.load_model('trained_model.keras')
#     image = tf.keras.preprocessing.image.img_to_array(test_image)
#     image = np.expand_dims(image, axis=0)
#     prediction = model.predict(image)
#     result_index = np.argmax(prediction)
#     return result_index

# # Sidebar
# st.sidebar.title("Dashboard")
# app_mode = st.sidebar.selectbox("Select Page", ["Home", "About", "Disease Recognition"])

# # Custom CSS for styling
# st.markdown("""
#     <style>
#     .main {
#         background-color: #f5f5f5;
#         color: #333;
#     }
#     .sidebar .sidebar-content {
#         background-color: #f5f5f5;
#         color: #333;
#     }
#     .reportview-container .markdown-text-container {
#         font-family: 'Arial';
#         font-size: 1.1em;
#     }
#     .css-1cpxqw2 p {
#         font-size: 1.2em;
#     }
#     </style>
# """, unsafe_allow_html=True)

# # Home Page
# if app_mode == "Home":
#     st.header("🌿 LEAF DISEASE RECOGNITION SYSTEM")
#     st.image("background.png", use_column_width=True)
#     st.markdown("""
#     Welcome to the Plant Disease Recognition System! 🌿🔍

#     Our mission is to help in identifying plant diseases efficiently. Upload an image of a plant, and our system will analyze it to detect any signs of diseases. Together, let's protect our crops and ensure a healthier harvest!

#     ### How It Works
#     1. **Upload Image:** Go to the **Disease Recognition** page and upload an image of a plant with suspected diseases.
#     2. **Analysis:** Our system will process the image using advanced algorithms to identify potential diseases.
#     3. **Results:** View the results and recommendations for further action.

#     ### Why Choose Us?
#     - **Accuracy:** Our system utilizes state-of-the-art machine learning techniques for accurate disease detection.
#     - **User-Friendly:** Simple and intuitive interface for a seamless user experience.
#     - **Fast and Efficient:** Receive results in seconds, allowing for quick decision-making.

#     ### Get Started
#     Click on the **Disease Recognition** page in the sidebar to upload an image and experience the power of our Plant Disease Recognition System!

#     ### About Us
#     Learn more about the project, our team, and our goals on the **About** page.
#     """)

# # About Page
# elif app_mode == "About":
#     st.header("About")
#     st.markdown("""
#     #### About Dataset
#     This dataset is recreated using offline augmentation from the original dataset. The original dataset can be found on this GitHub repo.
#     This dataset consists of about 87K RGB images of healthy and diseased crop leaves categorized into 38 different classes. The total dataset is divided into an 80/20 ratio of training and validation set preserving the directory structure.
#     A new directory containing 33 test images is created later for prediction purposes.

#     #### Content
#     1. train (70295 images)
#     2. test (33 images)
#     3. validation (17572 images)
#     """)

# # Disease Recognition Page
# elif app_mode == "Disease Recognition":
#     st.header("Disease Recognition")
    
#     upload_option = st.selectbox("Choose an option", ["Upload Image", "Live Camera"])
    
#     if upload_option == "Upload Image":
#         uploaded_file = st.file_uploader("Choose an image...", type="jpg")
        
#         if uploaded_file is not None:
#             image = Image.open(uploaded_file)
#             st.image(image, caption='Uploaded Image', use_column_width=True)
#             st.write("")
#             st.write("Classifying...")
#             image = image.resize((128, 128))
#             result_index = model_prediction(image)
#             st.success(f"Prediction: {result_index}")
    
    elif upload_option == "Live Camera":
        st.write("Press 'Capture' to capture image.")
        start_camera_button = st.button("Start Camera", key="start_camera")
        captured_image = None
        
        if start_camera_button:
            cap = cv2.VideoCapture(0)
            frame_window = st.image([])

            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_window.image(frame)
                
                if st.button("Capture", key="capture"):
                    captured_image = frame
                    break
            
            cap.release()
            cv2.destroyAllWindows()

            if captured_image is not None:
                temp_image = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
                cv2.imwrite(temp_image.name, cv2.cvtColor(captured_image, cv2.COLOR_RGB2BGR))
                image = Image.open(temp_image.name)
                st.image(image, caption='Captured Image', use_column_width=True)
                st.write("")
                st.write("Classifying...")
                image = image.resize((128, 128))
                result_index = model_prediction(image)
                st.success(f"Prediction: {result_index}")

