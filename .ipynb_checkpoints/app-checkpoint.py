import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import tempfile
import cv2

# Dictionary of solutions for each disease
solutions_dict = {
    "Corn (Maize)": {
        "Common Rust": {
            "Solution": "Use resistant corn varieties. Apply fungicides such as azoxystrobin or pyraclostrobin at the first sign of disease, especially during the early growth stages.",
            "Prevention": "Crop rotation and proper field sanitation to remove infected plant debris."
        },
        "Northern Leaf Blight": {
            "Solution": "Use resistant corn varieties. Fungicides like mancozeb, azoxystrobin, or pyraclostrobin are effective when applied at the first sign of disease.",
            "Prevention": "Crop rotation and residue management to reduce pathogen survival."
        }
    },
    "Grape": {
        "Black Rot": {
            "Solution": "Apply fungicides such as myclobutanil, trifloxystrobin, or captan starting from the early growth stages (bud break) and continuing through the season.",
            "Prevention": "Prune and remove infected plant parts. Ensure good air circulation and use resistant grape varieties."
        },
        "Esca (Black Measles)": {
            "Solution": "There is no effective chemical treatment. Focus on cultural practices such as pruning infected wood and avoiding trunk injuries.",
            "Prevention": "Regular monitoring and sanitation by removing diseased plant material."
        },
        "Leaf Blight (Isariopsis Leaf Spot)": {
            "Solution": "Apply fungicides like copper-based products or mancozeb.",
            "Prevention": "Avoid overhead irrigation and ensure proper pruning for air circulation."
        }
    },
    "Orange": {
        "Huanglongbing (Citrus Greening)": {
            "Solution": "There is no cure for citrus greening once a tree is infected. Management focuses on controlling the insect vector (Asian citrus psyllid) using insecticides like imidacloprid or thiamethoxam.",
            "Prevention": "Use disease-free planting material and remove infected trees. Implement integrated pest management (IPM) strategies to control psyllid populations."
        }
    },
    "Peach": {
        "Bacterial Spot": {
            "Solution": "Use bactericides such as copper-based products or oxytetracycline. Start applications before symptoms appear and continue at regular intervals.",
            "Prevention": "Use resistant varieties and ensure good air circulation through proper pruning."
        },
        "Healthy": {
            "Solution": "No treatment needed for healthy plants. Maintain regular monitoring to detect early signs of disease."
        }
    },
    "Pepper (Bell)": {
        "Bacterial Spot": {
            "Solution": "Apply copper-based bactericides or streptomycin. Start treatments early, before symptoms appear, and repeat during wet conditions.",
            "Prevention": "Use certified disease-free seeds and resistant varieties. Practice crop rotation and avoid overhead watering."
        },
        "Healthy": {
            "Solution": "No treatment needed for healthy plants. Continue with preventive measures such as proper watering and fertilization."
        }
    },
    "Potato": {
        "Early Blight": {
            "Solution": "Use fungicides such as chlorothalonil, mancozeb, or azoxystrobin. Start application early in the season and continue on a regular schedule.",
            "Prevention": "Practice crop rotation and use disease-free seed potatoes. Remove and destroy infected plant debris."
        },
        "Late Blight": {
            "Solution": "Apply fungicides like metalaxyl, cymoxanil, or mancozeb as soon as disease symptoms are detected or based on weather conditions that favor disease development.",
            "Prevention": "Use resistant varieties and certified disease-free seed potatoes. Ensure proper field sanitation and avoid overhead irrigation."
        },
        "Healthy": {
            "Solution": "Maintain healthy potato crops with regular monitoring, proper fertilization, and irrigation practices."
        }
    },
    "Raspberry": {
        "Healthy": {
            "Solution": "No treatment needed for healthy plants. Regular monitoring and proper cultural practices will help prevent disease."
        }
    },
    "Soybean": {
        "Healthy": {
            "Solution": "No treatment needed for healthy plants. Continue with preventive measures such as crop rotation and pest management."
        }
    },
    "Squash": {
        "Powdery Mildew": {
            "Solution": "Apply fungicides such as sulfur, myclobutanil, or potassium bicarbonate at the first sign of disease.",
            "Prevention": "Ensure good air circulation around plants and avoid excessive nitrogen fertilization."
        }
    },
    "Strawberry": {
        "Healthy": {
            "Solution": "No treatment needed for healthy plants. Monitor regularly for early signs of disease and maintain good cultural practices."
        },
        "Leaf Scorch": {
            "Solution": "Apply fungicides like captan or copper-based products when symptoms appear.",
            "Prevention": "Ensure good air circulation and avoid overhead watering."
        }
    },
    "Tomato": {
        "Bacterial Spot": {
            "Solution": "Use copper-based bactericides or streptomycin at the first sign of disease.",
            "Prevention": "Use disease-free seeds and resistant varieties. Practice crop rotation and avoid overhead watering."
        },
        "Early Blight": {
            "Solution": "Apply fungicides such as chlorothalonil or mancozeb starting from the early growth stages.",
            "Prevention": "Remove and destroy infected plant material and avoid overhead irrigation."
        },
        "Healthy": {
            "Solution": "No treatment needed for healthy plants. Regular monitoring and preventive care will help keep plants disease-free."
        },
        "Late Blight": {
            "Solution": "Use fungicides like metalaxyl, chlorothalonil, or mancozeb when disease symptoms are first observed.",
            "Prevention": "Use resistant varieties and ensure proper field sanitation."
        },
        "Leaf Mold": {
            "Solution": "Apply fungicides such as mancozeb or copper-based products when symptoms are observed.",
            "Prevention": "Ensure good ventilation and reduce humidity around plants."
        },
        "Septoria Leaf Spot": {
            "Solution": "Use fungicides like chlorothalonil or mancozeb at the first sign of symptoms.",
            "Prevention": "Remove and destroy infected leaves and avoid overhead irrigation."
        },
        "Spider Mites (Two-spotted spider mite)": {
            "Solution": "Use miticides like abamectin or bifenazate. Neem oil and insecticidal soap can also be effective.",
            "Prevention": "Maintain proper irrigation to reduce plant stress, which makes them less susceptible to mites."
        },
        "Target Spot": {
            "Solution": "Apply fungicides such as chlorothalonil or mancozeb when symptoms are observed.",
            "Prevention": "Use resistant varieties and practice crop rotation."
        },
        "Tomato Mosaic Virus": {
            "Solution": "There is no cure for tomato mosaic virus. Focus on prevention by using virus-free seeds and practicing good sanitation.",
            "Prevention": "Avoid tobacco products near plants and wash hands thoroughly before handling plants."
        },
        "Tomato Yellow Leaf Curl Virus": {
            "Solution": "There is no direct treatment for this virus. Control the whitefly vector using insecticides like imidacloprid or neem oil.",
            "Prevention": "Use resistant varieties and implement IPM strategies to manage whitefly populations."
        }
    }
}

# Load your model here
model = tf.keras.models.load_model('model1.h5')

# Define the function for model prediction
def model_prediction(image):
    # Resize the image to match the model's expected input shape
    image = image.resize((128, 128))
    
    # Convert the image to a numpy array and normalize it
    image_array = np.array(image) / 255.0

    # Expand dimensions to match the model's input format (batch_size, height, width, channels)
    image_array = np.expand_dims(image_array, axis=0)

    # Predict using the model
    prediction = model.predict(image_array)

    # Get the index of the class with the highest probability
    result_index = np.argmax(prediction, axis=1)[0]

    return result_index

# Streamlit app interface code
st.title("AI Leaf Disease Detection System")

# Image upload feature
uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    # Load the image
    image = Image.open(uploaded_file)
    
    # Display the image in Streamlit
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    
    # Classify the image
    st.write("Classifying...")
    result_index = model_prediction(image)
    
    # Display the result
    st.write(f'Result Index: {result_index}')

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
    Choose the **Disease Recognition** option from the menu and start diagnosing plant diseases right away!

    ### About Us
    Learn more about our mission and technology on the **About** page.

    Thank you for choosing our Plant Disease Recognition System! 🍃
    """)

# About Page
elif app_mode == "About":
    st.header("ABOUT")
    st.subheader("Leaf Disease Recognition")
    st.write("""
    Our Plant Disease Recognition System uses cutting-edge machine learning techniques to identify various plant diseases from images. It helps farmers, gardeners, and agricultural experts to quickly diagnose plant health issues and take appropriate action.

    ### How it Works:
    - **Upload an Image:** The user uploads an image of the plant leaf.
    - **Processing:** The system analyzes the image using a trained machine learning model.
    - **Result:** The disease is identified, and solutions are provided to mitigate the damage.

    ### Technologies Used:
    - **Streamlit:** For creating a user-friendly web application.
    - **TensorFlow:** For building and deploying the machine learning model.
    - **OpenCV and PIL:** For image processing.
    """)

# Disease Recognition Page
elif app_mode == "Disease Recognition":
    st.header("Disease Recognition")
    st.write("Upload an image of a plant leaf to recognize the disease.")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Make prediction
        st.write("Classifying...")
        result_index = model_prediction(image)
        
        disease_classes = ["Healthy", "Bacterial Spot", "Late Blight", "Early Blight", ...]  # Add all classes here
        result_label = disease_classes[result_index]  # Convert index to label

        st.success(f"Prediction: {result_label}")

        # Display disease information and solutions
        if result_label in solutions_dict:
            disease_info = solutions_dict[result_label]
            st.write(f"### Solution: {disease_info['Solution']}")
            st.write(f"### Prevention: {disease_info['Prevention']}")
        else:
            st.info("No solution information available for this disease.")
            
        # Option for using a live camera feed
        st.write("Capture image from live camera:")
        use_camera = st.checkbox("Use camera")
        if use_camera:
            cap = cv2.VideoCapture(0)
            if st.button("Capture"):
                ret, frame = cap.read()
                if ret:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    image = Image.fromarray(frame)
                    st.image(image, caption="Captured Image", use_column_width=True)
                    # Make prediction on captured image
                    result_index = model_prediction(image)
                    result_label = disease_classes[result_index]  # Convert index to label
                    st.success(f"Prediction: {result_label}")
                    if result_label in solutions_dict:
                        disease_info = solutions_dict[result_label]
                        st.write(f"### Solution: {disease_info['Solution']}")
                        st.write(f"### Prevention: {disease_info['Prevention']}")
                    else:
                        st.info("No solution information available for this disease.")
                cap.release()

