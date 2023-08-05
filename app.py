import streamlit as st
import cv2
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from preprocess import preprocess_image
import base64



# Load the trained CNN model .
model = load_model("cnn_model.h5", compile=False)
lb = pickle.load(open("label_binarizer.pkl", "rb"))

# Streamlit app 
def main():


# Your Streamlit app code here

    # Set the page title and favicon
    st.set_page_config(page_title="Hand-Written Image Recognition", page_icon="✏️", layout="wide",initial_sidebar_state="expanded")
    st.markdown('<link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/css/bootstrap.min.css" integrity="sha384-Gn5384xqQ1aoWXA+058RXPxPg6fy4IWvTNh0E263XmFcJlSAwiGgFAW/dAiS6JXm" crossorigin="anonymous">', unsafe_allow_html=True)

    # Top bar with blue background and white text
    st.markdown("""
    <nav class="navbar fixed-top  navbar-expand-lg navbar-dark" style="background-color: #3498DB;">
        <a class="navbar-brand" style="color: white;text-align: center;">BD-HDR</a>
    </nav>
    """, unsafe_allow_html=True)

    # # Introduction and instructions
    st.write("Welcome to our Hand-Written Bengali District Name Recognition System!")
    st.write("With this system, you can easily predict the district name of hand-written Bengali district name from images. It utilizes a powerful Deep Learning model that has been trained on a diverse dataset of hand-written district names to provide accurate predictions.")
    
    # How to Use the System
    st.markdown("### How to Use the System:")
    st.markdown("- Upload an image of a hand-written Bengali district name.")
    st.markdown("- Click the \"Predict\" button to get the district name prediction.")
    st.write("To help you understand the system better, we have provided some example images on the left.You can easily interact with the system by dragging and dropping these example images or by uploading your own hand-written district name image.")
    
    # Add example images in the sidebar
    st.sidebar.markdown("<h3 style='color: #3366cc;'>Example Images:</h3>", unsafe_allow_html=True)
    example_image_paths = [
        "word_9_AH_16.jpg",
        "word_13_AH_18.jpg",
        "word_28_AH_18.jpg"
    ]
    for i, image_path in enumerate(example_image_paths):
        image = cv2.imread(image_path)
        st.sidebar.image(image, caption=f"Example {i+1}", use_column_width=False, width=150)
        

    #  input field 
    st.markdown("<hr style='border-top: 3px dotted green;'>", unsafe_allow_html=True)
    st.markdown("<hr style='border-top: 3px dotted green;'>", unsafe_allow_html=True)
    # st.markdown("<hr style='border-top: 10px dashed yellow;'>", unsafe_allow_html=True)
    col11, col22 =st.columns(2)
    
    with st.form(key='my_form'):
        st.markdown("<h3 style='color: #3366cc;'>Upload an Image:</h3>", unsafe_allow_html=True)
        file = st.file_uploader("", type=["jpg", "jpeg", "png", "gif"], key="fileUploader", help="Accepted formats: JPG, JPEG, PNG, GIF")
        submit_button = st.form_submit_button(label='Predict')

        if submit_button and file is not None:
            # Read the image and preprocess it
            image = cv2.imdecode(np.fromstring(file.read(), np.uint8), cv2.IMREAD_COLOR)
            preprocessed_image = preprocess_image(image)
            
            # Make predictions using the loaded model
            prediction = model.predict(preprocessed_image)
            predicted_label = lb.inverse_transform(prediction)[0]

            # Convert the image to base64 format for displaying it on the web page
            _, img_buffer = cv2.imencode(".jpg", image)
            image_base64 = base64.b64encode(img_buffer).decode("utf-8")

            col1, col3 = st.columns(2)

            # Display the results
            col1.markdown("<h3 style='color: #3366cc;'>Given Image:</h3>", unsafe_allow_html=True)
            col1.image(image, caption="Input Image", use_column_width=False, width=200)
        
            # The model predicts the character as:
            col3.markdown(" <h3 style='color: #3366cc;'>Prediction:</h3>", unsafe_allow_html=True)
            col3.markdown(f"<h3 style='color: green;'>Predicted District: <span style='color: red;'>{predicted_label}</span></h3>", unsafe_allow_html=True)

    st.markdown("<hr style='border-top: 3px dotted green;'>", unsafe_allow_html=True)
    st.markdown("<hr style='border-top: 3px dotted green;'>", unsafe_allow_html=True)
    st.write("We hope you enjoy using our Hand-Written Bengali District Name Recognition System! Feel free to explore the example images, test the model with your own inputs, and experience the efficiency of our Deep Learning model.")
    st.write("Please note that this system is intended for educational and demonstration purposes. If you have any questions or feedback, don't hesitate to reach out to us. Happy predicting!")
    st.write("With this system, you can easily predict the district name of hand-written Bengali district name from images. It utilizes a powerful Deep Learning model that has been trained on a diverse dataset of hand-written district names to provide accurate predictions.")

    

    # Copyright text in bottom middle
    st.markdown(
        """
        <style>
        .footer {
            position: fixed;
            left: 0;
            bottom: 0;
            width: 100%;
            background-color: #3366cc;
            color: white;
            text-align: center;
            padding: 10px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    st.markdown("<div class='footer'>© 2023 Team_AAA. All rights reserved.</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
        
