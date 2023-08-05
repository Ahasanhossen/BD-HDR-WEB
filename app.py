import streamlit as st
import cv2
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from preprocess import preprocess_image
import base64
from streamlit_drawable_canvas import st_canvas

 # Set the page title and favicon
st.set_page_config(page_title="Hand-Written Image Recognition", page_icon="plot_words.png", layout="wide",initial_sidebar_state="expanded")
st.markdown('<link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/css/bootstrap.min.css" integrity="sha384-Gn5384xqQ1aoWXA+058RXPxPg6fy4IWvTNh0E263XmFcJlSAwiGgFAW/dAiS6JXm" crossorigin="anonymous">', unsafe_allow_html=True)

# open css file
with open('style.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# Load the trained CNN model 
model = load_model("cnn_model.h5", compile=False)
lb = pickle.load(open("label_binarizer.pkl", "rb"))

# Streamlit app 
def main():

    # Top bar with blue background and white text
    st.markdown("""
    <nav class="navbar fixed-top navbar-expand-lg navbar-dark" style="background-color: #3498DB;">
         <a class="navbar-brand" style="color: white;text-align: center;">BD-HDR</a>
    </nav>
    """, unsafe_allow_html=True)
    
    st.sidebar.markdown("""
    <nav class="navbar fixed-top navbar-expand-lg navbar-dark" style="background-color: #3498DB;top:2px;padding:7px;width:244px;">
         <a class="navbar-brand" style="color: white;text-align: center; ">BD-HDR</a>
    </nav>
    """, unsafe_allow_html=True)
  
    # Introduction and instructions
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
        

    # input field 
    col1, col2 = st.columns(2)
    # From for predict draw word
    with col1:
      with st.form('Form1',clear_on_submit=True):
        # Free drawing board
        st.markdown("<h3 style='color: #3366cc;'><center>Draw District Name:</center></h3>", unsafe_allow_html=True)
        drawing = st_canvas(
        fill_color="rgba(255, 255, 255, 1)",
        stroke_width=6,
        stroke_color="black",
        background_color="white",
        height=100,
        width=300,
        key="full_app",
                          )
        submitted2 = st.form_submit_button('predict')
     
        if submitted2 and drawing is not None:
            image=np.array(drawing.image_data)
            preprocessed_image = preprocess_image(image)
            
            # Make predictions using the loaded model
            prediction = model.predict(preprocessed_image)
            predicted_label = lb.inverse_transform(prediction)[0]
            confidence = prediction.max()
        
            if confidence>0.9:
                col1, col3 = st.columns(2)
                # Display the input image
                col1.markdown("<h3 style='color: #3366cc;'>Given Image:</h3>", unsafe_allow_html=True)
                col1.image(image, caption="Input Image", use_column_width=False, width=200)
                # The model predicts 
                col3.markdown(" <h3 style='color: #3366cc;'>Prediction:</h3>", unsafe_allow_html=True)
                col3.markdown(f"<h3 style='color: green;'>Predicted District: <span style='color: red;'>{predicted_label}</span></h3>", unsafe_allow_html=True)
                # Max probability
                # st.markdown(f"<p>Confidence Level: {confidence:.2f}</p>", unsafe_allow_html=True)
            else:
                 st.warning('Please Input Appropriate Word', icon="⚠️")
    # From for predict input image
    with col2:
     with st.form('Form2',clear_on_submit=True):
       # Image Input field
       st.markdown("<h3 style='color: #3366cc;'><center>Upload an Image:</center></h3>", unsafe_allow_html=True)
       file = st.file_uploader("", type=["jpg", "jpeg", "png", "gif"], key="fileUploader", help="Accepted formats: JPG, JPEG, PNG, GIF")
       submitted1 = st.form_submit_button('Predict')
       if submitted1:
        if file is not None:
            # If the user uploaded an image, read and preprocess it
             image = cv2.imdecode(np.fromstring(file.read(), np.uint8), cv2.IMREAD_COLOR)
             preprocessed_image = preprocess_image(image)
            
            # Make predictions using the loaded model
             prediction = model.predict(preprocessed_image)
             predicted_label = lb.inverse_transform(prediction)[0]
             confidence = prediction.max()
             if confidence>0.9:
                col1, col3 = st.columns(2)
                
                # Display the results
                col1.markdown("<h3 style='color: #3366cc;'>Given Image:</h3>", unsafe_allow_html=True)
                col1.image(image, caption="Input Image", use_column_width=False, width=200)
                
                # The model predicts the character as:
                col3.markdown(" <h3 style='color: #3366cc;'>Prediction:</h3>", unsafe_allow_html=True)
                col3.markdown(f"<h3 style='color: green;'>Predicted District: <span style='color: red;'>{predicted_label}</span></h3>", unsafe_allow_html=True)
                
        
             else:
                 st.warning('Please Input Appropriate Image', icon="⚠️")
        else:
            st.warning('Please Input an Image', icon="⚠️")
            
        
            

    

        
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
        