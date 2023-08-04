import cv2

def preprocess_image(image):
    # Resize the image
    new_weight = 240
    new_height = 80
    

    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret,candidate_bw = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    resized_image = cv2.resize(candidate_bw, (new_weight, new_height))

    # Reshape the image to match the input shape of the model
    preprocessed_image = resized_image.reshape(-1, new_height, new_weight, 1)

    return preprocessed_image