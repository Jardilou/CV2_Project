import cv2
import numpy as np
from sklearn.cluster import KMeans
import pytesseract as tess

tess.pytesseract.tesseract_cmd = 'tess/tesseract.exe'

#################################################################################################################################
#STEP 1 : Resize the image for better enhancemants (This simple operation fixed a fewx issues)

def resize_image(image):
    #Remember : image.shape[0] returns original width, image.shape[1] returns original height. 
    # This is simply a rule of 3 to have a bigger image using an interpolation method
    width = 450
    height = int(image.shape[0] * (width / image.shape[1]))
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)


#################################################################################################################################
#STEP 2 : Highlight the colors by applying an arbitrary mask for each color

#At first I tried working with the rgb color workspace but I realized later that working with YCrCb yielded better results using K-means clustering

def highlight_colors(image):
    
    # Convert the frame to HSV color space
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define HSV ranges for the desired colors
    color_ranges = {
        "red": [(0, 50, 50), (10, 255, 255)],  
        "blue": [(100, 50, 50), (140, 255, 255)],
        "green": [(40, 50, 50), (80, 255, 255)],
        "yellow": [(20, 50, 50), (30, 255, 255)],
        "purple": [(130, 50, 50), (160, 255, 255)],
        "orange": [(10, 50, 50), (20, 255, 255)],
        "rose": [(160, 50, 50), (170, 255, 255)]
    }
    # Create masks for each color 
    masks = {}
    for color, ranges in color_ranges.items():
        
        lower, upper = ranges
        masks[color] = cv2.inRange(hsv, np.array(lower), np.array(upper))
    
    # Display each color mask separately for visualization
    for color, mask in masks.items():
        cv2.imshow(f"{color} Mask", mask)

   
    # Break the loop on 'q' key press
    cv2.waitKey(0)
    cv2.destroyAllWindows()



#################################################################################################################################
#STEP 3 : Cluster the image using K means algorithm to group each pixel with their closest color

def clustering(image):
    # Convert the image from BGR to YCrCb color space. YCrCb represents Luminance, Red Chrominance component, Blue Chrominance Component
    image_ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)

    # Extract the Cr channel (index 1) for clustering
    cr_channel = image_ycrcb[:, :, 1]

    # Flatten the 2D Cr channel into a 1D array of pixels because K-means needs 1D data. 
    pixels = cr_channel.reshape(-1, 1)

    # Since the images are dichromic, we will need 2 clusters
    num_clusters = 2

    # Apply K-means algorithm clustering to group pixels into 2 clusters where each pixel will be grouped in the cluster with the closest color
    kmeans = KMeans(n_clusters=num_clusters, n_init=30, max_iter=500, random_state=42)
    kmeans.fit(pixels)

    # Retrieve the cluster labels, which is the cluster to which each pixel belongs for each pixel 
    # and reshape them back into the original image dimensions
    cluster_labels = kmeans.labels_.reshape(cr_channel.shape[:2])
    labels = [0,1]
    # Create an blank output image to represent the clusters with the right dimensions
    clustered_image = np.zeros_like(cluster_labels, dtype=np.uint8)

    # Assign black or white values to clusters based on their cluster labvel.
    for i, label in enumerate(labels):
        grayscale_value = 255  * i
        clustered_image[cluster_labels == label] = grayscale_value

    return clustered_image





#################################################################################################################################
#STEP 4 : Blur the image to connect isolated pixels and get a better shape
#A low kernel size didn't distinguish the 7 from the 9 as there was a little bit of noise but a high kernel size didn't recognize the 1 
#as the "pointy end" is too smooth

def blur_image(image,kernel_size ):
    # Blur the image to connect the dots. Attention : kernel size should be a positive odd integer
    blur = cv2.medianBlur(image,kernel_size )
    return blur



#################################################################################################################################
#STEP 5 : Extract the text using the pytesseracy algorithm (quite simple but getting the right config is a bit tricky)

def extract_text(image):

    # Use PyTesseract to detect numbers. psm 8 recognises best images with a single number as it treats the image as a single word
    config = "--psm 8 --oem 3 -c tessedit_char_whitelist=0123456789"
    extracted_text = tess.image_to_string(image, config=config)

    #psm 10 detects best when 2 numbers are present even though it treats the image as a single character
    if extracted_text == "":
        config_single_word = "--psm 10 --oem 3 -c tessedit_char_whitelist=0123456789"
        extracted_text = tess.image_to_string(image, config=config_single_word)
    return extracted_text



#################################################################################################################################
#STEP 6 : Display the images


def display(path):
    image = cv2.imread(path)
    highlight_colors(image)
    resized_image = resize_image(image)
    clustered = clustering(resized_image)
    blurred = blur_image(clustered,kernel_size )
    extracted_text = extract_text(blurred)
    print("Extracted Text:", extracted_text)

    # Display the images
    # Display the original, clustered and blurred
    cv2.imshow("Original", image)
    cv2.imshow("Clustered Image", clustered)
    cv2.imshow("Blurred Image", blurred)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


paths = [ "Ishi0.png", "Ishi1.png", "Ishi2.png", "Ishi3.png", "Ishi4.png", "Ishi5.png", "Ishi6.png", "Ishi7.png", "Ishi8.png", "Ishi9.png" ]
kernel_size = 31

#paths = ["Ishi0.png", "Ishi3.png"] #Use a lower blurring kernel_size parameter to see the error

#The image where the number is detected only with a lower kernel size for the blur function such as 11 :
#paths = ["Ishi7.png"]

for path in paths:
    display(path)


#################################################################################################################################
#STEP 6 : Test on a live camera feed

def Ishihara_on_video():
    cam = cv2.VideoCapture(0)
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    while True:
        kernel_size = 31
        ret, frame = cam.read()
        # Convert to grayscale for circle detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Apply blur to reduce noise
        blur = cv2.blur(gray, (5, 5))
        # Detect circles using HoughCircles method. It returns a tuple containing the x-coordinate of the circle's center (float), 
        # the y-coordinate of the circle's center (float) and the radius of the circle (float)

        circles = cv2.HoughCircles(blur, method=cv2.HOUGH_GRADIENT,dp=1,minDist=400,param1=50,param2=13,minRadius=40,maxRadius=175)
        if circles is not None:
            # Process each detected circle
            for circle in circles[0, :]:
                #Convert the x center, y center and radius to int
                x, y, radius = map(int, circle)
                #Create a custom zone to work in
                working_zone = frame[y - radius:y + radius, x - radius:x + radius]                
                if working_zone.size > 0:
                    #Use all previous operations on the circle
                    resized_image = resize_image(working_zone)
                    clustered = clustering(resized_image)
                    blurred_and_clustered = blur_image(clustered, kernel_size)
                    extracted_text = extract_text(blurred_and_clustered)
                    if (extracted_text is not None):
                        print(f"Extracted Text: {extracted_text}")
                    
                    # Draw the circle on the original frame
                    cv2.circle(frame, (x, y), radius, (0, 255, 0), 2)
                    cv2.circle(frame, (x, y), 2, (0, 0, 255), 3)
        cv2.imshow('Detected', frame) 
        # Press the 'ESC' key to stop
        if cv2.waitKey(1) & 0xFF == 27:
            break
    cam.release()
    cv2.destroyAllWindows()


#Ishihara_on_video()