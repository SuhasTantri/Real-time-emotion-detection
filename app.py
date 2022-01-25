import cv2
import time
import numpy as np
from PIL import Image
import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import img_to_array
from streamlit_webrtc import VideoTransformerBase, webrtc_streamer

# importing the necessary files
faceCascade = cv2.CascadeClassifier(r"haarcascade_frontalface_default.xml")
model = load_model(r"emotion detection model.h5")

emotion_labels = ['Angry','Fear','Happy','Neutral']

# class that captures real time webcam feed
class VideoTransformer(VideoTransformerBase):
    def transform(self, frame):
        font = cv2.FONT_HERSHEY_TRIPLEX
        img = frame.to_ndarray(format="bgr24")
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(gray,1.3,1)
        

        for (x,y,w,h) in faces:
                x = x - 5
                y = y + 7
                w = w + 10
                h = h + 2
                roi_grey =  gray[y:y+h,x:x+w]
                roi_color = img[y:y+h,x:x+w]
                cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
                faces = faceCascade.detectMultiScale(roi_grey)
                if len(faces)==():
                    raise IOError('Face not detected')
                    
                else:
                    for (ex,ey,ew,eh) in faces:
                        face_roi = roi_color[ey:ey+eh,ex:ex+ew]
                        
                        final_image = cv2.resize(face_roi,(48,48))
                        final_image = np.expand_dims(final_image,axis=0)
                        final_image = final_image/255.0

                        predictions=model.predict(final_image)
                        label = emotion_labels[predictions.argmax()]
                        cv2.putText(img,label, (50,60),font,2, (255,0,0),3)
        return img

# Function to load image
@st.cache 
def load_image(image_):
    pic=Image.open(image_)
    return pic

# Funcation to return emotion \label from image
def result(img):
    img=np.array(img)
    font = cv2.FONT_HERSHEY_DUPLEX
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray)

    for (x,y,w,h) in faces:
                x = x - 5
                y = y + 7
                w = w + 10
                h = h + 2
                roi_grey =  gray[y:y+h,x:x+w]
                roi_color = img[y:y+h,x:x+w]
                cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
                faces = faceCascade.detectMultiScale(roi_grey)
                if len(faces)==():
                    raise IOError('Face not detected')
                    
                else:
                    for (ex,ey,ew,eh) in faces:
                        face_roi = roi_color[ey:ey+eh,ex:ex+ew]
                        
                        final_image = cv2.resize(face_roi,(48,48))
                        final_image = np.expand_dims(final_image,axis=0)
                        final_image = final_image/255.0

                        predictions=model.predict(final_image)
                        label = emotion_labels[predictions.argmax()]
                        cv2.putText(img,label, (50,60),font,2, (255,0,0),3)
    return label



# main function
def main():
    st.title("Face Emotion Detection App")

    activities = ["Basic emotion classification",'Live video stream emotion detection','About']
    choice = st.sidebar.selectbox('Choose',activities)
    

    if choice == "Live video stream emotion detection":
        st.subheader("Real time face emotion detection")
        webrtc_streamer(key="example",video_processor_factory=VideoTransformer)
        
    elif choice=='About':
       
        st.subheader("Realtime Face Emotion Detection Created By Suhas, Almabetter Data Science Trainee Using OpenCV, Transfer Learning MobileNet Model And Streamlit.\n The app has two functionalities. Real time emotion detection and Emotion detection of an uploaded image.\n
                     I hope you enjoy the experience! ")

    else:
        st.subheader("Upload image of face")
        image_file=st.file_uploader('Upload Image',type=['jpeg','png','jpg','jfif'])
        try:
            if image_file != None:
                img=load_image(image_file)
                st.text("original image")
                progress = st.progress(0)
                for i in range(100):
                    time.sleep(0.01)
                    progress.progress(i+1)
                st.image(image_file)

                if st.button("Detect emotion"):
                    progress = st.progress(0)
                    for i in range(100):
                        time.sleep(0.05)
                        progress.progress(i+1)
                    label = result(img)
                st.success(f"The detected emotion of the person is'{label}'")
            else:
                st.warning("No image uploaded yet")
        except Exception:
            pass

# calling main function
main()
