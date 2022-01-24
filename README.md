# Real-time-emotion-detection [Deep Learning Capstone Project Almabetter]

The goal of the project is to do real time facial emotion recognition of students in a live class so that they can be monitored easily. This is done by using Deep learning algorithms and Open CV.

The project was done by me and my teammate Raghavendra A Kulkarni.

## MobileNet Architecture

![App Screenshot](https://raw.githubusercontent.com/SuhasTantri/Real-time-emotion-detection/branch-1/media/mobile%20net%20architecture.png)

* The MobileNet structure is built on depthwise separable convolutions except for the first layer which is a full convolution. By defining the network in such simple terms we are able to easily explore
network topologies to find a good network. All layers are followed by a batchnorm and ReLU nonlinearity with the exception
of the final fully connected layer which has no nonlinearity and feeds into a softmax layer for classification. Counting depthwise and pointwise convolutions as separate layers, MobileNet has 28 layers.

## Transfer Learning

![App Screenshot](https://raw.githubusercontent.com/SuhasTantri/Live-Facial-Emotion-Recognition/branch-1/images/transfer%20learning%20image.jpeg)


* Transfer learning is a research problem in machine learning (ML) that focuses on storing knowledge gained while solving one problem and applying it to a different but related problem.

* In this project Mobile-Net transfer learning is used along with computer vision for real time Facial emotion recognition through webcam. A streamlit web app has also been built.

* The model is trained on FER-2013 dataset which has 5 emotions classes namely 'Happy','Neutral','Fear','Angry' and 'Disgust'.
* The model gives an accuracy of 81 % on train data and 76 % on test data.

Here is an image showing the confusion matrix of the model classification

![App Screenshot](https://raw.githubusercontent.com/SuhasTantri/Live-Facial-Emotion-Recognition/branch-1/images/model_confusion_matrix.jpg)

{0: 'Angry', 1: 'fear', 2: 'Happy', 3: 'Neutral'}

* The confusion matrix indicates that the model performs better on Happy and Neutral images.


## Demo
 ![streamlit demo gif (3)](https://user-images.githubusercontent.com/88608896/150626372-3d821423-9c33-481e-b35b-0aa662b7ce19.gif)


## Links

* [Transfer learning model](https://github.com/SuhasTantri/Real-time-emotion-detection/blob/branch-1/emotion_detection_model.ipynb)

* [Real time Facial emotion recognition using OpenCV](https://github.com/SuhasTantri/Real-time-emotion-detection/blob/branch-1/emotion_detection.ipynb)

* [Streamlit web app in streamlit cloud](https://share.streamlit.io/suhastantri/real-time-emotion-detection/branch-1/app.py)  

* [Deployed app in Heroku](https://realtime-emotion-app.herokuapp.com)

* The app may take a couple of minutes to load as I am using free-tier account of Heroku and streamlit cloud and the slug size is 491 MB while the maximum limit is 500 MB. If the app does not open. Try rebooting the app and running it again. Thanks

