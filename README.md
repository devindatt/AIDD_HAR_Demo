# Human Activty Recognition

[![P|Python](https://cldup.com/dTxpPi9lDf.thumb.png)](https://nodesource.com/products/nsolid)

RestNet-34 model using the Kinetic dataset to predict exercises in a video or webcam:

Kinetics dataset, the dataset used to train our human activity recognition model.

Extending ResNet, which typically uses 2D kernels, to instead leverage 3D kernels, enabling us to include a spatiotemporal component used for activity recognition.

Here we applied the human activity recognition model using the OpenCV library and the Python programming language to custom videos with only exercises.



[![HAR](https://github.com/devindatt/AIDD_DemoDay/blob/master/Assets/skeleton_data01.png)](https://github.com/devindatt/AIDD_DemoDay/blob/master/Assets/skeleton_data01.png)


Our project consists of the following files:

action_recognition_kinetics.txt : The class labels for the Kinetics dataset.
resnet-34_kinetics.onx : Hara et al.’s pre-trained and serialized human activity recognition convolutional neural network trained on the Kinetics dataset.
human_activity_reco.py : Our human activity recognition script which samples N frames at a time to make an activity classification prediction.
human_activity_reco_deque.py : A similar human activity recognition script that implements a rolling average queue. This script runs slower to run but is more accurate when making predictions

### Package Requirements:
Human Activity Recognition models require at least OpenCV 4.1.2.


### Deployment

1) Clone the repo with all pre-trained human activity recognition model, Python + OpenCV source code to your local drive.

2) Copy any sample exercise video to this cloned folder on your local hard drive.

3) Open your favorite Terminal and run these commands.

Command to run app:
```sh
$ python human_activity_reco_deque.py --model resnet-34_kinetics.onnx \
	--classes action_recognition_kinetics.txt \
	--input example_activities.mp4
[INFO] loading human activity recognition model...
[INFO] accessing video stream...
```

If your are running an older version of OpenCV you might receive the following error:
```sh
net = cv2.dnn.readNet(args["model"])
cv2.error: OpenCV(4.1.0) /Users/home_directory/build/skvark/opencv-python/opencv/modules/dnn/src/onnx/onnx_importer.cpp:245: error: (-215:Assertion failed) attribute_proto.ints_size() == 2 in function 'getLayerParams'
```
If you receive that error you need to upgrade your OpenCV install to at least OpenCV 4.1.2.

If all deployed correctly you should see another window open up (might be behind your terminal window) and you should see predictions being made on your video:

Note: the frame rate might be extremely slow if you are not using a GPU.

[![HAR](https://github.com/devindatt/AIDD_DemoDay/blob/master/Assets/activity_composition_film4.gif)](https://youtu.be/GHqioYqqkSc)



While these predictions are not perfect, it is still performing quite well given the simplicity of our technique (converting ResNet to handle 3D inputs versus 2D ones).

Human activity recognition is far from solved, but with deep learning and Convolutional Neural Networks, we’re making great strides.


