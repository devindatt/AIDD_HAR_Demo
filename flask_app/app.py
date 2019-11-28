from flask import Flask, render_template, request
from scipy.misc import imsave, imread, imresize
from collections import deque
import numpy as np
import argparse
import imutils
import sys, time, re, os
import cv2
import base64
import json
import pickle

#import capture_test.capture_video as capture_video



#Define the path to the model
#sys.path.append(os.path.abspath)


application = Flask(__name__)

@application.route("/")
def index():

    return render_template("index.html")



@application.route("/run_model")
def run_model():



	infile = open('GymnData.pickle','rb')
	Gym_Data = pickle.load(infile, encoding='latin1')
	infile.close()


	print(Gym_Data)
	print(type(Gym_Data))

#	with open('data.txt') as text_file:
#	    data = json.load(text_file)


	with open('data.json') as json_file:
	    data = json.load(json_file)

#	print('The length of the JSON file is {} records'.format(len(data['people'])))
	print('The length of the JSON file is {} records'.format(len(data)))
	print('The file type of the JSON file is: {}'.format(type(data)))
#	print(data[0])

#	for i in data:
#		print data.keys(i), data.values(i)
#	print(data['images']['vid_test50.jpg'])



#	imgdata = base64.b64decode(imgstring)
#	filename = 'some_image.jpg'  # I assume you have a way of picking unique filenames
#	with open(filename, 'wb') as f:
#	    f.write(imgdata)


	return render_template("run_model.html")


@application.route("/show_webcam")

def show_webcam(mirror=False):


	a=0
	i=1
#	data = []

	data = {}
#	data['images'] = []


	ENCODING = 'utf-8'
	JSON_NAME = 'output.json'


	initialtime = time.time()

	video = cv2.VideoCapture(0)

	#while(True):
	while(video.isOpened()):
		
		a=a+1

		check, frame = video.read()

		if check == False:
			break

	#	print(check)
	#	print(frame)



		gray=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		img = cv2.resize(gray, (0,0), fx = 0.25, fy = 0.25)
		if mirror:
			img = cv2.flip(img, 0)

	#	print(img.shape)

		#cv2.imshow('Capturing', frame)
		cv2.imshow('Capturing', img)


		#Saving every nth frame (ie. n=50) to a json format 

		if i%50 == 0:

			IMAGE_NAME = 'vid_test'+str(i)+'.jpg'
			cv2.imwrite(IMAGE_NAME, img)

			base64_bytes = base64.b64encode(img)
			base64_string = base64_bytes.decode(ENCODING)
			raw_data = {IMAGE_NAME: base64_string}
	#		print(raw_data)


			JSON_NAME = 'json_'+str(i)		
	#		print(JSON_NAME)
	#		if data[JSON_NAME] == 0:
	#			data[JSON_NAME] = base64_string
	#		data[JSON_NAME] = base64_string		

			data.update({JSON_NAME: base64_string})	

#			print('length of {} file is {} characters long'.format(JSON_NAME, len(data['images'])))
			

			imgdata = base64.b64decode(base64_string)
#			imgdata = base64.decodebytes(base64_string)

			filename = JSON_NAME+'_image.png' 
			with open(filename, 'wb') as f:
				f.write(imgdata)




		i+=1

	#	with open('json_array.json', 'w') as open_file:
	#		open_file.write(data)

		key=cv2.waitKey(1)


		#Breaking out of loop if 'q' button is pushed
		if (key & 0xFF) == ord('q'):
			print(" 'q' pressed. Exiting ...")
			break


	print(len(data))
	json_data = json.dumps(data, indent=2)
	with open('data.json', 'w') as open_file:
		open_file.write(json_data)


	endtime = time.time()
	print("Run time is {} in seconds".format(endtime-initialtime))

	print(a)

	video.release()
	cv2.destroyAllWindows()

	return render_template("show_webcam.html")




@application.route("/show_har_test")

# python human_activity_reco_deque.py 
# --model = resnet-34_kinetics.onnx 
# --classes = action_recognition_kinetics.txt 
# --input = example_activities.mp4



def show_har_test(mirror=False):

	# load the contents of the class labels file, then define the sample
	# duration (i.e., # of frames for classification) and sample size
	# (i.e., the spatial dimensions of the frame)
	# CLASSES = open(args["classes"]).read().strip().split("\n")
	CLASSES = open("action_recognition_kinetics.txt").read().strip().split("\n")

	SAMPLE_DURATION = 16
	SAMPLE_SIZE = 112

	# initialize the frames queue used to store a rolling sample duration
	# of frames -- this queue will automatically pop out old frames and
	# accept new ones
	frames = deque(maxlen=SAMPLE_DURATION)

	# load the human activity recognition model
	print("[INFO] loading human activity recognition model...")
	# net = cv2.dnn.readNet(args["model"])
	net = cv2.dnn.readNet("resnet-34_kinetics.onnx")



	# grab a pointer to the input video stream
	print("[INFO] accessing video stream...")
	# video2 = cv2.VideoCapture(args["input"] if args["input"] else 1) #0)
	video2 = cv2.VideoCapture(1)

	# loop over frames from the video stream
	while True:
		# read a frame from the video stream
		(grabbed, frame) = video2.read()

		# if the frame was not grabbed then we've reached the end of
		# the video stream so break from the loop
		if not grabbed:
			print("[INFO] no frame read from stream - exiting")
			break

		# resize the frame (to ensure faster processing) and add the
		# frame to our queue
		frame = imutils.resize(frame, width=680)
		frames.append(frame)

		# if our queue is not filled to the sample size, continue back to
		# the top of the loop and continue polling/processing frames
		if len(frames) < SAMPLE_DURATION:
			continue

		# now that our frames array is filled we can construct our blob
		blob = cv2.dnn.blobFromImages(frames, 1.0, (SAMPLE_SIZE, SAMPLE_SIZE), 
			(114.7748, 107.7354, 99.4750), swapRB=True, crop=True)
		blob = np.transpose(blob, (1, 0, 2, 3))
		blob = np.expand_dims(blob, axis=0)

		# pass the blob through the network to obtain our human activity
		# recognition predictions
		net.setInput(blob)
		outputs = net.forward()
		label = CLASSES[np.argmax(outputs)]

		# draw the predicted activity on the frame
		cv2.rectangle(frame, (0, 0), (300, 40), (0, 0, 0), -1)
		cv2.putText(frame, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
			0.8, (255, 255, 255), 2)

		# display the frame to our screen
		cv2.imshow("Human Activity Recognition page", frame)
		key = cv2.waitKey(1) & 0xFF

		# if the `q` key was pressed, break from the loop
		if key == ord("q"):
			break


	video2.release()
	cv2.destroyAllWindows()

	return render_template("show_har_test.html")


if __name__ == "__main__":
#	port = int(os.environ.get('PORT', 5000))
	application.run(host="0.0.0.0", port=8000)


