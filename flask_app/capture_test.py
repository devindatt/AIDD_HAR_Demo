import cv2, time
import base64
import json


def capture_video():

	a=0
	i=1
	data = []
	#data['json'] =[]


	ENCODING = 'utf-8'
	JSON_NAME = 'output.json'


	initialtime = time.time()

	video = cv2.VideoCapture(1)

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
	#	img = cv2.flip(img, 0)

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


			JSON_NAME = 'json_test'+str(i)		
	#		print(JSON_NAME)
			data.append(raw_data)

	#		print('length of {} file is {} characters long'.format(JSON_NAME, len(raw_data)))


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

