from keras.models import load_model
from collections import deque
import numpy as np
import pickle
import cv2
model=load_model(r"C:\Users\S0m3_7h1ng\Documents\Summer_Intern\video_classification_model\videoclassificationmodel")
lb=pickle.load(open(r"C:\Users\S0m3_7h1ng\Documents\Summer_Intern\model\videoclassificationbinarizer.pickle","rb"))
outputvideo=r"C:\Users\S0m3_7h1ng\Documents\Summer_Intern\video_classification_model\demo_output.avi"
mean=np.array([123.68,116.779,103.939][::1],dtype="float32")
Queue=deque(maxlen=128)


capture_video=cv2.VideoCapture(r"C:\Users\S0m3_7h1ng\Documents\Summer_Intern\video_classification_model\final.mp4")
writer=None
(Width,Height)=(None,None)

while True:
	print(writer,end=" ")
	(taken,frame)=capture_video.read()
	if not taken:
		break
	if Width is None or Height is None:
		(Width,Height) = frame.shape[:2]

	output=frame.copy()
	frame=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
	frame=cv2.resize(frame,(244,244)).astype("float32")
	frame-=mean
	preds=model.predict(np.expand_dims(frame,axis=0))[0]
	Queue.append(preds)
	results=np.array(Queue).mean(axis=0)
	i=np.argmax(results)
	label=lb.classes_[i]
	text="They r playing : {}".format(label)
	cv2.putText(output,text,(45,60),cv2.FONT_HERSHEY_SIMPLEX,1.25,(255,0,0),5)

	if writer is None:
		fourcc=cv2.VideoWriter_fourcc(*"MJPG")
		writer=cv2.VideoWriter("outputvideo.mp4",fourcc,30,(Width,Height),True)
	writer.write(output)
	cv2.imshow("in progress",output)
	key=cv2.waitKey(1)&0xFF

	if key==ord("q"):
		break
	print(writer)
print("Finalizing..")
writer.release()
capture_video.release()

