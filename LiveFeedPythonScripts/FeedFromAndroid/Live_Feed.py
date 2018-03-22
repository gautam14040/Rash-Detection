# Run this on PC ( or any device that supports python and has required dependencies ) and it would receive and show feed from Android Device

import socket
import cv2
import numpy as np

from sklearn import tree
from sklearn.cross_validation import train_test_split
def ReadData():
    #Data in format [B G R Label] from
    data = np.genfromtxt('./skin-detection-example/data/Skin_NonSkin.txt', dtype=np.int32)

    labels= data[:,3]
    data= data[:,0:3]

    return data, labels

def BGR2HSV(bgr):
    bgr= np.reshape(bgr,(bgr.shape[0],1,3))
    hsv= cv2.cvtColor(np.uint8(bgr), cv2.COLOR_BGR2HSV)
    hsv= np.reshape(hsv,(hsv.shape[0],3))

    return hsv

def TrainTree(data, labels, flUseHSVColorspace):
    if(flUseHSVColorspace):
        data= BGR2HSV(data)

    trainData, testData, trainLabels, testLabels = train_test_split(data, labels, test_size=0.20, random_state=42)

    print trainData.shape
    print trainLabels.shape
    print testData.shape
    print testLabels.shape

    clf = tree.DecisionTreeClassifier(criterion='entropy')
    clf = clf.fit(trainData, trainLabels)
    print clf.feature_importances_
    print clf.score(testData, testLabels)

    return clf

TCP_IP = "192.168.170.200"
TCP_PORT = 8080

sock = socket.socket()
sock.connect((TCP_IP, TCP_PORT))

# cv2.namedWindow("Feed", cv2.WINDOW_AUTOSIZE)

imgData = ''
data, labels= ReadData()
clf= TrainTree(data, labels, True)
while True:
    k = cv2.waitKey(1)
    if k & 0xff is 27:
        break
    data = sock.recv(1024)
    if not data:
        continue
    imgData += data
    a = imgData.find('\xff\xd8')
    b = imgData.find('\xff\xd9')
    if a != -1 and b != -1:
        img = cv2.imdecode(np.fromstring(imgData[a:b + 2], dtype=np.uint8), 1)
        #cv2.imshow("Feed", feed)
	    #cv2.imwrite('./skin-detection-example/s.png',feed)
        data= np.reshape(img,(img.shape[0]*img.shape[1],3))

        data= BGR2HSV(data)
        predictedLabels= clf.predict(data)
        imgLabels= np.reshape(predictedLabels,(img.shape[0],img.shape[1],1))
        # print imgLabels
        cv2.imwrite('current.png',img)
        cv2.imwrite('./skin-detection-example/results/result_HSV.png',((-(imgLabels-1)+1)*255))# from [1 2] to [0 255]
        # cv2.imshow("Feed",((-(imgLabels-1)+1)*255))
        # cv2.imwrite('../results/result_HSV.png',((-(imgLabels-1)+1)*255))# from [1 2] to [0 255]
        # else:
            # cv2.imwrite('../results/result_RGB.png',((-(imgLabels-1)+1)*255))
        imgData = imgData[b + 2:]
