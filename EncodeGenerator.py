import cv2
import face_recognition
import pickle
import os

folderPath = '/Users/cholponklv/python/face id/Images'
pathList = os.listdir(folderPath)
imgList = []
studentIDs = []
for path in pathList:
    imgList.append(cv2.imread(os.path.join(folderPath,path)))
    studentIDs.append(os.path.splitext(path)[0])
print(studentIDs)

def findEncodings(imagesList):
    encodeList = []
    for img in imagesList:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)

    return encodeList

encodeListKnown = findEncodings(imgList)
encodeListKnownWithIds = [encodeListKnown,studentIDs]


file = open("EncodeFile.p",'wb')
pickle.dump(encodeListKnownWithIds,file)
file.close()