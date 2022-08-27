from imutils import paths
import cv2
from numpy import asarray
from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
from torchvision import datasets
from torch.utils.data import DataLoader
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import os
import datetime
from retinaface import RetinaFace
from deepface import DeepFace
from docx import Document
import docx
import matplotlib.image as mpimg
import glob
import sqlalchemy as db
import base64
import time

start_time = time.time()

# Database connection
engine = db.create_engine('postgresql://postgres:ers12345@localhost:5432/postgres')
connection = engine.connect()
metadata = db.MetaData()
test=db.Table('test',metadata, autoload=True, autoload_with=engine)

#Image Show
imagePaths = list(paths.list_images('C:/Users/B2HD High End Rigs/Desktop/multiple_img_recogn/Recognize/'))
print(len(imagePaths))
for imagePath in imagePaths:
    image = cv2.imread(imagePath)
    #plt.imshow(image[:,:,::-1])
    #plt.show()
    
#Crop Face
path = 'C:/Users/B2HD High End Rigs/Desktop/multiple_img_recogn/CF_Totrain/'
count1=0
for imagePath in imagePaths:
    image = cv2.imread(imagePath)         
    obj = RetinaFace.detect_faces(imagePath)
    #print(len(obj.keys()))
    for key in obj.keys():
        identity = obj[key]
        #print(identity)
        facial_area = identity["facial_area"]
        #cv2.rectangle(img, (facial_area[2],facial_area[3]),(facial_area[0],facial_area[1]), (0, 0, 0), 1)
        #print(facial_area[0])
        #print(facial_area[1])
        #print(facial_area[2])
        #print(facial_area[3])
        x1 = facial_area[0]
        y1 = facial_area[1]
        x2 = facial_area[2]
        y2 = facial_area[3]
        imgCrop = image[y1-20:y2+20, x1-20: x2+20]
        #imgCrop1 = cv2.resize(imgCrop, 300, 300)
        #cv2.imwrite(key+".jpg",imgCrop)
        try:
            cv2.imwrite(os.path.join(path , str(count1)+".jpg"), imgCrop)
            count1+=1
        except Exception as e:
            print('Ignoring Exception', e)      
print("Cropped Face: ", count1)


#Recognize Face
imagePaths = list(paths.list_images('C:/Users/B2HD High End Rigs/Desktop/multiple_img_recogn/CF_Totrain/'))
print(len(imagePaths))

mtcnn = MTCNN(image_size=240, margin=0, min_face_size=20) # initializing mtcnn for face detection
resnet = InceptionResnetV1(pretrained='vggface2').eval() # initializing resnet for face img to embeding conversion
def face_match(img_path, data_path): # img_path= location of photo, data_path= location of data.pt 
    # getting embedding matrix of the given img
    img = Image.open(img_path)
    face, prob = mtcnn(img, return_prob=True) # returns cropped face and probability
    emb = resnet(face.unsqueeze(0)).detach() # detech is to make required gradient false
    
    saved_data = torch.load('data.pt') # loading data.pt file
    embedding_list = saved_data[0] # getting embedding data
    name_list = saved_data[1] # getting list of names
    dist_list = [] # list of matched distances, minimum distance is used to identify the person
    
    for idx, emb_db in enumerate(embedding_list):
        dist = torch.dist(emb, emb_db).item()
        dist_list.append(dist)
        
    idx_min = dist_list.index(min(dist_list))
    return (name_list[idx_min], min(dist_list))

count=0
count1=0
count2=0
for imagePath in imagePaths:
    image = cv2.imread(imagePath)
    try:
        result = face_match(imagePath, 'data.pt')
        if result[1] <= 0.40:
            print(result)
            temp = result[0]
            #read from db
            query = db.select([test.columns.Name]).where(test.columns.id == temp)
            name = connection.execute(query).scalar()
            #retrieve fb_link
            query = db.select([test.columns.Link]).where(test.columns.id == temp)
            Facebook_link = connection.execute(query).scalar()
            #retrieve work
            query = db.select([test.columns.Work]).where(test.columns.id == temp)
            work= connection.execute(query).scalar()
            print("Name -  "+str(name))
            print("Facebook link -  "+str(Facebook_link))
            print("Work -  "+str(work)) 
            numpydata = asarray(imagePath)
            print(numpydata)
            print("Face Matched With -  "+str(name))
            #plt.imshow(image[:,:,::-1])
            #plt.show()
            print("--"*30)
            count+=1   
    except Exception as e:
        print('Ignoring Exception', e)  
print("Face Matched: "+str(count))
print("##"*10)
print('\n')

for imagePath in imagePaths:
    image = cv2.imread(imagePath)
    try:
        result = face_match(imagePath, 'data.pt')
        if  result[1]>0.40 and result[1]<=0.60:
            print(result)
            temp = result[0]
            #read from db
            query = db.select([test.columns.Name]).where(test.columns.id == temp)
            name = connection.execute(query).scalar()
            #retrieve fb_link
            query = db.select([test.columns.Link]).where(test.columns.id == temp)
            Facebook_link = connection.execute(query).scalar()
            #retrieve work
            query = db.select([test.columns.Work]).where(test.columns.id == temp)
            work= connection.execute(query).scalar()
            print("Name -  "+str(name))
            print("Facebook link -  "+str(Facebook_link))
            print("Work -  "+str(work)) 
            numpydata = asarray(imagePath)
            print(numpydata)
            print("Face Similar to -  "+str(name))
            #plt.imshow(image[:,:,::-1])
            #plt.show()
            print("--"*30)
            count1+=1   
    except Exception as e:
        print('Ignoring Exception', e) 
print("Face Similar: "+str(count1))
print("##"*10)
print('\n')

for imagePath in imagePaths:
    image = cv2.imread(imagePath)
    try:
        result = face_match(imagePath,'data.pt')
        if result[1] >0.60:
            print(result)
            temp = result[0]
            #read from db
            query = db.select([test.columns.Name]).where(test.columns.id == temp)
            name = connection.execute(query).scalar()
            #retrieve fb_link
            query = db.select([test.columns.Link]).where(test.columns.id == temp)
            Facebook_link = connection.execute(query).scalar()
            #retrieve work
            query = db.select([test.columns.Work]).where(test.columns.id == temp)
            work= connection.execute(query).scalar()
            print("Name -  "+str(name))
            print("Facebook link -  "+str(Facebook_link))
            print("Work -  "+str(work)) 
            numpydata = asarray(imagePath)
            print(numpydata)
            #plt.imshow(image[:,:,::-1])
            #plt.show()
            print("--"*30)
            count2+=1   
    except Exception as e:
        print('Ignoring Exception', e) 
print("Face Not Match: "+str(count2))

'''#Delete images from file
target = 'C:/Users/B2HD High End Rigs/Desktop/multiple_img_recogn/CF_Totrain/'
for x in os.listdir(target):
    if x.endswith('.jpg'):
        os.unlink(target + x)'''


#Recognize Face
imagePaths = list(paths.list_images('C:/Users/B2HD High End Rigs/Desktop/multiple_img_recogn/CF_Totrain/'))
print(len(imagePaths))
mtcnn = MTCNN(image_size=240, margin=0, min_face_size=20) # initializing mtcnn for face detection
resnet = InceptionResnetV1(pretrained='vggface2').eval() # initializing resnet for face img to embeding conversion
def face_match(img_path, data_path): # img_path= location of photo, data_path= location of data.pt 
    # getting embedding matrix of the given img
    img = Image.open(img_path)
    face, prob = mtcnn(img, return_prob=True) # returns cropped face and probability
    emb = resnet(face.unsqueeze(0)).detach() # detech is to make required gradient false
    
    saved_data = torch.load('data1.pt') # loading data.pt file
    embedding_list = saved_data[0] # getting embedding data
    name_list = saved_data[1] # getting list of names
    dist_list = [] # list of matched distances, minimum distance is used to identify the person
    
    for idx, emb_db in enumerate(embedding_list):
        dist = torch.dist(emb, emb_db).item()
        dist_list.append(dist)
        
    idx_min = dist_list.index(min(dist_list))
    return (name_list[idx_min], min(dist_list))

count=0
count1=0
count2=0
for imagePath in imagePaths:
    image = cv2.imread(imagePath)
    try:
        result = face_match(imagePath, 'data1.pt')
        if result[1] <= 0.40:
            print(result)
            temp = result[0]
            #read from db
            query = db.select([test.columns.Name]).where(test.columns.id == temp)
            name = connection.execute(query).scalar()
            #retrieve fb_link
            query = db.select([test.columns.Link]).where(test.columns.id == temp)
            Facebook_link = connection.execute(query).scalar()
            #retrieve work
            query = db.select([test.columns.Work]).where(test.columns.id == temp)
            work= connection.execute(query).scalar()
            print("Name -  "+str(name))
            print("Facebook link -  "+str(Facebook_link))
            print("Work -  "+str(work)) 
            numpydata = asarray(imagePath)
            print(numpydata)
            print("Face Matched With -  "+str(name))
            #plt.imshow(image[:,:,::-1])
            #plt.show()
            print("--"*30)
            count+=1   
    except Exception as e:
        print('Ignoring Exception', e) 
print("Face Matched: "+str(count))
print("##"*10)
print('\n')

for imagePath in imagePaths:
    image = cv2.imread(imagePath)
    try:
        result = face_match(imagePath, 'data1.pt')
        if  result[1]>0.40 and result[1]<=0.60:
            print(result)
            temp = result[0]
            #read from db
            query = db.select([test.columns.Name]).where(test.columns.id == temp)
            name = connection.execute(query).scalar()
            #retrieve fb_link
            query = db.select([test.columns.Link]).where(test.columns.id == temp)
            Facebook_link = connection.execute(query).scalar()
            #retrieve work
            query = db.select([test.columns.Work]).where(test.columns.id == temp)
            work= connection.execute(query).scalar()
            print("Name -  "+str(name))
            print("Facebook link -  "+str(Facebook_link))
            print("Work -  "+str(work)) 
            numpydata = asarray(imagePath)
            print(numpydata)
            print("Face Similar to -  "+str(name))
            #plt.imshow(image[:,:,::-1])
            #plt.show()
            print("--"*30)
            count1+=1   
    except Exception as e:
        print('Ignoring Exception', e) 
print("Face Similar: "+str(count1))
print("##"*10)
print('\n')

for imagePath in imagePaths:
    image = cv2.imread(imagePath)
    try:
        result = face_match(imagePath,'data1.pt')
        if result[1] >0.60:
            print(result)
            temp = result[0]
            #read from db
            query = db.select([test.columns.Name]).where(test.columns.id == temp)
            name = connection.execute(query).scalar()
            #retrieve fb_link
            query = db.select([test.columns.Link]).where(test.columns.id == temp)
            Facebook_link = connection.execute(query).scalar()
            #retrieve work
            query = db.select([test.columns.Work]).where(test.columns.id == temp)
            work= connection.execute(query).scalar()
            print("Name -  "+str(name))
            print("Facebook link -  "+str(Facebook_link))
            print("Work -  "+str(work)) 
            numpydata = asarray(imagePath)
            print(numpydata)
            #plt.imshow(image[:,:,::-1])
            #plt.show()
            print("--"*30)
            count2+=1 
    except Exception as e:
        print('Ignoring Exception', e) 
print("Face Not Match: "+str(count2))

print("##"*10)
time_execution = time.time() - start_time
time_execution = time_execution/60
print("Time Execution: ", time_execution, "m")