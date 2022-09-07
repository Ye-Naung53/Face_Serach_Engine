
import os
import streamlit as st
from imutils import paths
import cv2
from numpy import asarray
from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
from torchvision import datasets
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import os
from retinaface import RetinaFace
import sqlalchemy as db
import base64
from PIL import Image
import pandas as pd





def load_image(image_file):
    img = Image.open(image_file)
    return img

st.title("Face Serch Engine")
image_file = st.file_uploader("Upload Images", type=["png","jpg","jpeg"])

if image_file is not None:

    file_details = {"filename":image_file.name, "filetype":image_file.type,
                              "filesize":image_file.size}
	
    #sst.write(file_details)
    # To View Uploaded Image
    st.image(load_image(image_file),width=320)

    with open(os.path.join("fileDir",image_file.name),"wb") as f:
        f.write((image_file).getbuffer())


search_btt = st.button("Search")

if search_btt:
    imagePaths = list(paths.list_images(os.getcwd()+'/fileDir/'))
    #print(len(imagePaths))
    count1=0
    for imagePath in imagePaths:
        image = cv2.imread(imagePath)         
        obj = RetinaFace.detect_faces(imagePath)   
        for key in obj.keys():
            identity = obj[key]
            facial_area = identity["facial_area"]
            x1 = facial_area[0]
            y1 = facial_area[1]
            x2 = facial_area[2]
            y2 = facial_area[3]
            imgCrop = image[y1-20:y2+20, x1-20: x2+20]
            #imgCrop1 = cv2.resize(imgCrop, 300, 300)
            #cv2.imwrite(key+".jpg",imgCrop)
            try:
                file=os.getcwd()+'/CroppedFace/face'+str(count1)+'.jpg'
                cv2.imwrite(file,imgCrop)
                count1+=1
            except Exception as e:
                print('Ignoring Exception', e)          
    print("Cropped Face")

    imagePaths = list(paths.list_images(os.getcwd()+'/CroppedFace/'))
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
    for imagePath in imagePaths:
        image = cv2.imread(imagePath)
        try:
            result = face_match(imagePath, 'data.pt')
            if result[1] <= 0.40:
                print(result)
                temp = result[0]
                df=pd.read_csv(os.getcwd()+'/test.csv',index_col=[0])
                #read from db
                name=df.loc[[int(temp)],'Name'].astype('string')
                name = name.tolist()
                name = ' '.join([str(elem) for elem in name])
                Facebook_link=df.loc[[int(temp)],'Link'].astype('string')
                Facebook_link = Facebook_link.tolist()
                Facebook_link = ' '.join([str(elem) for elem in Facebook_link])
                print("Name - "+str(name))
                print("Facebook link - "+str(Facebook_link))
                st.text("Face Match with - "+str(name))
                #st.text("Facebook link -  "+str(Facebook_link))
                st.markdown("Facebook link - "+Facebook_link,unsafe_allow_html=True)
                
                count+=1   
            if result[1]>0.40 and result[1]<=0.60:
                print(result)
                temp = result[0]
                #read from db
                df=pd.read_csv(os.getcwd()+'/test.csv',index_col=[0])
                #read from db
                name=df.loc[[int(temp)],'Name'].astype('string')
                name = name.tolist()
                name = ' '.join([str(elem) for elem in name])
                Facebook_link=df.loc[[int(temp)],'Link'].astype('string')
                Facebook_link = Facebook_link.tolist()
                Facebook_link = ' '.join([str(elem) for elem in Facebook_link])
                print("Name - "+str(name))
                print("Facebook link - "+str(Facebook_link))
                st.text("Face similar with - "+str(name))
                #st.text("Facebook link -  "+str(Facebook_link))
                st.markdown("Facebook link - "+Facebook_link,unsafe_allow_html=True)
                count+=1   
            if result[1] >0.60:
                print(result)
                temp = result[0]
                #read from db
                df=pd.read_csv(os.getcwd()+'/test.csv',index_col=[0])
                #read from db
                name=df.loc[[int(temp)],'Name'].astype('string')
                name = name.tolist()
                name = ' '.join([str(elem) for elem in name])
                Facebook_link=df.loc[[int(temp)],'Link'].astype('string')
                Facebook_link = Facebook_link.tolist()
                Facebook_link = ' '.join([str(elem) for elem in Facebook_link])
                print("Name -  "+str(name))
                print("Facebook link -  "+str(Facebook_link))
                st.text("Face not Match ")
                count+=1   

        except Exception as e:
            print('Ignoring Exception', e)  

    target = os.getcwd()+'/fileDir/'
    for x in os.listdir(target):
        if x.endswith('.jpg'):
            os.unlink(target + x)         
    target1 = os.getcwd()+'/CroppedFace/'
    for x in os.listdir(target1):
        if x.endswith('.jpg'):
            os.unlink(target1 + x)  
    
    
