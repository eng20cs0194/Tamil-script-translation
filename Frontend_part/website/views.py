from flask import Blueprint,render_template,request,jsonify
from PIL import Image
from website.model_part.full_pipeline import predict_
import cv2
import numpy as np
import warnings 
warnings.filterwarnings("ignore",)
from website import app
import os
import pythoncom

import win32com.client



views=Blueprint('views',__name__)

@views.route('/',methods=['GET'])
def home():
    
    return render_template('index.html')


@views.route('/get_image',methods=['POST'])
def get_image():

    data=request.files

    image=data['image']
    name=image.filename
    


    try:
        path=f'{app.config["UPLOADER_FOLDER"]}/{image.filename}'
        for i in os.listdir(app.config['UPLOADER_FOLDER']):
            os.remove(f'{app.config["UPLOADER_FOLDER"]}/{i}')
        
        image.save(path)
        prediction=predict_(path)
        

        return jsonify({"prediction":prediction,'name':name})

        
    except Exception as e:
        print(f'Image not saved due to {e}')

        request.files.get('image')

        return jsonify({"prediction":None})
@views.route('/speak',methods=['POST'])
def voice_talk():
    pythoncom.CoInitialize()
    data=request.json 
    word=data['speak']
    print(word)
    
    speaker=win32com.client.Dispatch('SAPI.SpVoice')
    speaker.Speak(word) 

    return "200"
    
    




    



        
    
    

        
            
        

            


    




        
