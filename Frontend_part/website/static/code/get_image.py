from browser import document,html,ajax,window,alert
import json

def show_output(res):
    data=res.json
    prediction=data['prediction']
    name=data['name']

    document['display'].clear()


    
    

    
    if prediction is not None:
        title_div=html.DIV()
        image_title=html.LABEL('Image:-')
        image_title.classList.add('image-title')
        title_div<=image_title

        img_div=html.DIV()
        img=html.IMG(src=f"static/temp_images/{name}")
        img.classList.add('image-display')
        img_div<=img

        prediction_label=html.DIV(f'Prediction:-{prediction}')
        prediction_label.classList.add('prediction-label')

        document['display']<=title_div
        document['display']<=img_div
        document['display']<=prediction_label
        data=json.dumps({"speak":prediction})
        ajax.post(
            url='/speak',
            data=data,
            headers={"Content-type":"Application/json"}
        )
        
    
        
        
    else:
        document['display']<=html.LABEL('No image No prediction')

    
    document['form-image'].value=''
    


def send_form(ev):
    f=document['form-image'].files[0]
    
    
    ajax.file_upload(
        url='/get_image',
        file=f,
        method='post',
        field_name='image',
        oncomplete=show_output,
    )
    
    document['display'].clear()



document['Submit_pic'].bind('click',send_form)

