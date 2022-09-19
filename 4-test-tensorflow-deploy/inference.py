import requests
import json
import numpy as np
import cv2
import tensorflow as tf
import time

image = cv2.imread("imagen.jpg")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image = cv2.resize(image, (224,224))
image = np.expand_dims(image, axis=0)

#Create inference request
start_time = time.time()

#url = "http://localhost:8501/v1/models/vgg16:predict"
url = "https://vgg16app.herokuapp.com/v1/models/vgg16:predict"



data = json.dumps({"signature_name":"serving_default", "instances":image.tolist()})
print(image.tolist())
headers = {"content-type":"application/json"}
response = requests.post(url, data = data, headers = headers)
prediction = json.loads(response.text)["predictions"]
#print(prediction)

result = tf.keras.applications.imagenet_utils.decode_predictions(np.array(prediction))
print(result)
print("TIme taken", time.time()-start_time)
