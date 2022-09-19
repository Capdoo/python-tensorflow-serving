import requests
import json
import cv2
import numpy as np
import time
import pandas as pd

image = cv2.imread("imagen.jpg")
image = cv2.resize(image, (224,224))
image = image / 255.0
image = image.astype(np.float32)
image = np.expand_dims(image, axis=0)

labels_path = "labels.csv"
labels_df = pd.read_csv(labels_path)
breed = labels_df["breed"].unique()
id2breed = {i: name for i, name in enumerate(breed)}

start_time = time.time()

url = "https://petsbreed7app.herokuapp.com/v1/models/petsbreed7:predict"
#url = "http://localhost:8501/v1/models/petsbreed7:predict"

data = json.dumps({"signature_name":"serving_default","instances":image.tolist()})
headers = {"content-type":"application/json"}
response = requests.post(url, data=data, headers=headers)
prediction = json.loads(response.text)

breed_estimation = prediction["predictions"][0]
label_idx = np.argmax(breed_estimation)
breed_name = id2breed[label_idx]

#print(prediction["predictions"][0])
print(breed_name)

#print("Time taken", time.time()-start_time)





