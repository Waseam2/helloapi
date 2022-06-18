from fastapi import  FastAPI, File, UploadFile
import shutil
app = FastAPI()

################### For Model ###################################################

import cv2
import tensorflow as tf
import numpy as np
import sys
from keras.models import load_model

LABELS = ["Abou_al_Haggag_Mosque", "Akhenaten", "Tutankhamun", "amnhoutb_iii", "hatshepsut", "lake_19", "masla_21",
          "nkhtnbo_i", "ramses_ii", "wall_1", "wall_10", "wall_11", "wall_12", "wall_13", "wall_14", "wall_15",
          "wall_16", "wall_17", "wall_18", "wall_2", "wall_3", "wall_4", "wall_5", "wall_6", "wall_7", "wall_8",
          "wall_9", "wall_19"]
def prepare(image_path):
    IMG_SIZE = 250
    img_array = cv2.imread(image_path)
    img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
    return new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 3)


x = tf.keras.Input(shape=(250, 250, 3))
y = tf.keras.layers.Dense(16, activation='softmax')(x)
model = tf.keras.Model(x, y)
model = load_model('model2.h5')
########################################################################################

@app.get("/")
def root():
    return {"result":"hello world 1 1 "}
@app.post("/num")
def ret_num(id):
    return {"result" : id}

@app.post("/img")
async def create_upload_file(image: UploadFile):
    with open("destination.jpeg", "wb") as buffer:
        shutil.copyfileobj(image.file, buffer)
    try:
        prediction = model.predict([prepare("destination.jpeg")])

        a = prediction
        i, j = np.unravel_index(a.argmax(), a.shape)
        a[i, j]
        # print(prediction)
        if (prediction[0][j] >= .6):
            return {"result": LABELS[j]}
        else:
            return {"result": "not_recognized"}

    except:
        return {"result": "incorrect image path"}