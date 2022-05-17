import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from PIL import Image
import PIL.ImageOps 

x = np.load("image (1).npz")['arr_0']
y=pd.read_csv("labels.csv")["labels"]
classes=["A" , "B" , "C" , "D","E","F","G","H" , "I" , "J" , "K","L","M","N" , "O" , "P" , "Q","R","S","T" , "U" , "V" , "W","X","Y","Z"]
nclasses=len(classes)

x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=10,train_size=7500,test_size=2500)
x_train_scale=x_train/255.0
x_test_scale=x_test/255.0

lr=LogisticRegression(solver="saga" , multi_class="multinomial")
lr.fit(x_train_scale , y_train)

def get_prediction(image):
    impil=Image.open(image)
    img_bw=impil.convert("L")
    image_bw_resized=img_bw.resize((28,28) , Image.ANTIALIAS)
    pixel_filter=20
    min_pixel=np.percentile(image_bw_resized , pixel_filter)
    image_bw_resized_inverted_scale=np.clip(image_bw_resized-min_pixel,0,255)
    max_pixel=np.max(image_bw_resized)
    image_bw_resized_inverted_scale=np.asarray(image_bw_resized_inverted_scale)/max_pixel
    test_sample=np.array(image_bw_resized_inverted_scale).reshape(1,784)
    test_pred=lr.predict(test_sample)
    return test_pred[0]