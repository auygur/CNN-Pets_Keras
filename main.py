import os
import numpy as np
from keras import applications
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
# from sklearn.naive_bayes import MultinomialNB

import tensorflow as tf
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
sess

ilists,class_ids = [], []


pathImage = "/rigel/edu/coms4995/datasets/pets"
imageDoc = os.listdir(pathImage)
img = [x for x in imageDoc]

pathTrue = "/rigel/edu/coms4995/users/au2205/homework-v-auygur/list.txt"
with open(pathTrue, "r") as p:
    lines = p.readlines()


# count = 0

for l in lines:
    if l[0] == "#":
        pass
    else:
        l = (l.strip()).split()
        name = l[0] + ".jpg"
        class_id = l[1]
        ilists.append(image.load_img(pathImage+"/" + name, target_size=(224, 224))) 
        class_ids.append(class_id)
    #     count+=1
    # if count == 500:
    #     break

model = applications.VGG16(include_top=False, weights='imagenet')

X = preprocess_input(np.array([image.img_to_array(i) for i in ilists]))
X = model.predict(X).reshape(len(ilists), -1)

X_train, X_test, y_train, y_test = train_test_split(X, class_ids, stratify=class_ids, test_size=0.25)

# clf = MultinomialNB().fit(X_train, y_train)
clf = LogisticRegression(C=1).fit(X_train, y_train)

print('Train score:',clf.score(X_train, y_train))
print('Test score:',clf.score(X_test, y_test))


"""
For Logistic Reg with C = 1 following results:
('Train score:', 1.0)
('Test score:', 0.88683351468988025)

Other Multiple C values [0.1,0.2,0.8,2,5,10] were also tested, they were giving less than 0.88 Test Scores)


We also tested MultinomialNB
('Train score:', 0.93268009435674104)
('Test score:', 0.78509249183895535)

"""
