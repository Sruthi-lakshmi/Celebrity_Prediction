import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import tensorflow as tf
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tensorflow.keras.callbacks import EarlyStopping
from tqdm import tqdm

image_dir='cropped'
lionel_messi = os.listdir(image_dir+ r'\lionel_messi')
maria_sharapova = os.listdir(image_dir+ r'\maria_sharapova')
roger_federer = os.listdir(image_dir+ r'\roger_federer')
serena_williams = os.listdir(image_dir+ r'\serena_williams')
virat_kohli = os.listdir(image_dir+ r'\virat_kohli')

print('The length of Lionel Messi images is',len(lionel_messi))
print('The length of Maria Sharapova images is',len(maria_sharapova))
print('The length of Roger federer images is',len(roger_federer))
print('The length of Serena Williams images is',len(serena_williams))
print('The length of Virat Kohli images is',len(virat_kohli))

dataset=[]
label=[]
img_size=(128,128)


for i , image_name in tqdm(enumerate(lionel_messi),desc="lionel_messi"):
    if(image_name.split('.')[1]=='png'):
        image=cv2.imread(image_dir+'/lionel_messi/' +image_name)
        image=Image.fromarray(image,'RGB')
        image=image.resize(img_size)
        dataset.append(np.array(image))
        label.append(0)
        
        
for i ,image_name in tqdm(enumerate(maria_sharapova),desc="maria_sharapova"):
    if(image_name.split('.')[1]=='png'):
        image=cv2.imread(image_dir+'/maria_sharapova/' +image_name)
        image=Image.fromarray(image,'RGB')
        image=image.resize(img_size)
        dataset.append(np.array(image))
        label.append(1)

for i ,image_name in tqdm(enumerate(roger_federer),desc="roger_federer"):
    if(image_name.split('.')[1]=='png'):
        image=cv2.imread(image_dir+'/roger_federer/' +image_name)
        image=Image.fromarray(image,'RGB')
        image=image.resize(img_size)
        dataset.append(np.array(image))
        label.append(2)  

for i ,image_name in tqdm(enumerate(serena_williams),desc="serena_williams"):
    if(image_name.split('.')[1]=='png'):
        image=cv2.imread(image_dir+'/serena_williams/' +image_name)
        image=Image.fromarray(image,'RGB')
        image=image.resize(img_size)
        dataset.append(np.array(image))
        label.append(3)  

for i ,image_name in tqdm(enumerate(virat_kohli),desc="virat_kohli"):
    if(image_name.split('.')[1]=='png'):
        image=cv2.imread(image_dir+'/virat_kohli/'+image_name)
        image=Image.fromarray(image,'RGB')
        image=image.resize(img_size)
        dataset.append(np.array(image))
        label.append(4)  
        
dataset=np.array(dataset)
label = np.array(label)

print("--------------------------------------\n")
print('Dataset Length: ',len(dataset))
print('Label Length: ',len(label))
print("--------------------------------------\n")


print("--------------------------------------\n")
print("Train-Test Split")
x_train,x_test,y_train,y_test=train_test_split(dataset,label,test_size=0.2,random_state=42)
print("--------------------------------------\n")

print("--------------------------------------\n")
print("Normalising the Dataset. \n")

x_train=tf.keras.utils.normalize(x_train,axis=1)
x_test=tf.keras.utils.normalize(x_test,axis=1)

print("--------------------------------------\n")


# Model Compilation
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dropout(0.1),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(5, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',  # Change to 'categorical_crossentropy' if using one-hot encoding
              metrics=['accuracy'])

# Early stopping
early_stopping = EarlyStopping(monitor='accuracy', patience=50, restore_best_weights=True)

# Model Training
history = model.fit(x_train, y_train, epochs=200, batch_size=128, validation_split=0.1, callbacks=[early_stopping])

# Plot and save accuracy
plt.plot(history.epoch, history.history['accuracy'], label='accuracy')
plt.plot(history.epoch, history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0, 1])
plt.legend(loc='lower right')
plt.savefig(r'celebrity_accuracy_plot.png')

plt.clf()

# Plot and save loss
plt.plot(history.epoch, history.history['loss'], label='loss')
plt.plot(history.epoch, history.history['val_loss'], label='val_loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(loc='upper right')
plt.savefig(r'celebrity_sample_loss_plot.png')

# Model Evaluation Phase
print("Model Evaluation Phase.")
loss, accuracy = model.evaluate(x_test, y_test)
print(f'Accuracy: {round(accuracy * 100, 2)}')

# Classification Report
y_pred = model.predict(x_test)
y_pred_labels = np.argmax(y_pred, axis=1)
print("Classification Report:\n", classification_report(y_test, y_pred_labels))

# Model Prediction
print("Model Prediction.")
class_names = {0: 'Lionel Messi', 1: 'Maria Sharapova', 2: 'Roger Federer', 3: 'Serena Williams', 4: 'Virat Kohli'}

def make_prediction(img, model):
    img = cv2.imread(img)
    img = Image.fromarray(img)
    img = img.resize((128, 128))
    img = np.array(img)
    input_img = np.expand_dims(img, axis=0)
    input_img = tf.keras.utils.normalize(input_img, axis=1) 
    result = model.predict(input_img)
    predicted_class_index = np.argmax(result)
    predicted_celebrity_name = class_names[predicted_class_index]
    print("Predicted Celebrity:", predicted_celebrity_name)

make_prediction(r'cropped\lionel_messi\lionel_messi17.png', model)
make_prediction(r'cropped\maria_sharapova\maria_sharapova18.png', model)
make_prediction(r'cropped\roger_federer\roger_federer5.png', model)
make_prediction(r'cropped\serena_williams\serena_williams22.png', model)
make_prediction(r'cropped\virat_kohli\virat_kohli2.png', model)
