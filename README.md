# Celebrity Prediction Using CNN
The Model Predict the images of Celebrities such as Lionel Messi, Maria Sharapova, Roger Federer, Serena Williams and Virat Kohli.
the model is trained using Convolutional Neural Network and predicted the images of celebrities. The code has 73.53 accuracy.

## Model Architecture
The model architecture is a simple convolutional neural network (CNN) with the following layers:

* Convolutional layer with 32 filters and a (3, 3) kernel, ReLU activation
* MaxPooling layer with (2, 2) pool size
* Flatten layer
* Dense layer with 256 neurons and ReLU activation
* Dropout layer with a dropout rate of 0.1
* Dense layer with 512 neurons and ReLU activation
* Output layer with 5 neurons (one for each celebrity) and softmax activation

## Libraries
* Python 3
* TensorFlow
* NumPy
* Matplotlib
* OpenCV
* Pillow (PIL)
* Scikit-learn

## Training
The model is compiled using the Adam optimizer and sparse categorical crossentropy loss. It is trained for 200 epochs with early stopping to prevent overfitting.
Training plots (accuracy and loss) are saved as celebrity_accuracy_plot.png and celebrity_sample_loss_plot.png respectively.

## Evaluation
The trained model is evaluated on a test set, and classification metrics are printed, including accuracy and a detailed classification report.

## Prediction
The model is tested on sample images for each celebrity to demonstrate its prediction capabilities.
