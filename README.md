# Image Classification using Tensorflow (Python 3.6)

Project written in python 3.6 to do data augmentation, create data train and test sets, train and persist model using tensorflow, test the persisted model to do image classification predictions.

Please note that you would require Python 3.6 along with specific packages which can installed using command. <br>
`python -m pip install tensorflow opencv-python opencv-contrib-python ipython matplotlib pillow h5py SciPy`
<br>
## Data Augmentation

Inside folder 'master', you can place your images into respective sub-folder. Note that the sub-folder name forms the classification label. In this example, we have under 'master', 3 folders for fruit 'apple', 'orange' and 'strawberry'; and the respective fruit images inside each folder. You can replace these folder with your own folders and images.<br>
Once you have put together the 'master' folder, run python script using command<br>
<br>
`python data-augmentation.py`
<br>
<br>
This command will result in creating 'train' and 'test' dataset folders under 'data' folder. <<br>
The script will take each individual images from 'master' folder and create multiple copies of same image (train data) with rotation of 1 degree increments (from 0 to 360). A small set of these rotated image is used as 'test' data for model validation.
<br>

## Train Model and Save

Run below command to train the model using the data generated as described in 'Data Augmentation' section. The trained model will be stored in .h5 format
<br><br>
`python create-train-model.py`
<br><br>
You can also use Jupyter Notebook and open file **create-train-model.ipynb** to study and run sections of code and understand the logic.
<br>

## Test Model

Run below command to test the model using the test data generated as described in 'Data Augmentation' section.
<br><br>
`python test-model.py`
<br><br>
You can also use Jupyter Notebook and open file **test-model.ipynb** to study and run sections of code and understand the logic.
<br>
