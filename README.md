# image_processing_with_cnn
I have created Machine learning project Image Processing with CNN

**Step 1: Download CIFAR-10 Dataset**
The CIFAR-10 dataset is automatically downloaded when running the training scripts. Ensure that you have a stable internet connection.

**Step 2: Data Preprocessing**
The preprocessing script normalizes the image pixel values to the range [0, 1].

**Run the preprocessing script:**

1)bash

2)Copy

3)python data/preprocessing.py

**Step 3: Build and Train the Models**

**Artificial Neural Network (ANN):**

**Model Architecture:**

Flatten layer
Dense layers with ReLU activation
Output layer with softmax activation

-->To build and train the ANN model:

1)bash

2)Copy

3)python models/train_model.py --model ann

**Convolutional Neural Network (CNN):**

**Model Architecture:**

Two convolutional layers followed by max-pooling layers
Flatten layer
Dense layers with ReLU and softmax activations


-->To build and train the CNN model:
1)bash

2)Copy

3)python models/train_model.py --model cnn

**Step 4: Evaluate the Models**

After training, you can evaluate the model on the test dataset.

1)bash

2)Copy

3)python models/evaluate_model.py --model ann

4)python models/evaluate_model.py --model cnn

This will output the classification report and accuracy of the models.

**Step 5: Visualizations**

You can visualize the training process and evaluate the performance of the models using visualizations.py. This includes plotting accuracy and loss curves, as well as confusion matrices.

1)bash

2)Copy

3)python utils/visualizations.py

**Step 6: Jupyter Notebook for Exploration**

You can also explore the models and experiment interactively using the provided Jupyter notebook.
1)bash

2)Copy

3)jupyter notebook notebooks/image_classification.ipynb
