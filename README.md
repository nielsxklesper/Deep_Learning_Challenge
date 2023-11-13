# Deep Learning Challenge

## Task description
The assignment involves a popular topic in computer vision, the prediction of pneumonia and COVID-19.

Image classification is one of the most studied tasks in computer vision. The milestone paper, AlexNet, proposed a CNN architecture with ReLU activation function and dropout layers to achieve accurate image classification results on ImageNet classification challenge. For the assignment, you will be using the COVID-19 X-ray dataset.
After downloading and preparing the datasets, your assignment is to:
- Create a virtual environment and install tensorflow, matplotlib, pandas, keras, seaborn libraries. These are the libraries we recommend but you can also install the ones of your choice.
- Create train, validation and test sets using stratified train-test splits with ratios of 0.2 and random state of 42.
- Normalize your image data to floating point numbers between 0 and 1 and convert your target values using the proper function for one hot encoding (consider the second practical tutorial of the course).
- Implement the baseline CNN algorithm (exactly, without any modification) that is shown in Fig. 1. It is a network consisting of: two consecutive convolutional layers with 64 and 32 filters of size 3 × 3 with ReLU activations followed by a max pooling layer of size 2 × 2 and this block of convolutional layer followed by a max pooling layer is repeated two times. Finally, two dense layers of sizes 32 with Relu activation function and an output layer of size 4 with the proper activation function (you are expected to find out) is added. The number of epochs should be set as 10 and batch size as 32. The optimizer is Adam, the metric should be accuracy and the loss function is expected from you :-).
- Analyze the performance of the baseline by plotting: (i) the training and validation losses and accuracies on the training and validation set (similar to Fig. 2a and Fig. 2b), (ii) the Receiver Operator Characteristic (ROC) curve with the Area under the Curve (AUC) score and a confusion matrix for the validation and test set. Examples of accuracy and loss plots are shown in Fig. 2, an example of a ROC curve and confusion matrix is shown in Fig. 3, respectively. Report performance measures (accuracy, sensitivity, specificity and F1-score).
- Once you have a baseline model, adapt/fine-tune the network to improve its performance by: (i) changing the hyper-parameters (e.g. add more layers) and/or (ii) apply- ing data augmentation techniques. Illustrate the improvements of your new network over the baseline by: (a) plotting the ROC curve with AUC score and (b) reporting performance measures. Compare and explain the differences between the two models as well as potential reasons behind the increase in performance.

## Dataset
The dataset is available online and can be downloaded from:
https://darwin.v7labs.com/v7-labs/covid-19-chest-x-ray-dataset?sort=priority%3Adesc.
The data entitled as ’darwin dataset pull v7-labs/covid-19-chest-x-ray-dataset:all-images’ will be used in this assignment.
The dataset contains 6500 images of AP/PA chest x-rays with pixel-level polygonal lung segmentations. There are 517 cases of COVID-19 amongst these. You are supposed to use images from 4 different categories (Bacterial Pneumonia, Viral Pneumonia, No Pneumonia (healthy), Covid-19).
Image resolutions, sources, and orientations vary across the dataset, with the largest image being 5600 × 4700 and smallest being 156 × 156. All images should be resized by 156 × 156 in image preprocessing step. You can use the provided code to load, resize and save images.
