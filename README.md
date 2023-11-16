# Deep Learning Challenge

## Task description
The assignment involves a popular topic in computer vision, the prediction of pneumonia and COVID-19.

Image classification is one of the most studied tasks in computer vision. The milestone paper, [AlexNet](https://proceedings.neurips.cc/paper_files/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf), proposed a CNN architecture with ReLU activation function and dropout layers to achieve accurate image classification results on ImageNet classification challenge. For the assignment, you will be using the COVID-19 X-ray dataset.
After downloading and preparing the datasets, your assignment is to:
- Create a virtual environment and install tensorflow, matplotlib, pandas, keras, seaborn libraries. These are the libraries we recommend but you can also install the ones of your choice.
- Create train, validation and test sets using stratified train-test splits with ratios of 0.2 and random state of 42.
- Normalize your image data to floating point numbers between 0 and 1 and convert your target values using the proper function for one hot encoding (consider the second practical tutorial of the course).
- Implement the baseline CNN algorithm (exactly, without any modification), which can be found below. It is a network consisting of: two consecutive convolutional layers with 64 and 32 filters of size 3 × 3 with ReLU activations followed by a max pooling layer of size 2 × 2 and this block of convolutional layer followed by a max pooling layer is repeated two times. Finally, two dense layers of sizes 32 with Relu activation function and an output layer of size 4 with the proper activation function (you are expected to find out) is added. The number of epochs should be set as 10 and batch size as 32. The optimizer is Adam, the metric should be accuracy and the loss function is expected from you :-).
- Analyze the performance of the baseline by plotting: (i) the training and validation losses and accuracies on the training and validation set, (ii) the Receiver Operator Characteristic (ROC) curve with the Area under the Curve (AUC) score and a confusion matrix for the validation and test set. Report performance measures (accuracy, sensitivity, specificity and F1-score).
- Once you have a baseline model, adapt/fine-tune the network to improve its performance by: (i) changing the hyper-parameters (e.g. add more layers) and/or (ii) apply- ing data augmentation techniques. Illustrate the improvements of your new network over the baseline by: (a) plotting the ROC curve with AUC score and (b) reporting performance measures. Compare and explain the differences between the two models as well as potential reasons behind the increase in performance.

<p align="center"> <img width="725" src="https://github.com/nielsxklesper/Deep_Learning_Challenge/assets/150530277/f0a707d5-faf5-4a9f-b4b8-ace3532f45fa"> </p>

## Dataset
The dataset is available online and can be downloaded from:
https://darwin.v7labs.com/v7-labs/covid-19-chest-x-ray-dataset?sort=priority%3Adesc.
The data entitled as ’darwin dataset pull v7-labs/covid-19-chest-x-ray-dataset:all-images’ will be used in this assignment.
The dataset contains 6500 images of AP/PA chest x-rays with pixel-level polygonal lung segmentations. There are 517 cases of COVID-19 amongst these. You are supposed to use images from 4 different categories (Bacterial Pneumonia, Viral Pneumonia, No Pneumonia (healthy), Covid-19).
Image resolutions, sources, and orientations vary across the dataset, with the largest image being 5600 × 4700 and smallest being 156 × 156. All images should be resized by 156 × 156 in image preprocessing step. You can use the provided code to load, resize and save images.

## Results 

The **baseline CNN** demonstrates reasonable performance, achieving an accuracy of approximately 70% on the validation set. However, it is important to note that due to the imbalance in the dataset, relying solely on the model's accuracy may not provide a comprehensive evaluation. Therefore, it is recommended to consider more robust evaluation metrics, such as the F1-score which shows a weaker performance of the model, and is presented alongside other metrics in the table below:
<div align="center">
  
| Evaluation Metric  | Value        |
|--------------------|--------------|
| **F1-Score**       | 0.6059       |
| **Precision**      | 0.6163       |
| **Recall**         | 0.6003       |

</div>

Based on the other evaluation metrics the performance of the baseline model doesn't appear to be that reasonable anymore. But what is the main issue of the model?! The answer might lie in the next two charts that visualize the development of the accuracy and loss of the training and validation dataset.

![image](https://github.com/nielsxklesper/Deep_Learning_Challenge/assets/150530277/be6846fb-2810-4802-aed1-bd757543aa29)

The accuracy/loss plot illustrates improvements in both training and validation accuracy, with decreasing losses in the initial epochs. However, after the 6th epoch, a slight increase in validation loss indicates the primary issue with the model, suggesting **overfitting**.

The **improved CNN** effectively tackles the limitations observed in the baseline model through the strategic implementation of measures such as **early stopping**, **data augmentation**, **dropout layers**, and **advanced architecture**. The integration of these measures proves to be advantageous when assessing performance metrics. Notably, the improved CNN attained an accuracy of approximately 74% on the validation set. More significantly, there were substantial improvements in F1-score, precision, and recall values compared to the baseline model.

<div align="center">
  
| Evaluation Metric  | Value        |
|--------------------|--------------|
| **F1-Score**       | 0.704374     |
| **Precision**      | 0.697562	    |
| **Recall**         | 0.772714     |

</div>

The improved model performance is further reflected in the accuracy/loss plot below:

![image](https://github.com/nielsxklesper/Deep_Learning_Challenge/assets/150530277/73a1659c-9c8f-4946-b0ba-41f1c5344087)

**Note!** To avoid overloading this readme page a more technical model evaluation can be found in the accompanied Jupyter notebook. Further, I would like to highlight that I am by no means an expert in the field of deep learning! I am continuously learning and refining my knowledge and skills in this field. Even with improvements, the enhanced model still doesn't quite meet the level of performance needed for practical use in daily applications. Consequently, future experiments will include exploring different model architectures, such as transfer learning.
