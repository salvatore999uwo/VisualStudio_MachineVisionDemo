# VisualStudio_MachineVisionDemo
Simple project to solve a binary image classification problem using a CNN (convolutional neural net) that classifies power line insulators as either "broken" or "not broken". Uses Emgu CV and Tensorflow with Keras, all in Visual Studio using C#.

Below is a normal power line insulator, somewhere in China: 
![Normal Insulator](https://github.com/salvatore999uwo/VisualStudio_MachineVisionDemo/blob/main/sample%20photos/0049.jpg)


And below is an artificially generated "abnormal" insulator: 
![Abnormal Insulator](https://github.com/salvatore999uwo/VisualStudio_MachineVisionDemo/blob/main/sample%20photos/026.jpg)

The goal of this project is to train and implement a lightweight neural net that can classify images accurately as a "normal" versus "abnormal" insulator. This version of the program will clearly not be useful, as real 
power line insulators do not look like the artificially generated image above when they have failed or been installed incorrectly. However, the techniques and model used here could easily be transferred over to a real 
defect-detection application, such as detecting an "abnormal" part during quality control imaging at a manufacturing plant. 
