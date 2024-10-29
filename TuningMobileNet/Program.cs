using System;
using System.IO;
using System.Collections.Generic;
using Emgu.CV;
using Emgu.CV.CvEnum;
using Tensorflow;
using Tensorflow.Keras;
using Tensorflow.Keras.Layers;
using Tensorflow.Keras.Models;
using Tensorflow.Keras.Optimizers;
using NumSharp;


namespace TuningMobileNet
{
    class Program
    {
        //define some parameters
        private static int ImageHeight = 224;
        private static int ImageWidth = 224;
        //the model only takes in 224x224 images
        //private of course, static because the "program" class 
        //is not going to be instantiated. it's the program class
        static void Main(string[] args)
        {

            //step 1: load, preprocess images
            string dataDir = "C:\\Users\\salva\\OneDrive\\Desktop\\InsulatorDataSet-master";
            var images = LoadAndPreprocessImages(dataDir);

            //step 2: load model and customize final layer 
            var model = LoadMobileNetV2Model();

            //step 3: train the model
            TrainModel(model, images);

            //step 4: save the newly trained model 
            model.Save("C:\\Users\\salva\\source\\repos\\TuningMobileNet");
            Console.WriteLine("model trained and saved!");
        }

        static List<(NDArray, int)> LoadAndPreprocessImages(string dataDir)
        {
            var images = new List<(NDArray, int)>();
            //the list<(NDArray, int)> syntax means that every item in this array is
            //a tuple that contains both an n-dimensional array and an int
            //so you can visualize it as, each i of the index contains a single 
            //n-dimensional array and a single int at that spot of the array index 
            //and of course this is not an array, but a list - it can change size 

            foreach (var file in Directory.GetFiles(dataDir))
            {
                //load images using emgu cv 
                Mat img = CvInvoke.Imread(file, ImreadModes.Color);

                //resize to the model-required size 
                CvInvoke.Resize(img, img, new System.Drawing.Size(ImageWidth, ImageHeight));

                //convert to NDArray for TensorFlow.NET
                var imgData = np.array(img.ToImage<Bgr, byte>().Data);

                //assign label based on filename or directory stucture
                int label = file.Contains("broken") ? 1 : 0;

                images.Add((imgData, label));
            }

            return images;
        }

        static keras.Sequential LoadMobileNetV2Model()
        {
            //load MobileNetV2 with pre-trained weights
            var model = keras.applications.MobileNetV2(weights: "imagenet",
                include_top: false, input_shape: (ImageHeight, ImageWidth, 3));

            //add custom classification layers
            model.add(new GlobalAveragePooling2D());
            model.add(new Dense(1, ActivationContext: "sigmoid"));

            model.compile(optimizer: new Adam(learning_rate 0.001), loss: keras.losses.BinaryCrossentropy(),
                metrics: new[] { "accuracy" });

            return model;
        }

        static void TrainModel(Tensorflow.Keras.Engine.Sequential model, List<(NDArray, int)> images)
        {
            //convert images to suitable tensors for model input
            var xTrain = np.array(images.ConvertAll(img => img.Item1));

            var yTrain = np.array(images.ConvertAll(img => img.Item2));

            //train the model
            model.fit(xTrain, yTrain, epochs: 10, batch_size: 32, validation_split: 0.2f);
        }
    }
}



