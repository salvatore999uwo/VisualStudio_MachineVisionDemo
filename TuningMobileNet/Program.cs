using System;
using System.Collections.Generic;
using System.IO;
using Emgu.CV;
using Emgu.CV.CvEnum;
using Tensorflow;
//using Tensorflow.NumPy;
using static Tensorflow.Binding;

using Keras;
using Keras.Layers;
using Keras.Models;
using Keras.Optimizers;

using NumSharp;
using Emgu.CV.Structure;
using Tensorflow.Keras.Engine;
using Tensorflow.Operations.Initializers;
using Emgu.CV.Features2D;
using System.Linq;
using Python.Runtime;


namespace PowerlineInsulatorClassifier
{
    class Program
    {
        // Define constants for image dimensions and path to save the model
        private const int ImageHeight = 224;
        private const int ImageWidth = 224;
        private static string ModelSavePath = "C:\\Users\\salva\\source\\repos\\TuningMobileNet\\Saved_Model";

        static void Main(string[] args)
        {
            // Load and preprocess images
            string dataDir = "C:\\Users\\salva\\source\\repos\\TuningMobileNet\\imageData"; // Set your directory path for images 
            var images = LoadAndPreprocessImages(dataDir);

            // Load and prepare the MobileNetV2 model for fine-tuning
            var model = LoadMobileNetV2Model();

            // Train the model
            TrainModel(model, images);

            // Save the trained model
            model.SaveOnnx(ModelSavePath);
        }

        static List<(NumSharp.NDArray, int)> LoadAndPreprocessImages(string dataDir)
        {
            var images = new List<(NumSharp.NDArray, int)>();

            foreach (var file in Directory.GetFiles(dataDir))
            {
                // Load image using Emgu CV
                Mat img = CvInvoke.Imread(file, ImreadModes.Color);

                // Resize to MobileNetV2's expected input size
                CvInvoke.Resize(img, img, new System.Drawing.Size(ImageWidth, ImageHeight));

                // Convert image to NDArray for TensorFlow
                var imgData = new NumSharp.NDArray(img.ToImage<Bgr, byte>().Data);

                // Determine label based on filename or directory structure
                int label = file.Contains("broken") ? 1 : 0; // Adjust based on your file naming

                images.Add((imgData, label));
            }

            return images;
        }

        static Sequential LoadMobileNetV2Model()
        {
            var model = new Sequential();
            model.Add(new Conv2D(32, new Tuple<int, int>(3, 3), strides: new Tuple<int, int>(2, 2), padding: "same", activation: "relu", input_shape: new Keras.Shape(224, 224, 3)));
            model.Add(new BatchNormalization());

            // Depthwise separable convolutions to approximate MobileNetV2 layers
            model.Add(new DepthwiseConv2D(new Tuple<int, int>(3, 3), padding: "same", strides: new Tuple<int, int>(1, 1)));
            model.Add(new BatchNormalization());
            model.Add(new Conv2D(64, new Tuple<int, int>(1, 1), activation: "relu"));
            model.Add(new BatchNormalization());

            // Additional Depthwise Convolution and Normalization layers
            model.Add(new DepthwiseConv2D(new Tuple<int, int>(3, 3), padding: "same", strides: new Tuple<int, int>(2, 2)));
            model.Add(new BatchNormalization());
            model.Add(new Conv2D(128, new Tuple<int, int>(1, 1), activation: "relu"));
            model.Add(new BatchNormalization());

            // Final pooling and classification layers
            model.Add(new GlobalAveragePooling2D());
            model.Add(new Dense(1, activation: "sigmoid"));

            // Compile the model
            model.Compile(optimizer: new Adam(0.001f), loss: "binary_crossentropy", metrics: new[] { "accuracy" });

            return model;
        }



        static void TrainModel(Sequential model, List<(NumSharp.NDArray, int)> images)
        {
  
            var xTrainData = images.Select(img => img.Item1.ToArray<float>()).ToArray();
            var xTrainNumSharp = NumSharp.np.array(xTrainData).reshape(images.Count, 224, 224, 3);


            var yTrainData = images.Select(img => (float)img.Item2).ToArray();
            var yTrainNumSharp = NumSharp.np.array(yTrainData).reshape(images.Count, 1);

      
            model.Fit(xTrainData, yTrainData, epochs: 10, batch_size: 32, validation_split: 0.2f);
        }

    }

}


