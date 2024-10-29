using System;
using System.Collections.Generic;
using System.IO;
using Emgu.CV;
using Emgu.CV.CvEnum;
using Emgu.CV.Structure;
using Tensorflow;
using Tensorflow.NumPy;
using static Tensorflow.Binding;

namespace MachineVisionDemo
{
    internal class Program
    {
        static void Main(string[] args)
        {
            //where i'll place code that calls in various methods from the classes i create. The code here will assume an already-trained model. 

            //first, declare strings with model and images file paths
            //then load the model
            //make directory of image files 
            
            //for each image,
                //preprocess the image (load, resize, normalize, put into proper tensor format)
                //send to model for classification 
                //save result of classification 
                //populate a csv of classifications 


           


        }
    }
}
