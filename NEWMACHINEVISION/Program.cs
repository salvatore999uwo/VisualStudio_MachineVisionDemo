using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms.Image;

using System.Drawing;


namespace NEWMACHINEVISION
{

    public class Program
    {
        // Input and Output data structure
        public class ImageInput
        {
            [ImageType(224, 224)]
            public Bitmap Image { get; set; }
        }

        public class ImagePrediction
        {
            [ColumnName("PredictedLabel")]
            public bool PredictedLabel { get; set; }
        }

        public static void Main()
        {
            var mlContext = new MLContext();

            // Path to the model file
            var modelPath = "path/to/your_model.pb";

            // Load TensorFlow model
            var tensorflowModel = mlContext.Model.LoadTensorFlowModel(modelPath);

            // Define preprocessing pipeline
            var pipeline = mlContext.Transforms.LoadImages(outputColumnName: "input", imageFolder: "", inputColumnName: nameof(ImageInput.Image))
                .Append(mlContext.Transforms.ResizeImages(outputColumnName: "input", imageWidth: 224, imageHeight: 224))
                .Append(mlContext.Transforms.ExtractPixels(outputColumnName: "input"))
                .Append(tensorflowModel.ScoreTensorName("output"))
                .Append(mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel"));

            // Create prediction engine
            var predictionEngine = mlContext.Model.CreatePredictionEngine<ImageInput, ImagePrediction>(pipeline);

            // Load an image and make a prediction
            var image = new Bitmap("path/to/real_image.jpg");
            var imageData = new ImageInput { Image = image };
            var prediction = predictionEngine.Predict(imageData);

            Console.WriteLine($"Prediction: {(prediction.PredictedLabel ? "Broken" : "Not Broken")}");
        }
    }













}
