using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms.Image;

using System.Drawing;
//using static System.Net.Mime.MediaTypeNames;
using System.Windows.Forms;


namespace MachineVisionInference
{

    public class Program
    {
        // Input and Output data structure
        public class ImageInput
        {
            [ImageType(224, 224)]
            public Bitmap? Image { get; set; }
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
            var modelPath = "C:\\Users\\salva\\source\\repos\\SAVED_MODEL";

            // Load TensorFlow model
            var tensorflowModel = mlContext.Model.LoadTensorFlowModel(modelPath);

            // Define preprocessing pipeline
            var pipeline = mlContext.Transforms.LoadImages(outputColumnName: "input", imageFolder: "", inputColumnName: nameof(ImageInput.Image))
                .Append(mlContext.Transforms.ResizeImages(outputColumnName: "input", imageWidth: 224, imageHeight: 224))
                .Append(mlContext.Transforms.ExtractPixels(outputColumnName: "input"))
                .Append(mlContext.Model.LoadTensorFlowModel(modelPath)
                    .ScoreTensorFlowModel(
                        outputColumnNames: new[] { "output" },  // Replace "output" with the actual output tensor name of your model
                        inputColumnNames: new[] { "input" },    // Replace "input" with the actual input tensor name of your model
                        addBatchDimensionInput: true))
                .Append(mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel"));



            // Fit the pipeline to create an ITransformer
            var emptyDataView = mlContext.Data.LoadFromEnumerable(new List<ImageInput>());
            var model = pipeline.Fit(emptyDataView); // Fit the pipeline and get the transformer

            // Create the PredictionEngine using the fitted transformer (model)
            var predictionEngine = mlContext.Model.CreatePredictionEngine<ImageInput, ImagePrediction>(model);



            // Load an image and make a prediction
            var image = new Bitmap("path/to/real_image.jpg");
            var imageData = new ImageInput { Image = image };
            var prediction = predictionEngine.Predict(imageData);

            Console.WriteLine($"Prediction: {(prediction.PredictedLabel ? "Broken" : "Not Broken")}");




            // Display the image and overlay prediction information
            string predictionText = prediction.PredictedLabel ? "Broken" : "Not Broken";
            string actualLabel = "Actual Label Here"; // Replace with the actual label if available

            // write on the image with prediction and actual label
            using (Graphics g = Graphics.FromImage(image))
            {
                Font font = new Font("Arial", 16);
                SolidBrush brush = new SolidBrush(Color.Red);

                g.DrawString($"Prediction: {predictionText}", font, brush, new PointF(10, 10));
                g.DrawString($"Actual: {actualLabel}", font, brush, new PointF(10, 40));
            }

            // Display the annotated image
            DisplayImage(image);

        }


        // quick method that shows the image in a windows form 
        private static void DisplayImage(Bitmap image)
        {
            Form form = new Form
            {
                Text = "Inference Result",
                ClientSize = new Size(image.Width, image.Height)
            };

            PictureBox pictureBox = new PictureBox
            {
                Dock = DockStyle.Fill,
                Image = image,
                SizeMode = PictureBoxSizeMode.StretchImage
            };

            form.Controls.Add(pictureBox);
            Application.Run(form);

        }



    }
}
