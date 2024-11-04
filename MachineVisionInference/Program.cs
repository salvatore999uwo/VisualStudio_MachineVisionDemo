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
        //setup the input and output data structure (metadata is for setting the image size, matches what the model needs to be fed)
        public class ImageInput
        {
            [ImageType(64,64)]
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

            //path to the model file on my machine
            var modelPath = "C:\\Users\\salva\\source\\repos\\SAVED_MODELS";

            //load TensorFlow model using the MLContext we created earlier 
            var tensorflowModel = mlContext.Model.LoadTensorFlowModel(modelPath);

            //define preprocessing pipeline, from loading to preprocessing to turning our bitmaps into a tensor
            var pipeline = mlContext.Transforms.LoadImages(outputColumnName: "input", imageFolder: "", inputColumnName: nameof(ImageInput.Image))
                .Append(mlContext.Transforms.ResizeImages(outputColumnName: "input", imageWidth: 64, imageHeight: 64))
                .Append(mlContext.Transforms.ExtractPixels(outputColumnName: "input")) //this part converts the bitmap to a tensor for the model to understand
                .Append(mlContext.Model.LoadTensorFlowModel(modelPath)
                    .ScoreTensorFlowModel(
                        outputColumnNames: new[] { "dense_1" },  //this should be the actual output tensor name
                        inputColumnNames: new[] { "serving_default_conv2d_input" },    //this should be the actual input tensor name
                        addBatchDimensionInput: true))
                .Append(mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel"));



            //fit the pipeline to create an ITransformer
            var emptyDataView = mlContext.Data.LoadFromEnumerable(new List<ImageInput>()); //make an empty list of our ImageInput type, make an emptyDataView from it 
            var model = pipeline.Fit(emptyDataView); //fit the pipeline and get the transformer 

            //create the PredictionEngine using the fitted transformer (model)
            var predictionEngine = mlContext.Model.CreatePredictionEngine<ImageInput, ImagePrediction>(model);
            //a predictionEngine is a wrapper around the model used to classify one image at a time 
            //it takes in that list of our types ImageInput, ImagePrediction 
            //ImageInput (a 64x64 bitmap) is the input to the predictionEngine, and an ImagePrediction (a true/false value for broken/not broken) is the output



            //load an image and make a prediction
            var image = new Bitmap("C:\\Users\\salva\\source\\repos\\MachineVisionInference\\0049.jpg");
            var imageData = new ImageInput { Image = image };
            var prediction = predictionEngine.Predict(imageData);

            Console.WriteLine($"Prediction: {(prediction.PredictedLabel ? "Broken" : "Not Broken")}");




            //display the image and overlay prediction information
            string predictionText = prediction.PredictedLabel ? "Broken" : "Not Broken";
            string actualLabel = "actual label"; // Replace with the actual label if available

            //write on the image with prediction and actual label
            using (Graphics g = Graphics.FromImage(image))
            {
                Font font = new Font("Arial", 16);
                SolidBrush brush = new SolidBrush(Color.Red);

                g.DrawString($"Prediction: {predictionText}", font, brush, new PointF(10, 10));
                g.DrawString($"Actual: {actualLabel}", font, brush, new PointF(10, 40));
            }

            //display the annotated image
            DisplayImage(image);

        }


        //quick method that shows the image in a windows form 
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
