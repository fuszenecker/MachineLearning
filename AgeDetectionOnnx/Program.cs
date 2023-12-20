using Microsoft.ML;
using Microsoft.ML.Transforms.Image;

var mlContext = new MLContext();

var data = mlContext.Data.LoadFromEnumerable(new List<ImageData>());

var pipeline = 
    mlContext.Transforms.LoadImages(outputColumnName: "image",
        imageFolder: "Images",
        inputColumnName: nameof(ImageData.ImagePath))

    .Append(
        mlContext.Transforms.ResizeImages(
            outputColumnName: "image",
            imageWidth: 224,
            imageHeight: 224,
            inputColumnName: "image",
            resizing: ImageResizingEstimator.ResizingKind.Fill
        )
    )

    .Append(
        mlContext.Transforms.ExtractPixels(
            outputColumnName: "input",
            inputColumnName: "image"
        )
    )

    .Append(
        mlContext.Transforms.ApplyOnnxModel(
            modelFile: "age_googlenet.onnx"
        )
    );

var model = pipeline.Fit(data);

var engine = mlContext.Model.CreatePredictionEngine<ImageData, AgePrediction>(model);
var result = engine.Predict(new ImageData { ImagePath = "RóbertFuszenecker.jpg" });

var ageList = new string[] { "0-2", "4-6", "8-12", "15-20", "25-32", "38-43", "48-53", "60-" }; 

for (int i = 0; i < result.PredictedLabels.Length; i++)
{
    Console.WriteLine($"{ageList[i]}:\t{result.PredictedLabels[i],10:P}");
}
