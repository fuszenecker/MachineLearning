
using System.Reflection;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms.Image;

if (args.Length == 0)
{
    Console.WriteLine("Usage: ObjectDetectionOnnx <image>");
    return;
}

var mlContext = new MLContext();

var pwd = Path.GetDirectoryName(Assembly.GetExecutingAssembly().Location);

var pipeline =
    mlContext.Transforms.LoadImages(
        outputColumnName: "image",
        imageFolder: "",
        inputColumnName: "ImagePath"
        )
    .Append(mlContext.Transforms.ResizeImages(
        outputColumnName: "image",
        imageWidth: 224,
        imageHeight: 224,
        inputColumnName: "image"
        ))
    .Append(mlContext.Transforms.ExtractPixels(
        outputColumnName: "data",
        inputColumnName: "image",
        offsetImage: 117,
        orderOfExtraction: ImagePixelExtractingEstimator.ColorsOrder.ABRG,
        scaleImage: 1f / 255f
        ))
    .Append(mlContext.Transforms.ApplyOnnxModel(
        Path.Combine(pwd, "resnet152-v1-7.onnx")
    ));

var data = mlContext.Data.LoadFromEnumerable(
    new List<InputModel>()    
);

var trainedData = pipeline.Fit(data);

var predictionEngine = mlContext.Model.CreatePredictionEngine<InputModel, OutputModel>(trainedData);

var result = predictionEngine.Predict(new InputModel()
{
    ImagePath = args[0]
});

var items = File.ReadAllLines(Path.Combine(pwd, "synset.txt"));

var max = result.Head.OrderByDescending(x => x)
    .Take(10)
    .Select(r => result.Head.ToList().IndexOf(r))
    .Select(v => $"{items[v]} - {result.Head[v]}");

Console.WriteLine(String.Join("\n", max));

public class InputModel
{
    public string ImagePath { get; set; } = "";
}

public class OutputModel : InputModel
{
    [ColumnName("softmax_tensor")]
    public string PredictedLabel { get; set; } = "";

    [ColumnName("resnetv19_dense0_fwd")]
    public float[] Head { get; set; } = new float[1000];
}
