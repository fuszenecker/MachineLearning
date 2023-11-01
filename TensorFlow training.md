# TensorFlow training

## Dependencies

⚠️ TensorFlow must be exactly **2.3.1**.

```xml
<PackageReference Include="Microsoft.ML" Version="2.0.1" />
<PackageReference Include="Microsoft.ML.ImageAnalytics" Version="2.0.1" />
<PackageReference Include="Microsoft.ML.Vision" Version="2.0.1" />
<PackageReference Include="SciSharp.TensorFlow.Redist-Linux-GPU" Version="2.3.1" />
```

or `<PackageReference Include="SciSharp.TensorFlow.Redist" Version="2.3.1" />`.

## Models

```csharp
using Microsoft.ML.Data;

public class ImageData
{
    public string ImagePath { get; set; }

    public string Label { get; set; }
}

public class ModelOutput : ImageData
{
    [ColumnName("PredictedLabelValue")]
    public string PredictedLabel { get; set; }

    [ColumnName("Score")]
    public float[] Score { get; set; }
}
```

## Training code

```csharp
using Microsoft.ML;

MLContext mlContext = new MLContext(seed: 1);

IEnumerable<ImageData> images = Directory.GetFiles(
            "training_images", "*",
            searchOption: SearchOption.AllDirectories
        )
        .Select(file => new ImageData()
        {
            ImagePath = file,
            Label = Directory.GetParent(file).Name
        });

Console.WriteLine($"Images: {images.Count()}");

var preprocessingPipeline = mlContext.Transforms.Conversion.MapValueToKey(
        inputColumnName: "Label",
        outputColumnName: "LabelAsKey")
    // .Append(mlContext.Transforms.LoadImages(
    //     outputColumnName: "Image",
    //     imageFolder: "training_images",
    //     inputColumnName: "ImagePath"))
    // .Append(mlContext.Transforms.ResizeImages(
    //     outputColumnName: "Image",
    //     imageWidth: 320,
    //     imageHeight: 320,
    //     inputColumnName: "Image"))
    .Append(mlContext.Transforms.LoadRawImageBytes(
        outputColumnName: "Image",
        imageFolder: "",
        inputColumnName: "ImagePath"))
    ;

IDataView imageData = mlContext.Data.LoadFromEnumerable(images);

IDataView preprocessedData = preprocessingPipeline
                    .Fit(imageData)
                    .Transform(imageData);

var trainingPipeline = mlContext.MulticlassClassification.Trainers.ImageClassification(
        featureColumnName: "Image",
        labelColumnName: "LabelAsKey");

var postProcessingPipeline = mlContext.Transforms.Conversion.MapKeyToValue(
        outputColumnName: "PredictedLabelValue",
        inputColumnName: "PredictedLabel");

var pipeline = preprocessingPipeline
    .Append(trainingPipeline)
    .Append(postProcessingPipeline);

ITransformer trainedModel = pipeline.Fit(preprocessedData);
mlContext.Model.Save(trainedModel, imageData.Schema, "model.zip");
```

## Prediction code

```csharp
var loadedModel = mlContext.Model.Load("model.zip", out var modelInputSchema);

var predictionEngine = mlContext.Model.CreatePredictionEngine<ImageData, ModelOutput>(loadedModel);
var result = predictionEngine.Predict(new ImageData() { ImagePath = "training_images/roses/10503217854_e66a804309.jpg" });

Console.WriteLine($"Image: {result.ImagePath}");
Console.WriteLine($"Predicted: {result.PredictedLabel}");
Console.WriteLine($"Probability: {result.Score.Max():P2}");
```

## Data

From [TensorFlow examples](http://download.tensorflow.org/example_images/flower_photos.tgz).
