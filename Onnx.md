# ONNX age prediction

## ONNX tools

Install tool:

```
pip3 install onnxcli
```

Inspect ONNX model I/O:

```
onnx inspect -io ./resnet152-v1-7.onnx
```

## Models

```csharp
using Microsoft.ML.Data;

public class ImageNetData
{
    [LoadColumn(0)]
    public string ImagePath;

    [LoadColumn(1)]
    public string Label;
}

public class ImageNetPrediction
{
    [ColumnName("loss3/loss3_Y")]
    public float[] PredictedLabels;
}
```

## Code

```csharp
using Microsoft.ML;
using Microsoft.ML.Transforms.Image;

var mlContext = new MLContext();

var data = mlContext.Data.LoadFromEnumerable(new List<ImageNetData>());

var pipeline = mlContext.Transforms.LoadImages(outputColumnName: "image", imageFolder: "/home/fuszenecker/dev/onnx01/img", inputColumnName: nameof(ImageNetData.ImagePath))
    .Append(mlContext.Transforms.ResizeImages(
        outputColumnName: "image",
        imageWidth: 224,
        imageHeight: 224,
        inputColumnName: "image",
        resizing: ImageResizingEstimator.ResizingKind.Fill))
    .Append(mlContext.Transforms.ExtractPixels(outputColumnName: "input", inputColumnName: "image"))
    .Append(mlContext.Transforms.ApplyOnnxModel(modelFile: "age_googlenet.onnx"));

var model = pipeline.Fit(data);

var engine = mlContext.Model.CreatePredictionEngine<ImageNetData, ImageNetPrediction>(model);
var result = engine.Predict(new ImageNetData { ImagePath = "fr.jpg" });

Console.WriteLine(String.Join(", ", result.PredictedLabels.Select(x => x.ToString())));
```

## Other useful pipeline steps

```csharp
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
        interleavePixelColors: false,
        offsetImage: 117,
        orderOfExtraction: ImagePixelExtractingEstimator.ColorsOrder.ABRG,
/* ! */ scaleImage: 1f / 255f
    ))
    .Append(mlContext.Transforms.ApplyOnnxModel(
        "resnet152-v1-7.onnx"
    ));
```

