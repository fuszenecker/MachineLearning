# Multiclass classification

## Training data

```csv
Gyümölcs,Szín,Illat,Íz
alma,piros,édes,savanykás
körte,sárga,gyümölcsös,édes
banán,sárga,nincs,édes
szilva,lila,gyümölcsös,édes
```

## Models

```csharp
public class InputModel
{
    [LoadColumn(0)]
    [ColumnName("Name")]
    public string Name { get; set; }

    [LoadColumn(1)]
    [ColumnName("Colour")]
    public string Colour { get; set; }

    [LoadColumn(2)]
    [ColumnName("Smell")]
    public string Smell { get; set; }

    [LoadColumn(3)]
    [ColumnName("Flavour")]
    public string Flavour { get; set; }
}

public class OutputModel : InputModel
{
    [ColumnName("PredictedLabelValue")]
    public string PredictedLabel { get; set; }

    [ColumnName("Score")]
    public float[] Score { get; set; }
}
```

## Code

```csharp
using Microsoft.ML;
using Microsoft.ML.Data;

var mlContext = new MLContext();

var data = mlContext.Data.CreateTextLoader<InputModel>(separatorChar: ',', hasHeader: true).Load("data.csv");

var datatransform =
    // The Label needs to be a limited set of values.
    mlContext.Transforms.Conversion.MapValueToKey(
        outputColumnName: "NameKey",
        inputColumnName: "Name")

    // The features need to be encoded as numbers (floats/singles).
    .Append(mlContext.Transforms.Categorical.OneHotEncoding(
        outputColumnName: "ColourEncoded",
        inputColumnName: "Colour"
    ))

    .Append(mlContext.Transforms.Categorical.OneHotEncoding(
        outputColumnName: "SmellEncoded",
        inputColumnName: "Smell"
    ))

    .Append(mlContext.Transforms.Categorical.OneHotEncoding(
        outputColumnName: "FlavourEncoded",
        inputColumnName: "Flavour"
    ))

    // The features need to be concatenated into a single column.
    .Append(mlContext.Transforms.Concatenate(
        outputColumnName: "Features",
        inputColumnNames: ["ColourEncoded", "SmellEncoded", "FlavourEncoded"]
    ));

// The trainer needs to be a multiclass trainer.
// Please specify the label column name and the feature column name.
var trainer = mlContext.MulticlassClassification.Trainers.SdcaNonCalibrated(
    labelColumnName: "NameKey",
    featureColumnName: "Features"
);

// The predicted label needs to be mapped back to the original label (text).
var postprocessingPipeline = mlContext.Transforms.Conversion.MapKeyToValue(
    outputColumnName: "PredictedLabelValue",
    inputColumnName: "PredictedLabel"
);

// The pipeline needs to be trained on the data.
var trainingPipeline = datatransform
    .Append(trainer)
    .Append(postprocessingPipeline);

var model = trainingPipeline.Fit(data);

// Let's save the model to a file.
mlContext.Model.Save(model, data.Schema, "model.zip");

// Let's load the model from a file.
var loadedModel = mlContext.Model.Load("model.zip", out var schema);

// The prediction engine needs to be created from the loaded model.
var predictionEngine = mlContext.Model.CreatePredictionEngine<InputModel, OutputModel>(loadedModel);

// Let's make a prediction.
var prediction = predictionEngine.Predict(new InputModel
{
    Colour = "sárga",
    Smell = "gyümölcsös",
    Flavour = "édes"
});

Console.WriteLine($"Prediction: {prediction.PredictedLabel}");
```

Output:

```
Prediction: körte
```
