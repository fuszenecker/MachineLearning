using Microsoft.ML;

MLContext mlContext = new MLContext(seed: 1);

mlContext.Log += (s, e) =>  {
    if (e.Kind != Microsoft.ML.Runtime.ChannelMessageKind.Trace) {
        Console.WriteLine($"{e.Kind}: {e.Message}");    
    }
};

IEnumerable<ImageData> images = Directory.GetFiles(
            "TrainingImages", "*",
            searchOption: SearchOption.AllDirectories
        )
        .Select(file => new ImageData()
        {
            ImagePath = file,
            Label = Directory.GetParent(file).Name
        });

Console.WriteLine($"Images: {images.Count()}");

IDataView imageData = mlContext.Data.LoadFromEnumerable(images);

var preprocessingPipeline = mlContext.Transforms.Conversion.MapValueToKey(
        inputColumnName: "Label",
        outputColumnName: "LabelAsKey")

    .Append(mlContext.Transforms.LoadRawImageBytes(
        outputColumnName: "Image",
        imageFolder: "",
        inputColumnName: "ImagePath"));

var trainingPipeline = mlContext.MulticlassClassification.Trainers.ImageClassification(
        featureColumnName: "Image",
        labelColumnName: "LabelAsKey");

var postProcessingPipeline = mlContext.Transforms.Conversion.MapKeyToValue(
        outputColumnName: "PredictedLabelValue",
        inputColumnName: "PredictedLabel");

var pipeline = preprocessingPipeline
    .Append(trainingPipeline)
    .Append(postProcessingPipeline);

ITransformer trainedModel = pipeline.Fit(imageData);
mlContext.Model.Save(trainedModel, imageData.Schema, "model.zip");

var loadedModel = mlContext.Model.Load("model.zip", out var modelInputSchema);
var predictionEngine = mlContext.Model.CreatePredictionEngine<ImageData, ModelOutput>(loadedModel);

var result = predictionEngine.Predict(new ImageData() { 
    ImagePath = args.Any() ? args[0] : "TrainingImages/roses/10503217854_e66a804309.jpg" 
});

Console.WriteLine($"Image: {result.ImagePath}");
Console.WriteLine($"Predicted: {result.PredictedLabel}");
Console.WriteLine($"Probability: {result.Score.Max():P2}");
