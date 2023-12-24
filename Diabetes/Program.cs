using Microsoft.ML;
using Microsoft.Data.Analysis;

namespace Diabetes;

class Program
{
    static void Main(string[] args)
    {
        TrainTheModel();
        UseTheModel();
    }

    static void TrainTheModel()
    {
        var mlContext = new MLContext();

        var data = DataFrame.LoadCsv("data.csv", separator: ',', header: true);
        var split = mlContext.Data.TrainTestSplit(data, testFraction: 0.2);

        var pipeline = 

            mlContext.Transforms.Conversion.MapValue<string, bool>(
                outputColumnName: "Label",
                new Dictionary<string, bool> {
                    ["tested_positive"] = true,
                    ["tested_negative"] = false
                },
                inputColumnName: "class"
            )

            .Append(
                mlContext.Transforms.Concatenate(
                    outputColumnName: "Features",
                    inputColumnNames: ["preg", "plas", "pres", "skin", "insu", "mass", "pedi", "age"]
                )
            )

            // .Append(
            //     mlContext.Transforms.NormalizeMinMax(
            //         outputColumnName: "Features",
            //         inputColumnName: "Features"
            //     )
            // )

            .Append(
                mlContext.BinaryClassification.Trainers.LbfgsLogisticRegression(
                    labelColumnName: "Label",
                    featureColumnName: "Features"
                )
            )

            .AppendCacheCheckpoint(mlContext);

        var model = pipeline.Fit(split.TrainSet);

        var evaluationData = model.Transform(split.TestSet);
        var evaluationResult = mlContext.BinaryClassification.Evaluate(evaluationData);
    
        mlContext.Model.Save(model, ((IDataView)data).Schema, "model.zip");

        Console.WriteLine($"Accuracy: {evaluationResult.Accuracy}");
    }

    static void UseTheModel()
    {
        var mlContext = new MLContext();
        var model = mlContext.Model.Load("model.zip", out var schema);
        var predictionEngine = mlContext.Model.CreatePredictionEngine<InputData, OutputData>(model);

        var prediction1 = predictionEngine.Predict(new InputData
        {
            Preg = 4f,
            Plas = 110f,
            Pres = 92f,
            Skin = 0f,
            Insu = 0f,
            Mass = 37.6f,
            Pedi = 0.191f,
            Age = 30f
        });

        Console.WriteLine($"Prediction 1: {prediction1.PredictedLabel}");

        var prediction2 = predictionEngine.Predict(new InputData
        {
            Preg = 0f,
            Plas = 137f,
            Pres = 40f,
            Skin = 35f,
            Insu = 168f,
            Mass = 43.1f,
            Pedi = 2.288f,
            Age = 33f
        });

        Console.WriteLine($"Prediction 2: {prediction2.PredictedLabel}");
    }
}
