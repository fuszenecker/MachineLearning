using Microsoft.ML.Data;

public class ImageData
{
    [LoadColumn(0)]
    public string ImagePath { get; set; } = "";

    [LoadColumn(1)]
    public string Label { get; set; } = "";
}

public class AgePrediction
{
    [ColumnName("loss3/loss3_Y")]
    public float[] PredictedLabels { get; set; } = new float[8];
}