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