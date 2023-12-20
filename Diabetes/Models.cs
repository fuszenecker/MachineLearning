using Microsoft.ML.Data;

public class InputData
{
    [ColumnName("preg")]
    public float Preg { get; set; }
    
    [ColumnName("plas")]
    public float Plas { get; set; }

    [ColumnName("pres")]
    public float Pres { get; set; }

    [ColumnName("skin")]
    public float Skin { get; set; }

    [ColumnName("insu")]
    public float Insu { get; set; }

    [ColumnName("mass")]
    public float Mass { get; set; }

    [ColumnName("pedi")]
    public float Pedi { get; set; }

    [ColumnName("age")]
    public float Age { get; set; }

    [ColumnName("class")]
    public string Class { get; set; }
}

public class OutputData
{
    [ColumnName("PredictedLabel")]
    public bool PredictedLabel { get; set; }
}