using Microsoft.ML;
using Microsoft.ML.Data;

string _trainingFilePath = "PATH FOR SOURCE FILE\\department_email_subjects_input.tsv";
string _modelFilePath = "PATH FOR OUTPUT MODEL FILE\\department_email_subjects_model.zip";

MLContext _mlContext = new MLContext(seed: 0);
IDataView _trainingDataView = _mlContext.Data.LoadFromTextFile<EmailSubject>(_trainingFilePath, hasHeader: true);
ITransformer _model;

var pipeline = ProcessData();

var trainingPipeline = BuildAndTrainModel(_trainingDataView, pipeline);
SaveModelAsFile();

var result = PredictDepartmentForSubjectLine("Salary not credited");
Console.WriteLine(result);

PredictionEngine<EmailSubject, DepartmentPrediction> _predictionEngine;

string PredictDepartmentForSubjectLine(string SubjectLine)
{
    var model = _mlContext.Model.Load(_modelFilePath, out var modelInputSchema);
    var emailsubject = new EmailSubject() { Subject = SubjectLine };
    _predictionEngine = _mlContext.Model.CreatePredictionEngine<EmailSubject, DepartmentPrediction>(model);

    var result = _predictionEngine.Predict(emailsubject);
    return result.Department;
}

void SaveModelAsFile()
{
    _mlContext.Model.Save(_model, _trainingDataView.Schema, _modelFilePath);
}

IEstimator<ITransformer> ProcessData()
{
    var pipeline = _mlContext.Transforms.Conversion.MapValueToKey(inputColumnName: "Department", outputColumnName: "Label")
        .Append(_mlContext.Transforms.Text.FeaturizeText(inputColumnName: "Subject", outputColumnName: "EmailSubjectFeaturized")
        .Append(_mlContext.Transforms.Concatenate("Features", "EmailSubjectFeaturized"))
        .AppendCacheCheckpoint(_mlContext));
    return pipeline;
}

IEstimator<ITransformer> BuildAndTrainModel(IDataView trainingDataView, IEstimator<ITransformer> pipeline)
{
    var trainingPipeline = pipeline.Append(_mlContext.MulticlassClassification.Trainers.SdcaMaximumEntropy("Label","Features")
        .Append(_mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel")));
    _model = trainingPipeline.Fit(trainingDataView);
    return trainingPipeline;

}

public class EmailSubject
{
    [LoadColumn(0)]
    public string Subject { get; set; }

    [LoadColumn(1)]
    public string Department { get; set; }
}

public class DepartmentPrediction
{
    [ColumnName("PredictedLabel")]
    public string Department { get; set; }
}