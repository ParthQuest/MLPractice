using Microsoft.ML;

//https://docs.microsoft.com/en-us/dotnet/machine-learning/tutorials/predict-prices
//https://docs.microsoft.com/en-us/dotnet/machine-learning/tutorials/predict-prices-with-model-builder
//https://github.com/jwood803/MLNet_CrashCourse
namespace MLPractice1
{
    class Program
    {
        //Path to our data-set
        static readonly string _dataPath = Path.Combine(Environment.CurrentDirectory, "Data", "housing.csv");
        public static void Main(string[] args)
        {
            //Creating ML.NET context, It creates machine learning environment
            //If a fixed seed is provided by seed, MLContext environment becomes deterministic, meaning that the results are repeatable and will remain the same across multiple runs.
            var mlContext = new MLContext(seed: 0);

            var data = mlContext.Data.LoadFromTextFile<HousingData>(_dataPath, hasHeader: true, separatorChar: ',');

            var split = mlContext.Data.TrainTestSplit(data, testFraction: 0.2);

            var model = Train(mlContext, split.TrainSet);

            Evaluate(mlContext, model, split.TestSet);

            TestSinglePrediction(mlContext, model);
        }

        public static ITransformer Train(MLContext mlContext, IDataView trainDataSet)
        {
            var features = trainDataSet.Schema.Select(col => col.Name)
                .Where(colName => colName != "MedianHouseValue" && colName != "OceanProximity").ToArray();

            features = features.Append("OceanProximityEncoded").ToArray();

            var pipeline = mlContext.Transforms.CopyColumns(outputColumnName: "Label", inputColumnName: "MedianHouseValue")
                .Append(mlContext.Transforms.Categorical.OneHotEncoding(outputColumnName: "OceanProximityEncoded", inputColumnName: "OceanProximity"))
                .Append(mlContext.Transforms.Concatenate("Features", features))
                .Append(mlContext.Regression.Trainers.FastTree());

            var model = pipeline.Fit(trainDataSet);

            return model;
        }

        public static void Evaluate(MLContext mlContext, ITransformer model, IDataView testDataSet)
        {
            var predictions = model.Transform(testDataSet);

            var metrics = mlContext.Regression.Evaluate(predictions, "Label", "Score");

            Console.WriteLine($"RSquared Score: {metrics.RSquared:0.##}");

            Console.WriteLine($"Root Mean Squared Error: {metrics.RootMeanSquaredError:#.##}");
        }

        public static void TestSinglePrediction(MLContext mlContext, ITransformer model)
        {
            var predictionFunction = mlContext.Model.CreatePredictionEngine<HousingData, HousingDataPrediction>(model);

            var houseSample = new HousingData()
            {
                Longitude = -122.25f,
                Latitude = 37.84f,
                HousingMedianAge = 52.0f,
                TotalRooms = 3104.0f,
                TotalBedrooms = 687.0f,
                Population = 1157.0f,
                Households = 647.0f,
                MedianIncome = 3.12f,
                MedianHouseValue = 0.0f, // To predict. Actual/Observed = 241400.0
                OceanProximity = "NEAR BAY"
            };

            var prediction = predictionFunction.Predict(houseSample);

            Console.WriteLine($"Predicted value: {prediction.MedianHouseValue:0.####}, actual value: 241400.0");
        }
    }
}