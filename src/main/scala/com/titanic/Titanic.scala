package com.titanic

//import com.titanic.tmpTitanic2.{accuracyScore, convertCategoriesToNumericValue}
import org.apache.spark.sql.{DataFrame, Dataset, SaveMode, SparkSession}
import org.apache.spark.ml._
import org.apache.spark.ml.feature._
import org.apache.spark.ml.classification.{LogisticRegression, RandomForestClassifier}
import org.apache.spark.ml.clustering.KMeans
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types.IntegerType
import org.apache.spark.ml.tuning.ParamGridBuilder
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.tuning.CrossValidator
import org.apache.spark.ml.evaluation._
import org.apache.spark.mllib.classification.SVMWithSGD
import org.apache.spark.mllib.evaluation.{BinaryClassificationMetrics, MulticlassMetrics}

object Titanic  {

  def getAverageOfColumn(trainingData: DataFrame, testData: DataFrame, columnName: String) = {
    trainingData.select(columnName)
      .union(testData.select(columnName))
      .agg(avg(columnName))
      .head()
      .getDouble(0)
  }

  def doFeatureEngineering(spark:SparkSession, trainfilePath:String, testfilePath:String): (DataFrame, DataFrame)={
    var trainingData = spark.read.option("header", "true").option("inferSchema", "true").csv(trainfilePath)
    var testData = spark.read.option("header", "true").option("inferSchema", "true").csv(testfilePath)

    val averageAge = getAverageOfColumn(trainingData, testData, "Age")
    val averageFare = getAverageOfColumn(trainingData, testData, "Fare")

    trainingData = prepareAgeAndFareColumn(trainingData, averageAge, averageFare)
    testData = prepareAgeAndFareColumn(testData, averageAge, averageFare)

    trainingData = prepareTitleColumn(trainingData)
    testData = prepareTitleColumn(testData)

    trainingData = prepareFamilyColumn(trainingData)
    testData = prepareFamilyColumn(testData)

    (trainingData, testData)
  }

  def prepareAgeColumn(trainingDf:DataFrame, testDf:DataFrame): Unit ={
    val averageAge = getAverageOfColumn(trainingDf, testDf, "Age")
    val averageFare = getAverageOfColumn(trainingDf, testDf, "Fare")
    var trainData = trainingDf.na.fill(Map("Age" -> averageAge, "Fare" -> averageFare))
    var testData = testDf.na.fill(Map("Age" -> averageAge, "Fare" -> averageFare))

    //Bucketize the age column
    val ageBuckets = Array(0.0, 5.0, 18.0, 35.0, 60.0, 150.0)
    val bucketizer = new Bucketizer()
      .setInputCol("Age")
      .setOutputCol("AgeGroup")
      .setSplits(ageBuckets)

    trainData = bucketizer.transform(trainData).drop("Age").withColumnRenamed("AgeGroup", "Age")
  }

  def prepareTitleColumn(data:DataFrame): DataFrame = {
    val Pattern = ".*, (.*?)\\..*".r
    val titles = Map(
      "Mrs" -> "Mrs",
      "Lady" -> "Mrs",
      "Mme" -> "Mrs",
      "Ms" -> "Ms",
      "Miss" -> "Miss",
      "Mlle" -> "Miss",
      "Master" -> "Master",
      "Rev" -> "Rev",
      "Don" -> "Mr",
      "Sir" -> "Sir",
      "Dr" -> "Dr",
      "Col" -> "Col",
      "Capt" -> "Col",
      "Major" -> "Col"
    )
    val title: ((String, String) => String) = {
      case (Pattern(t), sex) => titles.get(t) match {
        case Some(tt) => tt
        case None =>
          if (sex == "male") "Mr"
          else "Mrs"
      }
      case _ => "Mr"
    }

    val titleUDF = udf(title)
    data.withColumn("Title", titleUDF(col("Name"), col("Sex")))
  }

  def prepareFamilyColumn(data: DataFrame): DataFrame={
    val familySize: ((Int, Int) => Int) = (sibSp: Int, parCh: Int) => sibSp + parCh + 1
    val familySizeUDF = udf(familySize)

    data.withColumn("FamilySize", familySizeUDF(col("SibSp"), col("Parch")))
  }

  def prepareAgeAndFareColumn(data:DataFrame, averageAge:Double, averageFare:Double): DataFrame={
    val tempData = data.na.fill(Map("Age" -> averageAge, "Fare" -> averageFare))

    //Bucketize the age column
    val ageBuckets = Array(0.0, 5.0, 18.0, 35.0, 60.0, 150.0)
    val bucketizer = new Bucketizer()
      .setInputCol("Age")
      .setOutputCol("AgeGroup")
      .setSplits(ageBuckets)

    bucketizer.transform(tempData).drop("Age").withColumnRenamed("AgeGroup", "Age")
  }

  def getModel(stages:Array[PipelineStage], data:DataFrame):(Pipeline, PipelineModel) = {
    val pipeline = new Pipeline().setStages(stages)

    (pipeline, pipeline.fit(data))
  }

  def convertCategoriesToNumericValue(column: String): Array[PipelineStage] = {
    val stringIndexer = new StringIndexer().setInputCol(column)
      .setOutputCol(s"${column}_index")
      .setHandleInvalid("skip")

    val oneHot = new OneHotEncoder().setInputCol(s"${column}_index").setOutputCol(s"${column}_encoded")
    Array(stringIndexer, oneHot)
  }

  def computeMetrics(df: DataFrame, label: String, predictCol: String) = {
    val rdd = df.select(label, predictCol).rdd.map(row ⇒ (row.getInt(0).toDouble, row.getDouble(1)))
    val trueNegative = df.select("*").where(s"${predictCol} = 0 AND ${label} = 0").count()
    val truePositive = df.select("*").where(s"${predictCol} = 1 AND ${label} = 1").count()
    val falseNegative = df.select("*").where(s"${predictCol} = 0 AND ${label} = 1").count()
    val falsePositive = df.select("*").where(s"${predictCol} = 1 AND ${label} = 0").count()

    var precision = truePositive/(truePositive + falsePositive).toDouble
    var recall = truePositive/(truePositive + falseNegative).toDouble
    var accuracy = new MulticlassMetrics(rdd).accuracy

    precision = BigDecimal(precision).setScale(2, BigDecimal.RoundingMode.HALF_UP).toDouble
    recall = BigDecimal(recall).setScale(2, BigDecimal.RoundingMode.HALF_UP).toDouble
    accuracy = BigDecimal(accuracy).setScale(2, BigDecimal.RoundingMode.HALF_UP).toDouble

    (accuracy, precision, recall)
  }

  def createCrossValidatorModel(pipeline: Pipeline, paramMap: Array[ParamMap], data: DataFrame): Model[_] = {
    val evaluator = new BinaryClassificationEvaluator()
      .setMetricName("areaUnderROC")

    val crossValidator = new CrossValidator()
      .setEstimator(pipeline)
      .setEvaluator(evaluator)
      .setEstimatorParamMaps(paramMap)
      .setNumFolds(10)

    crossValidator.fit(data)
  }

  def getPreProcessStages(): Array[PipelineStage] ={
    val genderStages = convertCategoriesToNumericValue("Sex")
    val embarkedStages = convertCategoriesToNumericValue("Embarked")
    val pClassStages = convertCategoriesToNumericValue("Pclass")
    val titleStages = convertCategoriesToNumericValue("Title")

    //Renaming features suitable for mllib library
    val cols = Array("Sex_encoded", "Embarked_encoded", "Pclass_encoded", "Title_encoded", /*"Parch", "SibSp",*/ "FamilySize", "Age", "Fare")
    val vectorAssembler = new VectorAssembler().setInputCols(cols).setOutputCol("features")

    genderStages ++ embarkedStages ++ pClassStages ++ titleStages ++ Array(vectorAssembler)
  }

  def showMetricResults(model:Model[_], trainData: DataFrame, text:String): Unit ={
    var (accuracy, precision, recall) = computeMetrics(model.transform(trainData), "label", "prediction")
    println(text)
    println("Accuracy = "+ accuracy + " Precision = " + precision + " Recall = " + recall)
  }

  def main(args: Array[String]) {
    val spark = SparkSession.builder.
      master("local[*]")
      .appName("Titanic")
      .getOrCreate()

    spark.sparkContext.setLogLevel("ERROR")
    val trainfilePath = "./src/main/resources/input/train.csv"
    val testfilePath = "./src/main/resources/input/test.csv"

    var (wholeData, testData) = doFeatureEngineering(spark, trainfilePath, testfilePath)

    var Array(trainingData, crossData) = wholeData.randomSplit(Array(0.7, 0.3))

    val trainData = trainingData.withColumnRenamed("survived", "label")

    val preProcessStages = getPreProcessStages()

    //Logistic Regression
    val logisticRegression = new LogisticRegression()
    val paramMapLR = new ParamGridBuilder()
      .addGrid(logisticRegression.regParam, Array(0.6, 0.1, 0.01, 0.001))
      .addGrid(logisticRegression.maxIter, Array(10,20, 50))
      .addGrid(logisticRegression.elasticNetParam, Array(0.1, 0.5))
      .build()

    //Random Forest
    val randomForestClassifier = new RandomForestClassifier()
    val paramMapRF = new ParamGridBuilder()
      .addGrid(randomForestClassifier.impurity, Array("gini", "entropy"))
      .addGrid(randomForestClassifier.maxDepth, Array(1, 2, 5, 10))
      .addGrid(randomForestClassifier.minInstancesPerNode, Array(1, 2, 4, 5))
      .build()

    val kmeans = new KMeans().setK(2).setSeed(1L)
    val paramMapKM = new ParamGridBuilder()
      .addGrid(kmeans.maxIter, Array(10,50,100,200))
      .build()

    //Serializing algorithms and tuning parameters
    val algorithmConfigurations = Array(
      (logisticRegression, paramMapLR),
      (randomForestClassifier, paramMapRF)
      //      (kmeans, paramMapKM)
    )

    val algorithmNames = Array("Logistic Regression", "Random Forest")
    var index = 0

    for((algorithm, paramMap) <- algorithmConfigurations){
      val (pipeline, model) = getModel(preProcessStages ++ Array(algorithm),trainData)

      println("<---------------  "+ algorithmNames(index) + "---------------->")
      println("[Pipeline]")
      println("Training Data:")
      println("(Accuracy, Precision, Recall) = "+ computeMetrics(model.transform(trainData), "label", "prediction"))

      println("Test Data:")
      println("(Accuracy, Precision, Recall) = " + computeMetrics(model.transform(crossData), "Survived", "prediction"))

      val crossValidationModel = createCrossValidatorModel(pipeline, paramMap, trainData)
      println("[CrossValidator]")
      println("Training Data:")
      println("(Accuracy, Precision, Recall) = "+ computeMetrics(crossValidationModel.transform(trainData), "label", "prediction"))

      println("Test Data:")
      println("(Accuracy, Precision, Recall) = " + computeMetrics(crossValidationModel.transform(crossData), "Survived", "prediction"))

      index = index+1

      val scoredDf = model.transform(testData)
      val outputDf = scoredDf.select("PassengerId", "prediction")
      val castedDf = outputDf.select(outputDf("PassengerId"), outputDf("prediction").cast(IntegerType).as("Survived"))
      castedDf.write.format("csv").option("header", "true").mode(SaveMode.Overwrite).save("src/main/resources/output/")
    }
  }
}
