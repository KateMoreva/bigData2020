package sample

import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.GBTClassifier
import org.apache.spark.ml.feature._
import org.apache.spark.mllib.feature.Stemmer
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types.DataTypes

object HomeWork2 {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession
      .builder()
      .appName("Task2")
      .config("spark.master", "local")
      .getOrCreate()

    spark.sparkContext.setLogLevel("OFF")

    val testSet = spark
      .read.format("csv")
      .option("header", "true")
      .option("inferSchema", "true")
      .load("src/main/resources/test.csv")
      .filter(col("id").isNotNull)
      .filter(col("text").isNotNull)
      .select("id", "text")

    val trainSet = spark
      .read.format("csv")
      .option("header", "true")
      .option("inferSchema", "true")
      .load("src/main/resources/train.csv")
      .filter(col("id").isNotNull)
      .filter(col("target").isNotNull)
      .filter(col("text").isNotNull)
      .select("id", "text", "target")
      .withColumnRenamed("target", "tLabel")

    val sampleSet = spark
      .read.format("csv")
      .option("header", "true")
      .option("inferSchema", "true")
      .load("src/main/resources/sample_submission.csv")
      .select("id")

    val token = new RegexTokenizer()
      .setInputCol("text")
      .setOutputCol("words")
      .setPattern("[\\W]")

    val stwr = new StopWordsRemover()
      .setInputCol(token.getOutputCol)
      .setOutputCol("removed")

    val stemmer = new Stemmer()
      .setInputCol(stwr.getOutputCol)
      .setOutputCol("stemmed")
      .setLanguage("English")

    val hashingTF = new HashingTF()
      .setNumFeatures(10000)
      .setInputCol(stemmer.getOutputCol)
      .setOutputCol("rawFeatures")

    val idf = new IDF()
      .setInputCol(hashingTF.getOutputCol)
      .setOutputCol("features")

    val labelIndexer = new StringIndexer()
      .setInputCol("tLabel")
      .setOutputCol("indexedLabel")

    val rideg = new GBTClassifier()
      .setLabelCol("indexedLabel")
      .setFeaturesCol("features")
      .setPredictionCol("target")
      .setMaxIter(20)

    val pipe = new Pipeline()
      .setStages(Array(token, stwr, stemmer, hashingTF, idf, labelIndexer, rideg))
    var result = pipe.fit(trainSet).transform(testSet).select(col("id"), col("target").cast(DataTypes.IntegerType))

    result = result.join(sampleSet, sampleSet.col("id").equalTo(result.col("id")), "right")
      .select(sampleSet.col("id"), when(result.col("id").isNull, lit(0)).otherwise(col("target")).as("target"))

    result.write.option("header", "true").option("inferSchema", "true").csv("src/main/resources/sample_submission11.csv")

     }
}
