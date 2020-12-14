package sample

import org.apache.spark.ml.PipelineModel
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types.{DataTypes, StringType, StructType}

object HomeWork2 {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder()
      .appName("Task3")
      .master("local[*]")
      .getOrCreate()

    spark.sparkContext.setLogLevel("OFF")

    val scheme = new StructType().add("id", StringType, nullable = true).add("text", StringType, nullable = true)

    val inputData = spark.readStream
      .format("socket")
      .option("host", "localhost")
      .option("port", 8065)
      .load()

    val inputJson =
      inputData.withColumn("json", from_json(col("value"), scheme))
        .select("json.*")
        .select(col("id"), col("text"))


    val model = PipelineModel.read.load("model/")
    inputJson.printSchema()
    val result = model.transform(inputJson.select(col("id"), col("text")))
      .select(col("id"), col("target").as("target").cast(DataTypes.IntegerType))

    val query = result
      .repartition(1)
      .writeStream
      .outputMode("append")
      .format("com.databricks.spark.csv")
      .option("header", "true")
      .option("path", "path/")
      .option("checkpointLocation", "checkpointLocation/")
      .start()
      .awaitTermination()
  }

}
