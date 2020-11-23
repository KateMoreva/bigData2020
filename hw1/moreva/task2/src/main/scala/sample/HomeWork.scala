package sample

import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.expressions.Window
import org.apache.spark.sql.functions._

object HomeWork {

  def main(args: Array[String]): Unit = {
    val spark = SparkSession
      .builder()
      .appName("Task1")
      .config("spark.master", "local")
      .getOrCreate()

    spark.sparkContext.setLogLevel("OFF")

    val dataSet = spark
      .read
      .option("header", "true")
      .option("mode", "DROPMALFORMED")
      .option("escape", "\"")
      .csv("src/main/resources/AB_NYC_2019.csv")

    val features = dataSet
      .withColumn("price", dataSet("price").cast("Integer"))
      .withColumn("latitude", dataSet("latitude").cast("Double"))
      .withColumn("longitude", dataSet("longitude").cast("Double"))
      .withColumn("minimum_nights", dataSet("minimum_nights").cast("Integer"))
      .withColumn("number_of_reviews", dataSet("number_of_reviews").cast("Integer"))
      .where(col("price") > 0)
      .na
      .drop()



    features.createOrReplaceTempView("features")

    println("Mean")
    spark.sql("select room_type, mean(price) as mean from features group by room_type ").show()

    println("Median")
    spark.sql("select room_type, percentile_approx(price, 0.5) as median from features group by room_type ").show()

    println("StDev")
    spark.sql("select room_type, std(price) as std from features group by room_type").show()

    println("Mode")
    features
      .groupBy("room_type", "price")
      .count().withColumn("row_number", row_number().over(Window.partitionBy("room_type").orderBy(desc("count"))))
      .select("room_type", "price")
      .where(col("row_number") === 1)
      .show()

    println("Most expensive offer")
    features.orderBy(desc("price")).show(1)

    println("Cheapest offer")
    features.orderBy("price").show(1)

    println("Correlation between price and minimum_nights")
    spark.sql("select corr(price, minimum_nights) as Correlation  from features ").show()

    println("Correlation between price and number_of_reviews")
    spark.sql("select corr(price, number_of_reviews) as Correlation from features ").show()

    val base32 = "0123456789bcdefghjkmnpqrstuvwxyz"

    val encode = ( lat:Double, lng:Double, precision:Int )=> {
      var (minLat,maxLat) = (-90.0,90.0)
      var (minLng,maxLng) = (-180.0,180.0)
      val bits = List(16,8,4,2,1)

      (0 until precision).map{ p => {
        base32 apply (0 until 5).map{ i => {
          if (((5 * p) + i) % 2 == 0) {
            val mid = (minLng+maxLng)/2.0
            if (lng > mid) {
              minLng = mid
              bits(i)
            } else {
              maxLng = mid
              0
            }
          } else {
            val mid = (minLat+maxLat)/2.0
            if (lat > mid) {
              minLat = mid
              bits(i)
            } else {
              maxLat = mid
              0
            }
          }
        }}.reduceLeft( (a,b) => a|b )
      }}.mkString("")
    }

    def decodeBounds( geohash:String ):((Double,Double),(Double,Double)) = {
      def toBitList( s:String ) = s.flatMap{
        c => ("00000" + base32.indexOf(c).toBinaryString).takeRight(5).map('1' == ) } toList

      def split( l:List[Boolean] ):(List[Boolean],List[Boolean]) ={
        l match{
          case Nil => (Nil,Nil)
          case x::Nil => ( x::Nil,Nil)
          case x::y::zs => val (xs,ys) =split( zs );( x::xs,y::ys)
        }
      }

      def dehash( xs:List[Boolean] , min:Double,max:Double):(Double,Double) = {
        ((min,max) /: xs ){
          case ((min,max) ,b) =>
            if( b )( (min + max )/2 , max )
            else ( min,(min + max )/ 2 )
        }
      }

      val ( xs ,ys ) = split( toBitList( geohash ) )
      ( dehash( ys ,-90,90) , dehash( xs, -180,180 ) )
    }

    def decode( geohash:String ):(Double,Double) = {
      decodeBounds(geohash) match {
        case ((minLat,maxLat),(minLng,maxLng)) => ( (maxLat+minLat)/2, (maxLng+minLng)/2 )
      }
    }

    val encode_udf = udf(encode)
    val features_geohash = features
      .withColumn("geoHash", encode_udf(col("latitude"), col("longitude"), lit(5)))
      .groupBy("geoHash")
      .mean("price")
      .orderBy(desc("avg(price)"))
      .first()
    println("The most expensive area 5x5 km in New-York:")
    print(decode(features_geohash.get(0).toString))
    print(" price: ")
    print(features_geohash.get(1))
  }
}
