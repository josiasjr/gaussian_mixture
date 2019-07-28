import java.util.Properties
import org.apache.spark.SparkConf
import org.apache.spark.sql.SparkSession
//import org.apache.spark.sql.types.{StringType, StructField, StructType}

import gmm.{GaussianMixture, GaussianMixtureModel}
import org.apache.hadoop.hbase.client.{ConnectionFactory, Put}
import org.apache.hadoop.hbase.{HBaseConfiguration, TableName}
import org.apache.spark.sql.catalyst.expressions.GenericRowWithSchema
import org.apache.log4j.LogManager

import scala.util.Random
import scala.util.hashing.MurmurHash3

//--conf "spark.executor.extraJavaOptions=-XX:+PrintGCDetails -XX:+PrintGCTimeStamps"

object Offline{
	val path = ""
	val hb_prop = new Properties
	hb_prop.load(getClass.getResourceAsStream(s"$path/hbase.properties"))
	val tableName = hb_prop.getProperty("model_table")
	val cf = hb_prop.getProperty("column")
	val hbconf = HBaseConfiguration.create()
	hbconf.set("hbase.zookeeper.quorum", hb_prop.getProperty("zookeeper"))
	val connection = ConnectionFactory.createConnection(hbconf)

	def toDouble(x:String) = x.toDouble
	val logger = LogManager.getLogger(this.getClass.toString)
	val off_prop = new Properties
	off_prop.load(getClass.getResourceAsStream(s"$path/offline.properties"))
	val limit_size = off_prop.getProperty("limit_size").toInt

	def main(args: Array[String]): Unit = {
		val start = System.currentTimeMillis
		val scconf = new SparkConf().setAppName("SparkAppModel")
			.set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
		    .set("spark.sql.tungsten.enabled","true")
			.set("spark.ui.killEnabled", "true")
		val spark = SparkSession
			.builder()
			.config(scconf)
			.getOrCreate()
		spark.udf.register("encode_date", TimeConversion.encode_date _)
		spark.udf.register("encode_hour", TimeConversion.encode_hour _)
		spark.udf.register("toDouble", toDouble _)

		val csv = spark.read.parquet(off_prop.getProperty("input_files"))
//		val header_schema = off_prop.getProperty("header_schema").split(',')
//		val schema = StructType(header_schema.map(x => StructField(x, StringType, true)))
//		val csv = spark.read.option("delimiter", "\u0001").schema(schema).csv(off_prop.getProperty("input_files"))
		csv.createOrReplaceTempView({off_prop.getProperty("spark_table")})
		val tfi = spark.sql(off_prop.getProperty("sql"))//.cache()
		val rows = tfi.rdd.map { row =>
			val pk = row.getAs[Int](0)
			val Xt = row.getSeq[GenericRowWithSchema](1).map(_.toSeq.toArray.map(_.asInstanceOf[Double]))
			val X = if (Xt.length > limit_size) Random.shuffle(Xt).take(limit_size).toArray else Xt.toArray
			val gx = new GaussianMixture(off_prop.getProperty("components").toInt, off_prop.getProperty("n_init").toInt)
			val (gmm, info) = gx.train_eval(X)
			val gmm_out = if (gmm != null){
				GaussianMixtureModel.encodeAvro(gmm)
			} else {
				logger.warn(s"$pk not processed. $info")
				null
			}
			Map("pk" -> pk, "gmm" -> gmm_out, "info" -> info)
		}//.filter(_.nonEmpty)//.cache()
		val out = rows.mapPartitions{part =>
			val table = connection.getTable(TableName.valueOf(tableName))
			part.map{r =>
				val pk = r("pk").asInstanceOf[Int]
				val msg = r("info").asInstanceOf[String]
				if (r("gmm") != null) {
					val rk = f"${pk % 256}%02X$pk"
					val put = new Put(rk.getBytes)
					put.addColumn(cf.getBytes, "m".getBytes, r("gmm").asInstanceOf[Array[Byte]])
					table.put(put)
					s"""{"mci":$pk,"rowkey":$rk,${msg.tail}"""
				} else {
					val msg = r("info").asInstanceOf[String]
					val hsh = MurmurHash3.stringHash(msg).toHexString.toUpperCase
					s"""{"mci":$pk,"cod":"$hsh","error":"$msg"}"""
				}
			}
		}
		out.saveAsTextFile(off_prop.getProperty("output_info"))//, classOf[GzipCodec]
		println(s"Time elapsed: ${(System.currentTimeMillis - start)/1e3}s")
	}
}
