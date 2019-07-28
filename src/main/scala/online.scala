import java.util.{Arrays, Properties}
import java.util.concurrent.{Executors, TimeUnit}
import akka.actor.{Actor, ActorSystem, Props}
import org.apache.hadoop.hbase.{HBaseConfiguration, TableName}
import org.apache.hadoop.hbase.client.{ConnectionFactory, Get, Table}
import org.apache.log4j.LogManager
import scala.math.{abs, log}
import gmm.{GaussianMixture, GaussianMixtureModel}
import org.apache.kafka.clients.consumer.KafkaConsumer
import org.apache.kafka.clients.producer.{KafkaProducer, ProducerRecord}
import scala.collection.Iterator
import scala.collection.JavaConverters._


class DataBus(input_topic:String, output_topic:String){
	private val src_props = new Properties
	src_props.setProperty("bootstrap.servers", "localhost:9092")
	src_props.setProperty("group.id", "rpca_consumer")
	src_props.setProperty("enable.auto.commit", "false")
	src_props.setProperty("auto.commit.interval.ms", "1000")
	src_props.setProperty("key.deserializer", "org.apache.kafka.common.serialization.StringDeserializer")
	src_props.setProperty("value.deserializer", "org.apache.kafka.common.serialization.StringDeserializer")
	private val consumer = new KafkaConsumer[String, String](src_props)
	consumer.subscribe(Arrays.asList(input_topic))
	consumer.poll(0) //start poll
	consumer.assignment.asScala.foreach(partition => consumer.seekToEnd(partition))
	val source = Iterator.continually(consumer.poll(Long.MaxValue).asScala).flatten.map(_.value)

	private val dst_props = new Properties
	dst_props.put("bootstrap.servers", "localhost:9092")
	dst_props.put("key.serializer", "org.apache.kafka.common.serialization.StringSerializer")
	dst_props.put("value.serializer", "org.apache.kafka.common.serialization.StringSerializer")
	private val sink = new KafkaProducer[String, String](dst_props)

	def write(msg:String) = sink.send(new ProducerRecord(output_topic, msg))
}


class GaussianModel(table:Table, cf:String, qf:String, regr:SoftLogistic){
	val header = List("dt","hr","vl")
	val scr_label = "score_model"

	def normalize(x:Double) = log(abs(x) + 1)

	def predict(json:String, verbose:Boolean=false):Map[String, Double] = {
		val js = CodecJson.parseJson(json)
		if (!header.forall(js.contains) || js("cd") != "2") return Map[String, Double]()

		val pk = js("cd").toInt
		val rk = f"${pk % 256}%02X$pk"
		val row = table.get(new Get(rk.getBytes))
		if (row.getRow == null) return Map(scr_label -> -1)

		val gmm = GaussianMixtureModel.decodeAvro(row.getValue(cf.getBytes, qf.getBytes))
		val dt = TimeConversion.encode_date(js("dt"))
		val hr = TimeConversion.encode_hour(js("hr"))
		val vl = js("vl").toDouble
		val input = Array(Array(dt, hr, vl))
		val score = GaussianMixture.truncate(regr.predict(normalize(gmm.score_samples(input).head)), 3)
//		val score = GaussianMixture.truncate(regr.predict(gmm.score_samples(input).head), 3)
		if (verbose) {
			println("row: ",row)
			println("gmm: ", gmm)
		}
		Map(scr_label -> score)
	}
}


class Estimator extends Actor {
	val onprop = Online.on_prop
	val hbprop = Online.hb_prop
	val lr = new SoftLogistic(onprop.getProperty("coef").toDouble, onprop.getProperty("intercept").toDouble)
	val table = Online.connection.getTable(TableName.valueOf(hbprop.getProperty("model_table")))
	val gm = new GaussianModel(table, hbprop.getProperty("column"), hbprop.getProperty("cell"), lr)

	override def receive: Receive = {
		case msg: String =>
			val scr = gm.predict(msg)
			Online.bus.write(s"""${msg.init}$scr}""")
	}
}


class ModelConsumer(source:Iterator[String], actorSystem:ActorSystem) extends Runnable{
	val prop = Online.on_prop
	val agents = Range(0, prop.getProperty("agents").toInt).map(x => actorSystem.actorOf(Props[Estimator]))//"ActorId-"+x
	val poolagents = Iterator.continually(agents).flatten

	override def run() = source.foreach(msg => poolagents.next ! msg)
}


object Online{
	val path = ""
	val logger = LogManager.getLogger(this.getClass.toString)
	val actorSystem = ActorSystem("OnlineActorSystem")
	val bus = new DataBus("input_topic", "output_topic")
	val hb_prop = new Properties
	hb_prop.load(getClass.getResourceAsStream(s"$path/hbase.properties"))
	val cf = hb_prop.getProperty("column_family")
	val hbconf = HBaseConfiguration.create()
	hbconf.set("hbase.zookeeper.quorum", hb_prop.getProperty("zookeeper"))
	val connection = ConnectionFactory.createConnection(hbconf)//pesado
	val on_prop = new Properties
	on_prop.load(getClass.getResourceAsStream(s"$path/online.properties"))

	def main(args: Array[String]){
		logger.info("*"*150)
		logger.info("Iniciando aplicacao online")
		val ncons = on_prop.getProperty("consumers").toInt
		val consumers = Range(0, ncons).map(_ => new ModelConsumer(bus.source, actorSystem))
		val pool = Executors.newFixedThreadPool(ncons)
		consumers.foreach(pool.execute)
		try{
			if (!pool.awaitTermination(Long.MaxValue, TimeUnit.DAYS)) logger.error("Timed out waiting for consumer threads to shut down, exiting uncleanly")
		}catch{
			case e:Exception => logger.error("Interrupted during shutdown, exiting uncleanly")
		}finally{
			actorSystem.terminate()
			pool.shutdown()
		}
	}
}
