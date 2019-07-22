package gmm

import java.io.{File, FileOutputStream, PrintWriter}
import java.nio.file.{Files, Paths}
import java.util.Properties
import com.fasterxml.jackson.databind.ObjectMapper
import com.fasterxml.jackson.module.scala.DefaultScalaModule

//import scala.Console.{RED, RESET, UNDERLINED}
import scala.io.Source
import scala.util.Random
import scala.language.reflectiveCalls

//case class EvaluateJson(y_error:Double, ylen:Int, x_error:Double, xlen:Int, total:Int, bic:Double, clock:Double)
case class EvaluateJson(ylen:Int, xlen:Int, total:Int, time:Double, bic:Double)
case class TrainJson(total:Int, time:String, bic:Double)


class GaussianMixture(n_components:Int=1, n_init:Int=1, init_params:String="kmeans") extends SelectGmmType{
	private var Xc:Array[Array[Double]] = _
	private val mapper = new ObjectMapper()
	mapper.registerModule(DefaultScalaModule)
	var gmm:GaussianMixtureModel = null

	def this(prop:Properties){
		this(
			prop.getProperty("components").toInt,
			prop.getProperty("n_init").toInt,
			prop.getProperty("init_params")
		)
	}

	def train_eval(X:Array[Array[Double]]):(GaussianMixtureModel, String) = {
		try {
			val start = GaussianMixture.clock
			val gmm = train(X)
			val total = X.length
			val time = GaussianMixture.truncate(GaussianMixture.clock - start, 3) + "s"
			val bic = GaussianMixture.truncate(gmm.bic(X), 3)
			(gmm, mapper.writeValueAsString(TrainJson(total, time, bic)))
		}catch{
			case e:Exception =>
				//Log.logger.warning(e.toString)
				//e.getStackTrace.foreach(x => Log.logger.warning(s"\tat $x"))
				//println(RED + UNDERLINED + e.toString + RESET)
				//e.getStackTrace.foreach(x => println(s"$RED$UNDERLINED\tat $x$RESET"))
				(null, e.getMessage)
		}
	}

	def train(X:Array[Array[Double]]):GaussianMixtureModel = {
		Xc = X
		gmm = new GaussianMixtureModel(n_components = n_components, n_init = n_init, init_params = init_params)
		gmm.fit(X)
		gmm
	}

	def evaluate(ratio:Double, iter:Int)={
		val start = GaussianMixture.clock
		val eval = (for(_ <- 0 until iter)yield{
			val res = Random.shuffle(Xc.toSeq).splitAt((Xc.length * ratio).toInt)
			val (train, test) = (res._1.toArray, res._2.toArray)
			val gmm = new GaussianMixtureModel(n_components=n_components, n_init=n_init, init_params=init_params)
			gmm.fit(train)
			val y_pred = gmm.score_samples(test)
			val x_pred = gmm.score_samples(train)
			(regScore(y_pred), regScore(x_pred))
		}).unzip
		val yscr = eval._1.flatten
		val y_error = yscr.sum/yscr.length.toDouble
		val ylen = yscr.length
		val xscr = eval._2.flatten
		val x_error = xscr.sum/xscr.length.toDouble
		val xlen = xscr.length
		val total = Xc.length
		val bic = GaussianMixture.truncate(gmm.bic(Xc), 3)
		val delta = GaussianMixture.truncate(GaussianMixture.clock - start, 3)
//		mapper.writeValueAsString(EvaluateJson(truncate(y_error, 2), ylen, truncate(x_error, 2), xlen, total, bic, (tock - tick)/1e3))
		mapper.writeValueAsString(EvaluateJson(ylen, xlen, total, delta, bic))
//		s"$init_params,$delta,$bic"
	}

	def regScore(scr:Array[Double]) = {
		val rand = new Random(seed = 0)
		Range(0, scr.length).map(_ => rand.nextInt(2))
	}
}

object GaussianMixture extends SelectGmmType {
	def saveModel(gmm: GaussianMixtureModel, local_path: String, schema_path: String = null) = {
		val avro = if (schema_path == null) GaussianMixtureModel.encodeAvro(gmm) else GaussianMixtureModel.encodeAvro(gmm, schema_path)
		val faos = new FileOutputStream(local_path)
		faos.write(avro)
		faos.close()
	}

	def loadModel(path: String, schema_path: String = null): GaussianMixtureModel = {
		val byteArray = Files.readAllBytes(Paths.get(path))
		if (schema_path == null) GaussianMixtureModel.decodeAvro(byteArray) else GaussianMixtureModel.decodeAvro(byteArray, schema_path)
	}

	private def clock() = System.currentTimeMillis / 1e3

	def truncate(n: Double, p: Int): Double = {
		val s = math pow(10, p);
		(math round n * s) / s
	}
}
