import scala.math.{Pi, cos, exp}
import com.fasterxml.jackson.core.{JsonFactory, JsonToken}
import org.apache.log4j.LogManager
import scala.collection.Iterator


object TimeConversion{
	private val hh = Pi / 12
	private val dd = 2 * Pi / 31
	def encode_date(date: String) = cos(date.takeRight(2).toInt * dd)
	def encode_hour(hour: String) = cos(hour.take(2).toInt * hh)
}


class SoftLogistic(k:Double=1, intercept:Double=0, L:Double=1){
	def predict(x:Double):Double = L/(1 + exp(-k*x - intercept))
	def predict(X:Seq[Double]):Seq[Double] = X.map(predict)
}


object CodecJson {
	private val logger = LogManager.getLogger(this.getClass.toString)
	private val jsonFactory = new JsonFactory

	def parseJson(line: String):Map[String, String] = {
		logger.trace("===>>> Inicializado o parser do JSON ...")
		val jp = jsonFactory.createParser(line)
		try Iterator.continually(jp).map(_.nextValue).drop(1).takeWhile(_ != JsonToken.END_OBJECT).map(_ => (jp.getCurrentName, jp.getText)).toMap
		catch {
			case e:Exception =>	logger.warn(s"JSON invalid format: ${line}")
								Map[String, String]()
		}
	}
}
