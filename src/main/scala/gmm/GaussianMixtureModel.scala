package gmm

import scala.language.implicitConversions
import scala.util.Random
import scala.math.{abs, log}
import scala.collection.JavaConverters._
import java.lang.{Double => jDouble}
import java.util.{List => jList}
import java.io.{ByteArrayInputStream, ByteArrayOutputStream}
import org.apache.avro.Schema
import org.apache.avro.generic.{GenericData, GenericDatumReader, GenericDatumWriter, GenericRecord}
import org.apache.avro.io.{DecoderFactory, EncoderFactory}
import org.apache.avro.specific.{SpecificDatumReader, SpecificDatumWriter}
import org.xerial.snappy.Snappy
import gmm.avro.AvroGmm
//import JulLogger

trait SelectGmmType {type GaussianMixtureModel = GaussianMixtureModelOjAlgo}//#ojAlgo
//trait SelectGmmType {type GaussianMixtureModel = GaussianMixtureModelNd4j}//#Nd4j
//trait SelectKmeansType {type KmeansLocal = KmeansWeka}//#weka
trait SelectKmeansType {type KmeansLocal = KmeansSmile}//#smile


trait GaussianMixtureModel{
	def weights:Array[Double]
	def means:Array[Array[Double]]
	def covariances:Array[Array[Array[Double]]]
	def precisions_cholesky:Array[Array[Array[Double]]]
	var converged:Boolean = _
	var lower_bound:Double = _
	var n_iter:Int= _
	def fit(X:Array[Array[Double]]):GaussianMixtureModel
	def score_samples(X:Array[Array[Double]]):Array[Double]
	def score(X:Array[Array[Double]]):Double
	def bic(X:Array[Array[Double]]):Double
}


object GaussianMixtureModel extends SelectGmmType{
	private implicit def arrayJavaList(arr:Array[Double]):jList[jDouble] = arr.map(jDouble.valueOf).toList.asJava
	private implicit def array2DJavaList(arr:Array[Array[Double]]):jList[jList[jDouble]] = (arr map arrayJavaList).toList.asJava
	private implicit def array3DJavaList(arr:Array[Array[Array[Double]]]):jList[jList[jList[jDouble]]] = (arr map array2DJavaList).toList.asJava

	def encodeAvro(gmm:GaussianMixtureModel):Array[Byte]={
		val avro = new AvroGmm(gmm.weights, gmm.means, gmm.precisions_cholesky)
		val baos = new ByteArrayOutputStream
		val encoder = EncoderFactory.get().binaryEncoder(baos, null)
		val datumWriter = new SpecificDatumWriter[AvroGmm](AvroGmm.getClassSchema)
		datumWriter.write(avro, encoder)
		encoder.flush()
		Snappy.compress(baos.toByteArray)
	}

	def encodeAvro(gmm:GaussianMixtureModel, schema_path:String):Array[Byte]={
		val schema = new Schema.Parser().parse(schema_path)
		val data = new GenericData.Record(schema)
		data.put("weights", gmm.weights)
		data.put("means", gmm.means)
		data.put("precisions", gmm.precisions_cholesky)
		val baos = new ByteArrayOutputStream
		val encoder = EncoderFactory.get().binaryEncoder(baos, null)
		val datumWriter = new GenericDatumWriter[GenericRecord](schema)
		datumWriter.write(data, encoder)
		encoder.flush()
		Snappy.compress(baos.toByteArray)
	}

	private implicit def javaListArray(arr:jList[jDouble]):Array[Double] = arr.asScala.map(_.toDouble).toArray
	private implicit def javaList2DArray(arr:jList[jList[jDouble]]):Array[Array[Double]] = arr.asScala.map(javaListArray).toArray
	private implicit def javaList3DArray(arr:jList[jList[jList[jDouble]]]):Array[Array[Array[Double]]] = arr.asScala.map(javaList2DArray).toArray

	def decodeAvro(snappyByteArray:Array[Byte]):GaussianMixtureModel={
		val bais = new ByteArrayInputStream(Snappy.uncompress(snappyByteArray))
		val decoder = DecoderFactory.get().binaryDecoder(bais, null)
		val datumReader = new SpecificDatumReader[AvroGmm](AvroGmm.getClassSchema)
		val avro = datumReader.read(null, decoder)
		new GaussianMixtureModel(weights_init=avro.getWeights, means_init=avro.getMeans, precisions_init=avro.getPrecisions)
	}

	def decodeAvro(snappyByteArray:Array[Byte], schema_path:String):GaussianMixtureModel={
		val bais = new ByteArrayInputStream(Snappy.uncompress(snappyByteArray))
		val decoder = DecoderFactory.get().binaryDecoder(bais, null)
		val schema = new Schema.Parser().parse(schema_path)
		val datumReader = new GenericDatumReader[GenericRecord](schema)
		val avro = datumReader.read(null, decoder)
		new GaussianMixtureModel(weights_init=avro.get("weights").asInstanceOf[jList[jDouble]], means_init=avro.get("weights").asInstanceOf[jList[jList[jDouble]]],
			precisions_init=avro.get("precisions").asInstanceOf[jList[jList[jList[jDouble]]]])
	}
}


abstract class GaussianMixtureBase[V,M,C](n_components:Int, tol:Double, reg_covar:Double, max_iter:Int, n_init:Int, init_params:String,
	weights_init:Array[Double], means_init:Array[Array[Double]], precisions_init:Array[Array[Array[Double]]], random_state:Option[Int], verbose:Int, verbose_interval:Int) extends GaussianMixtureModel{
	if (random_state.isDefined) Random.setSeed(random_state.get)
	//	val logger = JulLogger.logger//(getClass.getName)
	def weights:Array[Double] = tensor2vector(wgt)
	def means:Array[Array[Double]] = tensor2matrix(mu)
	def covariances:Array[Array[Array[Double]]] = tensor2cube(cov)
	def precisions_cholesky:Array[Array[Array[Double]]] = tensor2cube(pch)
	var wgt:V
	var mu:M
	var cov:C
	var pch:C
	private var init_prev_time:Long = _
	private var iter_prev_time:Long = _

	def tensor2vector(X:V):Array[Double]
	def tensor2matrix(X:M):Array[Array[Double]]
	def tensor2cube(X:C):Array[Array[Array[Double]]]

	def check_X(X:M, n_components:Int):Unit

	private def check_initial_parameters(n_components:Int, tol:Double, n_init:Int, max_iter:Int, reg_covar:Double)={
		if (n_components < 1) throw new UnsupportedOperationException(s"Invalid value for 'n_components': $n_components\nEstimation requires at least one component")
		if (tol < 0) throw new UnsupportedOperationException(s"Invalid value for 'tol': $tol\nTolerance used by the EM must be non-negative")
		if (n_init < 1) throw new UnsupportedOperationException(s"Invalid value for 'n_init': $n_init\nEstimation requires at least one run")
		if (max_iter < 1) throw new UnsupportedOperationException(s"Invalid value for 'max_iter': $max_iter \nEstimation requires at least one iteration")
		if (reg_covar < 0) throw new UnsupportedOperationException(s"Invalid value for 'reg_covar': $reg_covar\nRegularization on covariance must be non-negative")
	}

	private def print_verbose_msg_init_beg(n_init:Int, verbose:Int)={
		if (verbose == 1) println(s"Initialization $n_init")
		else if (verbose >= 2) {
			println(s"Initialization $n_init")
			init_prev_time = System.currentTimeMillis()
			iter_prev_time = init_prev_time
		}
	}

	private def print_verbose_msg_iter_end(n_iter:Int, diff_ll:Double, verbose:Int, verbose_interval:Int)={
		if (n_iter % verbose_interval == 0)
			if (verbose == 1) println(s"  Iteration $n_iter")
			else if (verbose >= 2) {
				val cur_time = System.currentTimeMillis
				println(s"  Iteration $n_iter\t time lapse ${(cur_time-iter_prev_time)/1000.0}\t ll change $diff_ll")
				iter_prev_time = cur_time
			}
	}

	private def print_verbose_msg_init_end(ll:Double, converged:Boolean, verbose:Int)={
		if (verbose == 1) println(s"Initialization converged: $converged")
		else if(verbose >= 2) println(s"Initialization converged: $converged\t time lapse ${(System.currentTimeMillis-init_prev_time)/1000.0}\t ll $ll")
	}

	def initialize_parameters(X:M, n_components:Int, init_params:String, reg_covar:Double):Unit

	def initialize(X:M, resp:M, reg_covar:Double):Unit

	def estimate_gaussian_parameters(X:M, resp:M, reg_covar:Double):(V,M,C)

	def estimate_gaussian_covariances_full(resp:M, X:M, nk:V, means:M, reg_covar:Double):C

	def compute_precision_cholesky(covariances:C, reg_covar:Double):C

	def e_step(X:M):(Double,M)

	def estimate_log_prob_resp(X:M):(V, M)

	def estimate_weighted_log_prob(X:M, means:M, weights:V, precisions_cholesky:C):M

	def estimate_log_prob(X:M, means:M, precisions_cholesky:C):M = estimate_log_gaussian_prob(X, means, precisions_cholesky)

	def estimate_log_gaussian_prob(X:M, means:M, precisions_cholesky:C):M

	def compute_log_det_cholesky(precisions_cholesky: C):V

	def estimate_log_weights(weights:V):V

	def logsumexp(X:M):V

	def m_step(X:M, log_resp:M, reg_covar:Double):Unit

	def compute_lower_bound(log_prob_norm:Double):Double = log_prob_norm

	def get_parameters():Map[String,Any] = Map("weights"->wgt, "means"->mu, "covariances"->cov, "precisions_cholesky"->pch)

	def set_parameters(best_params:Map[String, Any])={
		wgt = best_params("weights").asInstanceOf[V]
		mu = best_params("means").asInstanceOf[M]
		cov = best_params("covariances").asInstanceOf[C]
		pch = best_params("precisions_cholesky").asInstanceOf[C]
	}

	def score_samples(X:M):V = {
		val ewlp = estimate_weighted_log_prob(X, mu, wgt, pch)
		logsumexp(ewlp)
	}

	def score(X:Array[Array[Double]]):Double={
		val S = score_samples(X)
		S.sum/S.length
	}

	def bic(X:Array[Array[Double]]):Double = -2 * score(X) * X.length + n_parameters(n_components, X.head.length) * log(X.length)

	def n_parameters(n_components:Int, n_features:Int):Int = {
		val cov_params = n_components * n_features * (n_features + 1) / 2.0
		val mean_params = n_features * n_components
		(cov_params + mean_params + n_components - 1).toInt
	}

	override def toString:String={
		var str = getClass.getName + "\n"
		str += s"Lower bound: $lower_bound\nn_iter: $n_iter\nConverged: $converged\n"
		str += s"Components: ${weights.length}\nWeights:\n${weights.deep.mkString(", ")}\nMeans:\n${means.deep.mkString("\n")}\nPrecision Cholesky:\n${precisions_cholesky.deep.mkString("\n")}"
		str
	}

	def debug(A:V):Unit

	def fit(X:M):GaussianMixtureBase[V,M,C]={
		check_X(X, n_components)
		check_initial_parameters(n_components, tol, this.n_init, max_iter, reg_covar)
		val do_init = true//!warm_start esquece isso!
		val n_init_local = if (do_init) this.n_init else 1
		converged = false
		var max_lower_bound = Double.NegativeInfinity
		var best_params:Map[String, Any] = null
		var best_n_iter:Int = -1
		val init = for (init <- 0 until n_init_local)yield{
			print_verbose_msg_init_beg(init, verbose)
			initialize_parameters(X, n_components, init_params, reg_covar)
			lower_bound = Double.NegativeInfinity
			var break = false
			val n_iter = for(iter <- 0 until max_iter if !break)yield{
				val prev_lower_bound = lower_bound
				val (log_prob_norm, log_resp) = e_step(X)
				m_step(X, log_resp, reg_covar)
				lower_bound = compute_lower_bound(log_prob_norm)
				val change = lower_bound - prev_lower_bound
				print_verbose_msg_iter_end(iter, change, verbose, verbose_interval)
				if (abs(change) < tol) {
					converged = true
					break = true
				}
				iter
			}
			print_verbose_msg_init_end(lower_bound, converged, verbose)
			if (lower_bound > max_lower_bound) {
				max_lower_bound = lower_bound
				best_params = get_parameters()
				best_n_iter = n_iter.last
			}
			init
		}
		set_parameters(best_params)
		this.n_iter = best_n_iter
		if (!converged)	println(s"Initialization ${init.head+1} did not converge. \nTry different init parameters, \nor increase max_iter, tol \nor check for degenerate data.")
		this
	}
}
