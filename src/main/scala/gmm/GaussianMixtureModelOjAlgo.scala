package gmm

import java.lang.{Double => jDouble}
import org.ojalgo.matrix.PrimitiveMatrix
import org.ojalgo.matrix.decomposition.{Cholesky => CholeskyOjAlgo}
import org.ojalgo.matrix.store.{MatrixStore, PrimitiveDenseStore}
import org.ojalgo.matrix.task.InverterTask
import scala.collection.mutable.ListBuffer
import scala.language.implicitConversions
import scala.math.{Pi, exp, log, pow}
import scala.util.Random


class GaussianMixtureModelOjAlgo(n_components:Int=1, tol:Double=0.001, reg_covar:Double=1e-6, max_iter:Int=300, n_init:Int=1, init_params:String="random",
	weights_init:Array[Double]=null, means_init:Array[Array[Double]]=null, precisions_init:Array[Array[Array[Double]]]=null, random_state:Option[Int]=None, verbose:Int=0, verbose_interval:Int=10)
	extends GaussianMixtureBase[Array[Double],Array[Array[Double]],Array[Array[Array[Double]]]](n_components, tol, reg_covar, max_iter, n_init, init_params, weights_init, means_init, precisions_init, random_state, verbose, verbose_interval)
	with SelectKmeansType {
//	val logger = JulLogger.logger//(getClass.getName)
	val eps:Double = math.ulp(0.5)
	var wgt = weights_init
	var mu = means_init
	var cov:Array[Array[Array[Double]]] = _
	var pch = precisions_init
	private val sF = PrimitiveDenseStore.FACTORY
	private val mF = PrimitiveMatrix.FACTORY
	private val cF = CholeskyOjAlgo.PRIMITIVE
	def tensor2vector(X:Array[Double]):Array[Double] = X
	def tensor2matrix(X:Array[Array[Double]]):Array[Array[Double]] = X
	def tensor2cube(X:Array[Array[Array[Double]]]):Array[Array[Array[Double]]] = X

	def compute_precision_cholesky(covariances:Array[Array[Array[Double]]], reg_covar:Double):Array[Array[Array[Double]]]={
		val precisions_chol = ListBuffer[MatrixStore[jDouble]]()
		for (covariance <- covariances) {
			val cov_chol = cF.make
			cov_chol decompose sF.rows(covariance: _*)
			precisions_chol += InverterTask.PRIMITIVE.make(cov_chol.getL).invert(cov_chol.getL).transpose
		}
		precisions_chol.toArray.map(_.toRawCopy2D)
	}

	def compute_log_det_cholesky(matrix_chol: Array[Array[Array[Double]]]):Array[Double] = matrix_chol.map(r => sF.rows(r: _*)).map(_.sliceDiagonal(0,0)).map(_.toRawCopy1D.map(log)).map(_.sum)

	def estimate_log_gaussian_prob(X:Array[Array[Double]], means:Array[Array[Double]], precisions_chol:Array[Array[Array[Double]]]):Array[Array[Double]] = {
		val n_features = X.head.length
		val Xo = mF.rows(X:_*)
		val log_det = compute_log_det_cholesky(precisions_chol)
		val log_prob = ListBuffer[Array[Double]]()
		for ((mu, prec_chol) <- means.map(x => mF.rows(x)) zip precisions_chol.map(x => mF.rows(x:_*))){
			val y1 = Xo multiply prec_chol
			val y2 = mu multiply prec_chol
			val y = y1.toRawCopy2D.map(mF.rows(_) subtract y2)
			val lp = y.map(_.toRawCopy1D.map(pow(_, 2))).map(_.sum)
			log_prob += lp map (_ + n_features * log(2 * Pi)) map (-0.5*_)
		}
		log_prob.toArray.transpose.map(_ zip log_det map (x => x._1 + x._2))
	}

	def estimate_log_weights(weights:Array[Double]):Array[Double] = weights map log


	def estimate_weighted_log_prob(X:Array[Array[Double]], means:Array[Array[Double]], weights:Array[Double], precisions_chol:Array[Array[Array[Double]]]):Array[Array[Double]]={
		val elp = estimate_log_prob(X, means, precisions_chol)
		val elw = estimate_log_weights(weights)
		elp map (_ zip elw map (x => x._1 + x._2))
	}

	def logsumexp(X:Array[Array[Double]]):Array[Double] = {
		val vmax = X map (_.max)
		(for (i <- X.indices)yield(log(X(i).map(x => exp(x - vmax(i))).sum) + vmax(i))).toArray
	}

	def check_X(X:Array[Array[Double]], n_components:Int)={if (n_components > 0 && X.length < n_components) throw new UnsupportedOperationException(s"Expected n_samples >= n_components but got n_components = $n_components, n_samples = ${X.length}")}

	def check_initial_parameters(n_components:Int, tol:Double, n_init:Int, max_iter:Int, reg_covar:Double)={
		if (n_components < 1) throw new UnsupportedOperationException(s"Invalid value for 'n_components': $n_components\nEstimation requires at least one component")
		if (tol < 0) throw new UnsupportedOperationException(s"Invalid value for 'tol': $tol\nTolerance used by the EM must be non-negative")
		if (n_init < 1) throw new UnsupportedOperationException(s"Invalid value for 'n_init': $n_init\nEstimation requires at least one run")
		if (max_iter < 1) throw new UnsupportedOperationException(s"Invalid value for 'max_iter': $max_iter \nEstimation requires at least one iteration")
		if (reg_covar < 0) throw new UnsupportedOperationException(s"Invalid value for 'reg_covar': $reg_covar\nRegularization on covariance must be non-negative")
	}

	def initialize_parameters(X:Array[Array[Double]], n_components:Int, init_params:String, reg_covar:Double)={
		val n_samples = X.length
		val resp = if (init_params == "kmeans") {
			val eye = Array.ofDim[Double](n_samples, n_components)
			val label = new KmeansLocal(n_clusters = n_components, random_state = random_state).fit(X).labels
			for (i <- 0 until n_samples) eye(i)(label(i)) = 1
			eye
		}else if (init_params == "random") {
			val eye = Array.fill[Double](n_samples, n_components)(Random.nextDouble)
			val rs = eye.map(_.sum)
			for(i <- 0 until n_samples; j <- 0 until n_components) eye(i)(j) /= rs(i)
			eye
		}else throw new IllegalArgumentException(s"Unimplemented initialization method $init_params")
		initialize(X, resp, reg_covar)
	}

	def initialize(X:Array[Array[Double]], resp:Array[Array[Double]], reg_covar:Double)={
		val n_samples = X.length
		val (weights, means, covariances) = estimate_gaussian_parameters(X, resp, reg_covar)
		wgt = weights.map(_/n_samples)
		mu = means
		cov = covariances
		pch = compute_precision_cholesky(covariances, reg_covar)
	}

	def estimate_gaussian_parameters(X:Array[Array[Double]], resp:Array[Array[Double]], reg_covar:Double):(Array[Double], Array[Array[Double]], Array[Array[Array[Double]]])={
		val nk = resp.transpose.map(_.sum).map(_ + eps)
		val means = (mF.rows(resp.transpose:_*) multiply mF.rows(X:_*)).toRawCopy2D
		for (i <- 0 until means.length; j <- means.head.indices) means(i)(j) /= nk(i)
		val covariances = estimate_gaussian_covariances_full(resp, X, nk, means, reg_covar)
		(nk, means, covariances)
	}

	def estimate_gaussian_covariances_full(resp:Array[Array[Double]], X:Array[Array[Double]], nk:Array[Double], means:Array[Array[Double]], reg_covar:Double):Array[Array[Array[Double]]]={
		val (n_components, n_features) = (means.length, means.head.length)
		val covariances = ListBuffer[Array[Array[Double]]]()
		val respT = resp.transpose
		val RC = mF.makeEye(n_features, n_features).multiply(reg_covar)
		for (k <- 0 until n_components){
//			val diff = X.map(_ zip means(k)).map(_.map(t=> t._1 - t._2))
			val diff = X.map(r => mF.rows(r) subtract mF.rows(means(k))).map(_.toRawCopy1D)
			val A = diff.transpose.map(r => for(i <- respT(k).indices) yield(r(i) * respT(k)(i))).map(_.toArray)
			val B = mF.rows(A:_*) multiply mF.rows(diff:_*) divide nk(k) add RC
			covariances += B.toRawCopy2D
		}
		covariances.toArray
	}

	def e_step(X:Array[Array[Double]]):(Double, Array[Array[Double]])={
		val (log_prob_norm, log_resp) = estimate_log_prob_resp(X)
		(log_prob_norm.sum/log_prob_norm.length, log_resp)
	}

	def estimate_log_prob_resp(X:Array[Array[Double]]):(Array[Double], Array[Array[Double]])={
		val weighted_log_prob = estimate_weighted_log_prob(X, means, weights, precisions_cholesky)
		val log_prob_norm = logsumexp(weighted_log_prob)
//		val log_resp = weighted_log_prob.transpose.map(r => for(i <- 0 until log_prob_norm.length)yield(r(i) - log_prob_norm(i))).map(_.toArray).transpose
		val log_resp = weighted_log_prob.transpose.map(r => mF.rows(r) subtract mF.rows(log_prob_norm)).map(_.toRawCopy1D).transpose
		(log_prob_norm, log_resp)
	}

	def m_step(X:Array[Array[Double]], log_resp:Array[Array[Double]], reg_covar:Double)={
		val n_samples = X.length
		val (weights, means, covariances) = estimate_gaussian_parameters(X, log_resp.map(_.map(exp)), reg_covar)
		wgt = weights.map(_/n_samples)
		mu = means
		cov = covariances
		pch = compute_precision_cholesky(covariances, reg_covar)
	}

	override def score_samples(X:Array[Array[Double]]) = super.score_samples(X)
	override def fit(X:Array[Array[Double]]) = super.fit(X)

	def debug(A:Array[Double]):Unit = println(A.deep.mkString(","))
	def debug(A:Array[Array[Double]]):Unit = println(A.deep.mkString("\n"))
	def debug(A:Array[Array[Array[Double]]]):Unit = println(A.deep.mkString("\n"))
}
