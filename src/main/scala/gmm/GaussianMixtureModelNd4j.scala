//-Ddtype=double
//-Dcom.github.fommil.netlib.NativeSystemARPACK.natives=/usr/lib/x86_64-linux-gnu/libarpack.so.2
package gmm

import org.nd4j.linalg.api.buffer.DataBuffer
import org.nd4j.linalg.api.buffer.util.DataTypeUtil
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.inverse.InvertMatrix
import org.nd4j.linalg.ops.transforms.Transforms
import scala.collection.immutable.IndexedSeq
import scala.language.implicitConversions
import scala.math.{Pi, log}

//ND4J_SKIP_BLAS_THREADS=0
//OMP_NUM_THREADS=1
class GaussianMixtureModelNd4j(n_components:Int=2, tol:Double=0.001, reg_covar:Double=1e-6, max_iter:Int=300, n_init:Int=1, init_params:String="random",
	weights_init:Array[Double]=null, means_init:Array[Array[Double]]=null, precisions_init:Array[Array[Array[Double]]]=null, random_state:Option[Int]=None, verbose:Int=0, verbose_interval:Int=10)
	extends GaussianMixtureBase[INDArray,INDArray,IndexedSeq[INDArray]](n_components, tol, reg_covar, max_iter, n_init, init_params, weights_init, means_init, precisions_init, random_state, verbose, verbose_interval)
	with SelectKmeansType {
	Nd4j.zeros(1)
	DataTypeUtil.setDTypeForContext(DataBuffer.Type.DOUBLE)
	private val eps = math.ulp(0.5)
	var wgt:INDArray = if (weights_init == null) null else Nd4j.create(weights_init)
	var mu:INDArray = if (means_init == null) null else Nd4j.create(means_init)
	var cov:IndexedSeq[INDArray] = _
	var pch:IndexedSeq[INDArray] = if (precisions_init == null) null else precisions_init.map(Nd4j.create).toIndexedSeq
	def tensor2vector(X:INDArray):Array[Double] = X.data().asDouble()
	def tensor2matrix(X:INDArray):Array[Array[Double]] = X.data().asDouble().grouped(X.shape.last).toArray
	def tensor2cube(X:IndexedSeq[INDArray]):Array[Array[Array[Double]]] = X.map(tensor2matrix).toArray
	//def tensor2cube(X:INDArray):Array[Array[Array[Double]]] = X.data().asDouble().grouped(X.shape()(2)).toArray.grouped(X.shape()(1)).toArray

	def check_X(X:INDArray, n_components:Int)={if (n_components > 0 && X.shape.head < n_components) throw new UnsupportedOperationException(s"Expected n_samples >= n_components but got n_components = $n_components, n_samples = ${X.shape.head}")}

	def initialize_parameters(X:INDArray, n_components:Int, init_params:String, reg_covar:Double)={
		val n_samples = X.shape.head
		val resp = if (init_params == "kmeans") {
			val eye = Nd4j.zeros(n_samples, n_components)
			val label = new KmeansLocal(n_clusters = n_components, random_state = random_state).fit(tensor2matrix(X)).labels
			for (i <- 0 until n_samples) eye.putScalar(i, label(i), 1)
			eye
		}else if (init_params == "random") {
			val eye = Nd4j.rand(n_samples, n_components)
			val rs = eye.sum(1)
			for(i <- 0 until n_samples; j <- 0 until n_components) eye.putScalar(i,j, eye.getDouble(i,j)/rs.getDouble(i))
			eye
		}else throw new IllegalArgumentException(s"Unimplemented initialization method $init_params")
		initialize(X, resp, reg_covar)
	}

	def initialize(X:INDArray, resp:INDArray, reg_covar:Double)={
		val n_samples = X.shape.head
		val (weights, means, covariances) = estimate_gaussian_parameters(X, resp, reg_covar)
		wgt = weights divi n_samples
		mu = means
		cov = covariances
		pch = compute_precision_cholesky(covariances, reg_covar)
	}

	def estimate_gaussian_parameters(X:INDArray, resp:INDArray, reg_covar:Double):(INDArray,INDArray,IndexedSeq[INDArray])={
		val nk = resp.sum(0) addi eps
		val means = (resp.transpose mmul X) divColumnVector nk.transpose
		val covariances = estimate_gaussian_covariances_full(resp, X, nk, means, reg_covar)
		(nk, means, covariances)
	}

	def estimate_gaussian_covariances_full(resp:INDArray, X:INDArray, nk:INDArray, means:INDArray, reg_covar:Double):IndexedSeq[INDArray]={
		val Array(n_components, n_features) = means.shape
		val RC = Nd4j.eye(n_features) muli reg_covar
		for (k <- 0 until n_components) yield{
			val diff = X subRowVector means.getRow(k)
			((diff mulColumnVector resp.getColumn(k)).transposei mmul diff) divi nk.getDouble(k) addi RC
		}
	}

	def compute_precision_cholesky(covariances:IndexedSeq[INDArray], reg_covar:Double):IndexedSeq[INDArray]={
		for (covariance <- covariances) yield{
			val cov_chol = Cholesky.decompose(covariance)
			if (cov_chol.shape.head > 1)InvertMatrix.invert(cov_chol, true).transposei.dup else Nd4j.create(Array(Array(1/cov_chol.getDouble(0))))
		}
	}

	def e_step(X:INDArray):(Double, INDArray)={
		val (log_prob_norm, log_resp) = estimate_log_prob_resp(X)
		(log_prob_norm.meanNumber.doubleValue, log_resp)
	}

	def estimate_log_prob_resp(X:INDArray):(INDArray, INDArray)={
		val weighted_log_prob = estimate_weighted_log_prob(X, mu, wgt, pch)
		val log_prob_norm = logsumexp(weighted_log_prob)
		val log_resp = weighted_log_prob subiColumnVector log_prob_norm
		(log_prob_norm, log_resp)
	}

	def estimate_weighted_log_prob(X:INDArray, means:INDArray, weights:INDArray, precisions_cholesky:IndexedSeq[INDArray]):INDArray={
		val elp = estimate_log_prob(X, means, precisions_cholesky)
		val elw = estimate_log_weights(weights)
		elp addiRowVector elw
	}

	def estimate_log_gaussian_prob(X:INDArray, means:INDArray, precisions_cholesky:IndexedSeq[INDArray]):INDArray = {
		val n_features = X.shape.last
		val log_det = compute_log_det_cholesky(precisions_cholesky)
		val log_prob = for (k <- precisions_cholesky.indices) yield{
			val Y = (X mmul precisions_cholesky(k)) subiRowVector (means.getRow(k) mmul precisions_cholesky(k))
			Transforms.pow(Y, 2, true).sum(1)
		}
		val lp = Nd4j.hstack(log_prob:_*)
		lp addi n_features*log(2*Pi) muli -0.5 addiRowVector log_det
	}

	def compute_log_det_cholesky(precisions_cholesky: IndexedSeq[INDArray]):INDArray = {
		val dets = precisions_cholesky map Nd4j.diag map(x => Transforms.log(x, true)) map(_.sumNumber.doubleValue)
		Nd4j.create(dets.toArray)
	}

	def estimate_log_weights(weights:INDArray):INDArray = Transforms.log(weights)

	def logsumexp(X:INDArray):INDArray = {
		val vmax = X.max(1)
		Transforms.log(Transforms.exp(X subColumnVector vmax, true).sum(1), true) addi vmax
	}

	def m_step(X:INDArray, log_resp:INDArray, reg_covar:Double)={
		val n_samples = X.shape.head
		val (weights, means, covariances) = estimate_gaussian_parameters(X, Transforms.exp(log_resp), reg_covar)
		wgt = weights divi n_samples
		mu = means
		cov = covariances
		pch = compute_precision_cholesky(covariances, reg_covar)
	}

	def score_samples(X:Array[Array[Double]]):Array[Double] = tensor2vector(super.score_samples(Nd4j.create(X)))

//	def score_samples_normalized(X:INDArray):Array[Double] = tensor2vector(Transforms.log(Transforms.abs(super.score_samples(X))))

	def score(X:INDArray):Double = score_samples(X).meanNumber.doubleValue

	def bic(X:INDArray):Double = -2 * score(X) * X.shape.head + n_parameters(n_components, mu.shape.last) * log(X.shape.last)

//	def bic(X:Array[Array[Double]]):Double = -2 * score(Nd4j.create(X)) * X.length + n_parameters(n_components, mu.shape.last) * log(X.length)

	def fit(X:Array[Array[Double]]):GaussianMixtureModelNd4j = {
		fit(Nd4j.create(X))
		this
	}

	def debug(A:INDArray):Unit= {
		if (A.rank() == 1) println(tensor2vector(A).deep.mkString(","))
		else println(tensor2matrix(A).deep.mkString("\n"))
	}

	def debug(A:IndexedSeq[INDArray]):Unit = println(tensor2cube(A).deep.mkString("\n"))

}
