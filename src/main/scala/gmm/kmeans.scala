/*
-Dcom.github.fommil.netlib.NativeSystemBLAS.natives=/usr/lib/x86_64-linux-gnu/blas/libblas.so.3
-Dcom.github.fommil.netlib.NativeSystemARPACK.natives=/usr/lib/x86_64-linux-gnu/libarpack.so.2
-Dcom.github.fommil.netlib.NativeSystemLAPACK.natives=/usr/lib/x86_64-linux-gnu/lapack/liblapack.so.3
*/
package gmm

import scala.util.Random
import weka.core.{Attribute, DenseInstance, EuclideanDistance, Instances}
import weka.clusterers.SimpleKMeans
import scala.collection.JavaConverters._
import smile.clustering.KMeans


abstract class KmeansLocal(n_clusters:Int=2, n_init:Int=10, max_iter:Int=300, random_state:Option[Int]=None){
//	var cluster_centers:Array[Array[Double]] = _
	var labels:Array[Int] = _
	def fit(X:Array[Array[Double]]):KmeansLocal
}


class KmeansWeka(n_clusters:Int=2, n_init:Int=1, max_iter:Int=300, random_state:Option[Int]=None, n_jobs:Int=1, init:Int=SimpleKMeans.KMEANS_PLUS_PLUS)
	extends KmeansLocal(n_clusters, n_init, max_iter, random_state) {
	private val distfun = new EuclideanDistance
	distfun.setOptions("-D -R, first-last".split(" "))
	private val seed = if (random_state.isDefined) new Random(random_state.get) else new Random
	private def opt = s"-N $n_clusters -I $max_iter -init $init -S ${seed.nextInt} -num-slots $n_jobs".split(" ")
	private val eps = math.ulp(0.5)

	private def createWekaFrame(data:Array[Array[Double]]):Instances={//, target:Boolean=false)={
		val X = if (init == SimpleKMeans.KMEANS_PLUS_PLUS) data.map(_.map(_ + Random.nextDouble*eps)) else data
		val atts = new java.util.ArrayList(X.head.indices.map(x => new Attribute(x.toString)).asJava)
		val wf = new Instances("WekaFrame", atts, 0)
		X.foreach(x => wf.add(new DenseInstance(1.0, x)))
		wf
	}

	def fit(X:Array[Array[Double]]):KmeansLocal={
		val wf = createWekaFrame(X)
		val km = Range(0, n_init).map{_ =>
			val km = new SimpleKMeans
			km.setOptions(opt)
			km.setDistanceFunction(distfun)
			km.buildClusterer(wf)
			(km.getSquaredError, km)
		}.minBy(_._1)._2
//		cluster_centers = km.getClusterCentroids.asScala.map(_.toDoubleArray).toArray
		labels = wf.asScala.map(km.clusterInstance).toArray
		this
	}
}


class KmeansSmile(n_clusters:Int=2, n_init:Int=10, max_iter:Int=300, random_state:Option[Int]=None)
	extends KmeansLocal(n_clusters, n_init, max_iter, random_state) {
	def fit(X: Array[Array[Double]]):KmeansLocal = {
		val km = new KMeans(X, n_clusters, max_iter, n_init)
//		cluster_centers = km.centroids()
		labels = X.map(km.predict)
		this
	}
}
