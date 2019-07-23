package gmm

import scala.util.Random

case class Dist(mu:Double, sigma:Double)

object Example {
	def main(args: Array[String]): Unit = {
		val rnd = new Random(42)
		val da = Dist(20, 5)
		val db = Dist(100, 10)
		val train = Array.range(0, 1000).map(x => Array(rnd.nextGaussian()*da.sigma + da.mu)) ++
				Array.range(0, 1000).map(x => Array(rnd.nextGaussian()*db.sigma + db.mu))
		val test = Array.range(0, 5).map(x => Array(rnd.nextGaussian()*da.sigma + da.mu)) ++
				Array.range(0, 5).map(x => Array(rnd.nextGaussian()*db.sigma + db.mu))
		val g = new GaussianMixture(16, 1)

		val gmm = g.train(train)
		val R = gmm.score_samples(test)
		R.foreach(println)

		val (_, log) = g.train_eval(train)
		println(log)
	}
}
