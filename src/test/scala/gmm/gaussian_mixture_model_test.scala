package gmm

import org.scalatest.FunSuite
//import org.junit.runner.RunWith
//import org.scalatest.junit.JUnitRunner


//@RunWith(classOf[JUnitRunner])
class GaussianMixtureModelSuite extends FunSuite {
	test("Testing check_X:"){
		val gmm = new GaussianMixtureModel
		assert(gmm.check_X(Array.ofDim[Double](2,2), 4) == false)
		assert(gmm.check_X(Array.ofDim[Double](2,5), 4) == false)
		assert(gmm.check_X(Array.ofDim[Double](5,2), 4) == true)
		assert(gmm.check_X(Array.ofDim[Double](10,10), 4) == true)
	}

	test("Testing check_initial_parameters:"){
		assert((new GaussianMixtureModel(1, -0.1)).check_initial_parameters == false)
		assert((new GaussianMixtureModel(1, 0.1)).check_initial_parameters == true)

		assert((new GaussianMixtureModel(1, 0.1, -0.1)).check_initial_parameters == false)
		assert((new GaussianMixtureModel(1, 0.1, 0.1)).check_initial_parameters == true)

		assert((new GaussianMixtureModel(1, 0.1, 0.1, 0)).check_initial_parameters == false)
		assert((new GaussianMixtureModel(1, 0.1, 0.1, 10)).check_initial_parameters == true)

		assert((new GaussianMixtureModel(1, 0.1, 0.1, 10, 0)).check_initial_parameters == false)
		assert((new GaussianMixtureModel(1, 0.1, 0.1, 10, 10)).check_initial_parameters == true)
	}

	test("Testing fit:"){
		var X = Array.ofDim[Double](2,2)
		var gmm = new GaussianMixtureModel(4)
		assert(gmm.fit(X) == false)
		X = Array.ofDim[Double](5,10)
		gmm = new GaussianMixtureModel(4, -0.1)
		assert(gmm.fit(X) == false)
		X = Array.ofDim[Double](10,10)
		gmm = new GaussianMixtureModel(4, 0.1, 1e-6, 10, 1)
		assert(gmm.fit(X) == true)
	}
}

object GaussianMixtureModelSuite{
	def main(args: Array[String]): Unit = {
		(new GaussianMixtureModelSuite).execute()
	}
}
