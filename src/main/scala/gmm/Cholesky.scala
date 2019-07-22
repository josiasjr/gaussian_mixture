package gmm

import org.nd4j.linalg.api.buffer.DataBuffer
import org.nd4j.linalg.api.buffer.util.DataTypeUtil
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j

//export OMP_NUM_THREADS=1

object Cholesky {
	DataTypeUtil.setDTypeForContext(DataBuffer.Type.DOUBLE)
//	NativeOpsHolder.getInstance().getDeviceNativeOps().setOmpNumThreads(1)
	def decompose(X: Array[Array[Double]]):Array[Array[Double]] = {
		val m = X.length
		val L = Array.ofDim[Double](m, m)
		for (i <- 0 until m) {
			for (j <- 0 until (i + 1)) {
				var sum = 0.0
				for (k <- 0 until j) {
					sum += L(i)(k) * L(j)(k)
				}
				L(i)(j) = if (i == j) Math.sqrt(X(i)(i)-sum) else 1.0/L(j)(j)*(X(i)(j)-sum)
			}
		}
		L
	}

	def decompose(X: INDArray):INDArray={
		val m = X.shape.head
		val L = Nd4j.zeros(m,m)
		for (i <- 0 until m) {
			for (j <- 0 until (i + 1)) {
				var sum = 0.0
				for (k <- 0 until j) {
					sum += L.getDouble(i,k) * L.getDouble(j,k)
				}
				L.putScalar(i, j, if (i == j) Math.sqrt(X.getDouble(i,i)-sum) else 1.0/L.getDouble(j,j)*(X.getDouble(i,j)-sum))
			}
		}
		L
	}
}
