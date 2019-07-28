# A Gaussian Mixture Model port from scikit-learn to Scala 

This project uses an anomaly detection system to determine the threshold of desirable individual's behavior. It consists of two parts:
- The first part is the offline stage that provides a Spark job to generate individual gaussian mixture models for each individual.
- The second part is the online application that load the individual model from a NoSQL and gives the probability of the action.  







# For more information
The Gaussian mixture models implementation is a port from:
https://scikit-learn.org/stable/modules/generated/sklearn.mixture.GaussianMixture.html
