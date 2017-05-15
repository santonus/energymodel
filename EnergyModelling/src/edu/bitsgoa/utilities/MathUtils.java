package edu.bitsgoa.utilities;

/**
 * Various helpful math functions for use throughout the library
 */
public class MathUtils {

  /**
   * Calculate the covariance of two sets of data
   * 
   * @param x
   *          The first set of data
   * @param y
   *          The second set of data
   * @return The covariance of x and y
   */
  public static double covariance(double[] x, double[] y) {
    double xmean = mean(x);
    double ymean = mean(y);

    double result = 0;

    for (int i = 0; i < x.length; i++) {
      result += (x[i] - xmean) * (y[i] - ymean);
    }

    result /= x.length - 1;

    return result;
  }

  /**
   * Calculate the mean of a data set
   * 
   * @param data
   *          The data set to calculate the mean of
   * @return The mean of the data set
   */
  public static double mean(double[] data) {
    double sum = 0;

    for (int i = 0; i < data.length; i++) {
      sum += data[i];
    }

    return sum / data.length;
  }

  /**
   * Calculate the variance of a data set
   * 
   * @param data
   *          The data set to calculate the variance of
   * @return The variance of the data set
   */
  public static double variance(double[] data) {
    // Get the mean of the data set
    double mean = mean(data);

    double sumOfSquaredDeviations = 0;

    // Loop through the data set
    for (int i = 0; i < data.length; i++) {
      // sum the difference between the data element and the mean squared
      sumOfSquaredDeviations += Math.pow(data[i] - mean, 2);
    }

    // Divide the sum by the length of the data set - 1 to get our result
    return sumOfSquaredDeviations / (data.length - 1);
  }
}
