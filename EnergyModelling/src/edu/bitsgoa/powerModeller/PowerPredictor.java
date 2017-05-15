package edu.bitsgoa.powerModeller;

public class PowerPredictor {

	double powerWatts;
	public static int minPower = 48;
	
	public PowerPredictor() {
		powerWatts = 0.0;
	}
	
	public double getCalculatedPower(double achievedOccupancy) {
		powerWatts = 0;
		//System.out.println(achievedOccupancy);
		double exponent = achievedOccupancy + 3.04115 * achievedOccupancy;
		powerWatts = Math.exp(exponent) + minPower;
		return powerWatts;
	}
	
}