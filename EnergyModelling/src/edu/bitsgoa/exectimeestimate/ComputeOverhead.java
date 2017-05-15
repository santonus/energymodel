package edu.bitsgoa.exectimeestimate;

import edu.bitsgoa.driver.ConfigureModel;
import edu.bitsgoa.utilities.UtilsMisc;

public class ComputeOverhead {
	
	// const
	private double cudaContextInitializationOverhead;
	private double kernelLaunchOverhead;
	private long bandwidthPeak;
	private int eqCoeff1, eqConst;
	
	// variable
	@SuppressWarnings("unused") private int transferSize; // bytes
	
	// calculated
	private long bandwidthIncSize;
	private double transferOverhead;	
	private double totalOverhead;
	
	public void calculate(int transferSize) {
		initialize();
		bandwidthIncSize = (eqCoeff1 * transferSize + eqConst);
		long denom = UtilsMisc.minLong(bandwidthPeak, bandwidthIncSize);
		transferOverhead = 1.0 * transferSize/denom; // micros to ms
		totalOverhead = transferOverhead + cudaContextInitializationOverhead + kernelLaunchOverhead;
		if (ConfigureModel.testingOn) show(transferSize, denom);
	}
	
	private void show(int transferSize, long denom) {
		String sizeStr = "";
		if (transferSize < 1024)
			sizeStr = transferSize + " B";
		else if (transferSize >= 1024 && transferSize < 1024 * 1024)
			sizeStr = transferSize/1024 + " KB";
		else if (transferSize >= 1024 * 1024)
			sizeStr = transferSize/(1024 * 1024) + " MB";
		System.out.println(sizeStr + "\t" + String.format("%.5f", totalOverhead) + " ms \t\t" + 
				transferOverhead + "\t" + denom + "\t = min of: " + bandwidthPeak + " " + bandwidthIncSize);
	}
	
	private void initialize() {
		cudaContextInitializationOverhead = ConfigureModel.cudaContextInitializationOverhead;
		kernelLaunchOverhead = ConfigureModel.kernelLaunchOverhead ;// (/1000.0 in Kajal's Model);
		bandwidthPeak = ConfigureModel.transferBandwidthPeak;
		eqCoeff1 = ConfigureModel.eqCoeff1;
		eqConst = ConfigureModel.eqConst;
	}
	
	public ComputeOverhead() {
		cudaContextInitializationOverhead = 0.0;
		kernelLaunchOverhead = 0.0;
		bandwidthPeak = 0;
		eqCoeff1 = 0;
		eqConst = 0;
		bandwidthIncSize = 0;
		transferOverhead = 0.0;
		totalOverhead = 0.0;
	}
	
	public double getTotalOverhead() {
		return totalOverhead;
	}
	
}