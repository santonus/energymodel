package edu.bitsgoa.programAnalyzer.instructiontypes;

import edu.bitsgoa.driver.ConfigureModel;

public class DataMemoryInst {

	private EnumMemoryInsts instType;
	private String instName;
	private double latency;
	private int peakWarps;
	private double delay;
	
	public double getCalculatedDelay(double parallelism, EnumMemoryInsts memAccessType, double accessFactorBadCoalescing, int numBankConflicts) {
		
		double penalty = 0.0;
		
		if (parallelism < peakWarps)
			penalty = 0.0;
		else if (memAccessType == EnumMemoryInsts.GlobalLoad || memAccessType == EnumMemoryInsts.GlobalStore) {
			penalty = ( ConfigureModel.globalMemLineSize * accessFactorBadCoalescing ) / ConfigureModel.globalMemBandwidth;
		}
		else if (memAccessType == EnumMemoryInsts.SharedLoad || memAccessType == EnumMemoryInsts.SharedStore) {
			penalty =( (ConfigureModel.warpSize * ConfigureModel.sharedBytesTransferred )/
				(ConfigureModel.numBanks * ConfigureModel.sharedMemBandwidth))
				+ (numBankConflicts * ConfigureModel.globalMemLineSize) / ConfigureModel.globalMemBandwidth;
		}
		else {
			penalty = ConfigureModel.globalMemLineSize * accessFactorBadCoalescing / ConfigureModel.globalMemBandwidth;
		}
		
		delay = 1.0 * latency / parallelism + penalty;
		System.out.println("Latency: "+latency);
		System.out.println("Parallelism: "+parallelism);
		System.out.println("Penalty: "+penalty);
		System.out.println("Instruction:  "+instName +"   Latency:   "+latency+"  PeakWarps : "+peakWarps+"  Delay: "+delay);
		System.out.println(delay+"     .......................................................................................");
		return delay;
	}
	
	public void setData(EnumMemoryInsts instType, String instName, double latency, int peakWarps, double delay) {
		this.instType = instType;
		this.instName = instName;
		this.latency = latency;
		this.peakWarps = peakWarps;
		this.delay = delay;
		
	}
	
	public EnumMemoryInsts getInstType() {
		return instType;
	}
	
	public String getInstName() {
		return instName;
	}
	
	public double getLatency() {
		return latency;
	}
	
	public int getPeakWarps() {
		return peakWarps;
	}
	
	public DataMemoryInst() {
		instType = null;
		instName = "";
		latency = 0.0;
		peakWarps = 0;
		reset();
	}
	
	public void reset() {
		delay = 0.0;
	}
	
}