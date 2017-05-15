package edu.bitsgoa.programAnalyzer.instructiontypes;

import edu.bitsgoa.driver.ConfigureModel;

public class DataComputeInst {

	private EnumComputeInsts instType;
	private String instName;
	private int latency;
	private double throughput;
	private int peakWarps;
	private double delay;
	
	public double getCalculatedDelay(double parallelism) {
				
		if (parallelism < peakWarps)
			delay = 1.0 * latency / parallelism;
		else {
			delay = 1.0 * latency / (parallelism * peakWarps)
					+ ConfigureModel.warpSize / throughput;
		}
		//System.out.println("Latency is: "+latency);
		//System.out.println("Parallelism is: "+parallelism);
		//System.out.println("Peak warps is: "+peakWarps);
		//System.out.println("Warp size is: "+ConfigureModel.warpSize);
		//System.out.println("Throughput is: "+throughput);
		//System.out.println("Instruction:  "+instName +"   Latency:   "+latency+"  PeakWarps : "+peakWarps+"  Delay: "+delay);
		return delay;
		
	}
	
	public void setData(EnumComputeInsts instType, String instName, int latency, double throughput, int peakWarps, double delay) {
		this.instType = instType;
		this.instName = instName;
		this.latency = latency;
		//System.out.println("throughput:  "+throughput +"   Peak Warps:   "+peakWarps);
		this.throughput = throughput;
		this.peakWarps = peakWarps;
		this.delay = delay;
		//System.out.println("Instruction:  "+instName +"   Latency:   "+latency+"  throughput: "+throughput+"  PeakWarps : "+peakWarps);
	}
	
	public EnumComputeInsts getInstType() {
		return instType;
	}
	
	public String getInstName() {
		return instName;
	}
	
	public int getLatency() {
		return latency;
	}
	
	public double getThroughput() {
		return throughput;
	}
	
	public int getPeakWarps() {
		return peakWarps;
	}
	
	public DataComputeInst() {
		instType = null;
		instName = "";
		latency = 0;
		throughput = 0.0;
		peakWarps = 0;
		reset();
	}
	
	public void reset() {
		delay = 0.0;
	}
	
}