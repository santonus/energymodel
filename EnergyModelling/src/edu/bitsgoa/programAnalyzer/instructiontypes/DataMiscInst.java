package edu.bitsgoa.programAnalyzer.instructiontypes;


public class DataMiscInst {

	private EnumMiscInsts instType;
	private String instName;
	private int latency;
	private double throughput;
	private int peakWarps;
	private double delay;

	public double getCalculatedDelay(double parallelism) {
		
		double penalty = 0.0;
		
		if (parallelism < peakWarps) {
			delay = 1.0 * latency / parallelism;
		} else {
			delay = 1.0 * latency / parallelism + penalty;
		}
		//System.out.println("Instruction:  "+instName +"   Latency:   "+latency+"  PeakWarps : "+peakWarps+"  Delay: "+delay);
		return delay;
	}
	
	public void setData(EnumMiscInsts instType, String instName, int latency, double throughput, int peakWarps, double delay) {
		this.instType = instType;
		this.instName = instName;
		this.latency = latency;
		this.throughput = throughput;
		this.peakWarps = peakWarps;
		this.delay = delay;
		//System.out.println("Instruction:  "+instName +"   Latency:   "+latency+"  throughput: "+throughput+"  PeakWarps : "+peakWarps);
	}
	
	public EnumMiscInsts getInstType() {
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
	
	public DataMiscInst() {
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