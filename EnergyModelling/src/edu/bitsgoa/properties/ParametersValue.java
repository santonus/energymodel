package edu.bitsgoa.properties;
/**
 * This class is used to save the values of text-fields in Parameters.java. We just assign the values in the text-boxes to the static fields
 * of this class
 *
 */
public class ParametersValue {
	public static int timesused=0;	
	public static String maxNoThreadPerSM;	//max no of threads per SM
	public static String noSM;	//no of SMs present in the card
	public static String warpsize;	//warp size
	public static String noBanks;	//no of banks
	public static String devTohostBW;	//dev. to host bw
	public static String hostToDevBW;	//host to dev. bw
	public static String globalMemBW;	//global mem. bw
	public static String noCoresPerSM;	//no of cores per sm
	public static String globalMemLineSize;	//global mem. line size
	public static String noThreadsPerBlock;	//no. of threads per block
	public static String noBlocks;	//no. of blocks
	public static String noOfLoopIterations;	//no. of loop iterations
	public static String GPUclock;	//freq. of the processor
	public static String transferBWPeak;		//transfer bw. peak
	public static String transferSize;	//transfer size
	public static String branchProb;	//branch probablity
	public static String maxNoofKernels;	//max. no. of kernels
	public static String maxRegPerInst;	//max. no. of reg. per inst.
	public static String noIndptInst;	//no. of independent inst.
	public static String maxActvWarpPerInst;	//max. active warps per inst.
	public static String accFactorBadCoal;	//access factor bad coales.
	public static String noOfBankConf;	//no. of bank conflicts
	public static String sharedBytesTrans;	//shared bytes transferred	
	public static String version;
}
