package edu.bitsgoa.driver;

import edu.bitsgoa.properties.ParametersValue;

public final class ConfigureModel {
	// ms
	public static double cudaContextInitializationOverhead, kernelLaunchOverhead;
	// MBps
	public static long transferBandwidthPeak;
	// const
	public static int eqCoeff1, eqConst, transferSize;
	// kernel
	public static int numBlocks, numThreadsPerBlock, numThreadsLaunched;
	public static double achievedOccupancy;
	//CUDA occupancy calculator
	//	public static int maxTh, maxW, maxBl;
	//	public static double maxOcc;
	public static int accessFactorBadCoalescing, numBankConflicts;
	// GPU constants
	public static int warpSize, numBanks;
	public static int globalMemLineSize;
	public static int sharedBytesTransferred;
	public static double globalMemBandwidth, sharedMemBandwidth;
	public static long gpuClock; // MHz
	public static int maxActiveWarpsPerSM;
	public static int numSMs, numCoresPerSM, numTotalCores,maxThreadPerSm;
	// configure
	public static boolean testingOn;
	public static int numLoopIterations, maxNumberOfKernels, maxRegistersPerInstruction, numIndependentInsts;
	public static double branchProbability;
	public static String mainPath, benchmarkName, outputFileName;
	public static String inputProgPath, inputProgCUDA, inputProgPTX;
	public static float finalMaxBl;

	public static void getPropValues() {
		initialize();

		try {
			transferBandwidthPeak =(long) Float.parseFloat(ParametersValue.transferBWPeak);
			maxThreadPerSm= Integer.parseInt(ParametersValue.maxNoThreadPerSM);
			gpuClock = Integer.parseInt(ParametersValue.GPUclock );
			warpSize = Integer.parseInt(ParametersValue.warpsize);
			numSMs = Integer.parseInt(ParametersValue.noSM);
			globalMemLineSize = Integer.parseInt(ParametersValue.globalMemLineSize);
			globalMemBandwidth = Double.parseDouble(ParametersValue.globalMemBW);
			numBanks = Integer.parseInt(ParametersValue.noBanks);
			numCoresPerSM = Integer.parseInt(ParametersValue.noCoresPerSM);
			numTotalCores = numSMs * numCoresPerSM;
			sharedMemBandwidth = 4.0 * numBanks * numSMs * gpuClock/1.0;
			transferSize = Integer.parseInt(ParametersValue.transferSize);
			numBlocks = Integer.parseInt(ParametersValue.noBlocks);
			numThreadsPerBlock = Integer.parseInt(ParametersValue.noThreadsPerBlock);	
			numLoopIterations = Integer.parseInt(ParametersValue.noOfLoopIterations);
			numThreadsLaunched = numBlocks * numThreadsPerBlock;
			branchProbability = Float.parseFloat(ParametersValue.branchProb);
			maxNumberOfKernels = Integer.parseInt(ParametersValue.maxNoofKernels);
			maxRegistersPerInstruction = Integer.parseInt(ParametersValue.maxRegPerInst);
			numIndependentInsts = Integer.parseInt(ParametersValue.noIndptInst);
			maxActiveWarpsPerSM = Integer.parseInt(ParametersValue.maxActvWarpPerInst);
			accessFactorBadCoalescing = Integer.parseInt(ParametersValue.accFactorBadCoal);
			numBankConflicts = Integer.parseInt(ParametersValue.noOfBankConf);
			sharedBytesTransferred = Integer.parseInt(ParametersValue.sharedBytesTrans);
			inputProgPath = EnergyEstimator.pathToCurrentFile.substring(0,EnergyEstimator.pathToCurrentFile.lastIndexOf('/'))+"/";
			inputProgCUDA = EnergyEstimator.fileName;
			inputProgPTX = EnergyEstimator.fileName.substring(0,EnergyEstimator.fileName.lastIndexOf('.'))+".ptx";
			achievedOccupancy = 0.5;
			outputFileName="Results.txt";
			
			int totalThreads=ConfigureModel.numBlocks*ConfigureModel.numThreadsPerBlock;
			int intMaxBl=totalThreads/(ConfigureModel.maxThreadPerSm*ConfigureModel.numSMs);
			finalMaxBl=(int) Math.ceil((intMaxBl/ConfigureModel.numSMs));
			if(finalMaxBl==0.0)
				finalMaxBl++;


		} catch (Exception e) {
			e.printStackTrace();
		}

	}

	public static void initialize() {
		cudaContextInitializationOverhead = 0.0;
		kernelLaunchOverhead = 0.0;
		transferBandwidthPeak = 0;
		gpuClock = 0;
		eqCoeff1 = eqConst = 0;
		testingOn = false;
		sharedBytesTransferred = 0;
		warpSize = 0; maxActiveWarpsPerSM = 0;
		achievedOccupancy = 0.0;
		numBlocks = numThreadsPerBlock = numThreadsLaunched = 0;
		globalMemLineSize = 0;
		globalMemBandwidth = 0.0; sharedMemBandwidth = 0.0;
		numBanks = 0; numIndependentInsts = 0;
		numLoopIterations = 0; maxNumberOfKernels = 0; maxRegistersPerInstruction = 0;
		branchProbability = 0.0; numSMs = 0;
		mainPath = benchmarkName = outputFileName = "";
		inputProgPath = inputProgCUDA = inputProgPTX = "";
		numCoresPerSM = numTotalCores = 0;
		accessFactorBadCoalescing = numBankConflicts = 0;
	}

}
