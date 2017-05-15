package edu.bitsgoa.exectimeestimate;

import edu.bitsgoa.driver.ConfigureModel;
import edu.bitsgoa.programAnalyzer.PTXUtil;
import edu.bitsgoa.programAnalyzer.instructiontypes.EnumAllInsts;
import edu.bitsgoa.programAnalyzer.instructiontypes.EnumComputeInsts;
import edu.bitsgoa.programAnalyzer.instructiontypes.EnumMemoryInsts;
import edu.bitsgoa.programAnalyzer.instructiontypes.EnumMiscInsts;
import edu.bitsgoa.programAnalyzer.singlekernel.SingleKernel;
import edu.bitsgoa.programAnalyzer.singlekernel.basicblock.BasicBlock;
import edu.bitsgoa.utilities.UtilsMisc;

public class ComputeExecTimeModified {

	private double executionTimePerThread;
	private double executionTimeTotal;
	private int[] numBlocksExecuted;

	@SuppressWarnings("unused")
	private double execTimePerThreadAndReturnDelay(SingleKernel kernel, int bl, int tpbl) {
		BasicBlock curBlock = null;
		String curInst = null;
		int numBasicBlocks = kernel.getNumBasicBlocks();
		int numInstsInBlock = 0, maxComputeTypes = 0, maxMemTypes = 0, maxMiscTypes = 0, index = 0, numBankConflicts = 0;;
		double cumulativeDelay = 0.0, curBlockDelay = 0.0, curDelay = 0.0, accessFactorBadCoalescing = 0.0;
		EnumAllInsts instType = null; EnumComputeInsts computeType = null; EnumMemoryInsts memType = null; EnumMiscInsts miscType = null;

		int blocks = bl;
		int threadsPerBlock = tpbl;
		int totalThreads = threadsPerBlock * blocks;
		int totalWarps = totalThreads / ConfigureModel.warpSize;
		int maxBlSM =(int) ConfigureModel.finalMaxBl;
		double warpsPerBlock = 1.0 * threadsPerBlock / ConfigureModel.warpSize; // if 16/32

		accessFactorBadCoalescing = ConfigureModel.accessFactorBadCoalescing;
		numBankConflicts = ConfigureModel.numBankConflicts;

		for (int i = 0; i < numBasicBlocks; i++) {
			curBlock = kernel.getBasicBlockFromList(i);
			curBlockDelay = 0.0;
			numInstsInBlock = (int)curBlock.getAllInstructionsCount();
			maxComputeTypes = curBlock.computeDetails.length;
			maxMemTypes = curBlock.memoryDetails.length;
			maxMiscTypes = curBlock.miscDetails.length;

			for (int j = 0; j < numInstsInBlock; j++) {
				curInst = curBlock.getInstructionNumber(j);
				instType = PTXUtil.decodeInstruction(curInst);

				if (instType == EnumAllInsts.Computation) {
					computeType = PTXUtil.decodeComputationType(curInst);
					index = 0;
					for (int k = 0; k < maxComputeTypes; k++) {
						if (computeType == curBlock.computeDetails[k].getInstType()) {
							index = k;
							break;
						}
					}
					curBlock.calculateParallelism(tpbl, bl);
					curDelay = curBlock.computeDetails[index].getCalculatedDelay(curBlock.getParallelism());
					curBlockDelay += curDelay;
				}

				else if (instType == EnumAllInsts.MemAccess) {
					memType = PTXUtil.decodeMemoryType(curInst);
					index = 0;
					for (int k = 0; k < maxMemTypes; k++) {
						if (memType == curBlock.memoryDetails[k].getInstType()) {
							index = k;
							break;
						}
					}
					curBlock.calculateParallelism(tpbl, bl);
					curDelay = curBlock.memoryDetails[index].getCalculatedDelay(curBlock.getParallelism(), memType, accessFactorBadCoalescing, numBankConflicts);
					curBlockDelay += curDelay;
				}

				else if (instType == EnumAllInsts.Unknown)
					;

				else {
					miscType = PTXUtil.decodeMiscInstType(curInst);
					index = 0;
					for (int k = 0; k < maxMiscTypes; k++) {
						if (miscType == curBlock.miscDetails[k].getInstType()) {
							index = k;
							break;
						}
					}
					curBlock.calculateParallelism(tpbl, bl);
					curDelay = curBlock.miscDetails[index].getCalculatedDelay(curBlock.getParallelism());
					curBlockDelay += curDelay;
				}
			}

			if (curBlock.hasLoop())
				curBlockDelay *= ConfigureModel.numLoopIterations;
			cumulativeDelay += curBlockDelay;
		}

		executionTimePerThread = cumulativeDelay / ConfigureModel.gpuClock;

		int numWarpsIssuingInstsPerCyclePerCore = ConfigureModel.numCoresPerSM / ConfigureModel.warpSize;
		int numIssueCycles = (int)(warpsPerBlock / numWarpsIssuingInstsPerCyclePerCore);
		if (warpsPerBlock % numWarpsIssuingInstsPerCyclePerCore != 0)
			numIssueCycles = numIssueCycles + 1;

		executionTimePerThread *= numIssueCycles;
		return cumulativeDelay;
	}

	@SuppressWarnings("unused")
	public void calculateExecutionTime(SingleKernel kernel) {

		int blocks = kernel.getKernelNumBlocks();
		int threadsPerBlock = kernel.getKernelNumThreadsPerBlock();
		int totalThreads = threadsPerBlock * blocks;
		int totalWarps = totalThreads / ConfigureModel.warpSize;
		int maxBlSM = (int)ConfigureModel.finalMaxBl;
		double warpsPerBlock = 1.0 * threadsPerBlock / ConfigureModel.warpSize; // if 16/32
		double maxWarpsSM = maxBlSM * warpsPerBlock;
		double warpsPerSM = totalWarps/ConfigureModel.numSMs;
		double numRounds = warpsPerSM / maxWarpsSM;

		int numBlThisRound = 0, numWavesRounds = 0;
		int numBlOverall = 0;
		int numBlRemaining = ConfigureModel.numBlocks;
		int curSM = 0;

		numBlThisRound = numWavesRounds = 0;
		numBlOverall = 0;
		numBlRemaining = ConfigureModel.numBlocks;
		curSM = 0;
		executionTimeTotal = 0;
		double cumulDelay = 0;

		for (int i = 0; i < kernel.getKernelNumBlocks(); i++) {
			if (numBlRemaining == 0)
				break;
			cumulDelay = execTimePerThreadAndReturnDelay(kernel, threadsPerBlock, numBlRemaining);
			//			System.out.println("Delay for this kernel : " + cumulDelay + " cycles");
			if (numBlocksExecuted[curSM] == (int)ConfigureModel.finalMaxBl) {
				curSM = 0;

				executionTimeTotal += UtilsMisc.maxInArray(numBlocksExecuted, ConfigureModel.numSMs) *
						executionTimePerThread;

				numBlThisRound = UtilsMisc.sumArray(numBlocksExecuted, ConfigureModel.numSMs);
				numWavesRounds++;
				numBlOverall += numBlThisRound;
				numBlRemaining -= numBlThisRound;
				numBlThisRound = 0;
				executionTimeTotal += UtilsMisc.maxInArray(numBlocksExecuted, ConfigureModel.numSMs) *
						executionTimePerThread;
				for (int j = 0; j < ConfigureModel.numSMs; j++) {
					numBlocksExecuted[j] = 0;
				}

			} else {
				numBlocksExecuted[curSM]++;
				if (curSM == ConfigureModel.numSMs - 1)
					curSM = 0;
				else
					curSM++;
				numBlThisRound++;
			}
		}

		if (numBlThisRound != 0) {
			numBlThisRound = UtilsMisc.sumArray(numBlocksExecuted, ConfigureModel.numSMs);
			numWavesRounds++;
			numBlOverall += numBlThisRound;
			numBlRemaining -= numBlThisRound;
			numBlThisRound = 0;
			executionTimeTotal += UtilsMisc.maxInArray(numBlocksExecuted, ConfigureModel.numSMs) *
					executionTimePerThread;
		}

		for (int j = 0; j < ConfigureModel.numSMs; j++) {
			numBlocksExecuted[j] = 0;
		}
		for (int i = 0; i < numBlRemaining; i++) {
			if (numBlRemaining == 0)
				break;
			numBlocksExecuted[curSM]++;
			if (curSM == ConfigureModel.numSMs - 1)
				curSM = 0;
			else
				curSM++;
			numBlThisRound++;
		}

		curSM = 0;
		executionTimeTotal += UtilsMisc.maxInArray(numBlocksExecuted, ConfigureModel.numSMs) *
				executionTimePerThread;
		numBlThisRound = UtilsMisc.sumArray(numBlocksExecuted, ConfigureModel.numSMs);
		if (numBlThisRound != 0) numWavesRounds++;
		numBlOverall += numBlThisRound;
		numBlRemaining -= numBlThisRound;
		numBlThisRound = 0;
		for (int j = 0; j < ConfigureModel.numSMs; j++) {
			numBlocksExecuted[j] = 0;
		}

		if (numBlRemaining != 0 || numBlOverall != ConfigureModel.numBlocks)
			System.out.println("something wrong. numBlRemaining = " + numBlRemaining + ", numBlOverall = " + numBlOverall);
		if (ConfigureModel.testingOn) show();

	}

	private void show() {
		;
	}

	public ComputeExecTimeModified() {
		executionTimePerThread = 0;
		executionTimeTotal = 0;
		numBlocksExecuted = new int[ConfigureModel.numSMs];
		for (int i = 0; i < ConfigureModel.numSMs; i++) {
			numBlocksExecuted[i] = 0;
		}
	}

	public double getExecTimePerThread() {
		return executionTimePerThread;
	}

	public double getExecTimeTotal() {
		return executionTimeTotal;
	}

}