package edu.bitsgoa.exectimeestimate;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;

import javax.swing.JOptionPane;

import edu.bitsgoa.driver.ConfigureModel;
import edu.bitsgoa.driver.Preparation;
import edu.bitsgoa.programAnalyzer.PTXUtil;
import edu.bitsgoa.programAnalyzer.instructiontypes.EnumAllInsts;
import edu.bitsgoa.programAnalyzer.instructiontypes.EnumComputeInsts;
import edu.bitsgoa.programAnalyzer.instructiontypes.EnumMemoryInsts;
import edu.bitsgoa.programAnalyzer.instructiontypes.EnumMiscInsts;
import edu.bitsgoa.programAnalyzer.singlekernel.SingleKernel;
import edu.bitsgoa.programAnalyzer.singlekernel.basicblock.BasicBlock;
import edu.bitsgoa.utilities.UtilsMisc;


public class ComputeExecTime {

	private double executionTimePerThread;
	private double executionTimeTotal;
	private int[] numBlocksExecuted;
	public static double coeff1;
	public static double coeff2;
	
	public void calculate(SingleKernel kernel) {
		
		double factor = 1.0;
		
		BasicBlock curBlock = null;
		String curInst = null;
		int numBasicBlocks = kernel.getNumBasicBlocks();
		int numInstsInBlock = 0, maxComputeTypes = 0, maxMemTypes = 0, maxMiscTypes = 0, index = 0, numBankConflicts = 0;;
		double cumulativeDelay = 0.0, curBlockDelay = 0.0, curDelay = 0.0, accessFactorBadCoalescing = 0.0;
		EnumAllInsts instType = null; EnumComputeInsts computeType = null; EnumMemoryInsts memType = null; EnumMiscInsts miscType = null;
		
		int blocks = kernel.getKernelNumBlocks();
		int threadsPerBlock = kernel.getKernelNumThreadsPerBlock();		
		
		accessFactorBadCoalescing = ConfigureModel.accessFactorBadCoalescing;
		numBankConflicts = ConfigureModel.numBankConflicts;
		int ci,mi,msi,nb;
		ci=mi=msi=nb=0;
		for (int i = 0; i < numBasicBlocks; i++) {
			curBlock = kernel.getBasicBlockFromList(i);
			curBlockDelay = 0.0;
			numInstsInBlock = (int)curBlock.getAllInstructionsCount();
			maxComputeTypes = curBlock.computeDetails.length;
			maxMemTypes = curBlock.memoryDetails.length;
			maxMiscTypes = curBlock.miscDetails.length;
			nb++;
			
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
					curBlock.calculateParallelism(threadsPerBlock, blocks);
					curDelay = curBlock.computeDetails[index].getCalculatedDelay(curBlock.getParallelism());
					curBlockDelay += curDelay;
					ci++;
					System.out.println("Compute Cur Delay: "+curDelay+"Block Delay:"+curBlockDelay);
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
					curBlock.calculateParallelism(threadsPerBlock, blocks);
					curDelay = curBlock.memoryDetails[index].getCalculatedDelay(curBlock.getParallelism(), memType, accessFactorBadCoalescing, numBankConflicts);
					curBlockDelay += curDelay;
					mi++;
					System.out.println(" Memory Cur Delay: "+curDelay+"Block Delay:"+curBlockDelay);
				}
				
				else if (instType == EnumAllInsts.Unknown)
					;
				
				else {
					miscType = PTXUtil.decodeMiscInstType(curInst);
					index = 0;
				
					for (int k = 0; k < maxMiscTypes-1; k++) {
						if (miscType == curBlock.miscDetails[k].getInstType()) {
							index = k;
							msi++;
							break;
						}
						
					}
					curBlock.calculateParallelism(threadsPerBlock, blocks);
					curDelay = curBlock.miscDetails[index].getCalculatedDelay(curBlock.getParallelism());
					curBlockDelay += curDelay;
					System.out.println("Misc Cur Delay: "+curDelay+"Block Delay:"+curBlockDelay);
				}
				//System.out.println("Block: "+i+"  Compute: "+ci+"  Memory: "+mi+"  Misc: "+msi);
			}
			
			if (curBlock.hasLoop())
				curBlockDelay *=ConfigureModel.numLoopIterations;
			cumulativeDelay += curBlockDelay;
			
		}
		
		
		//System.out.println("Total Instructions : No.of.Blocks: "+nb+" Compute: "+ci+" Memory: "+mi+" Misc: "+msi);
		cumulativeDelay = factor * cumulativeDelay;
	    System.out.println("CUM Delay:"+cumulativeDelay);	
	    System.out.println("Block Delay:"+curBlockDelay);
		
	    executionTimePerThread = cumulativeDelay / ConfigureModel.gpuClock;
		
		int numWarpsIssuingInstsPerCyclePerCore = ConfigureModel.numCoresPerSM/ConfigureModel.warpSize;
		int numWarpsPerBlock = kernel.getKernelNumThreadsPerBlock()/ConfigureModel.warpSize;
		if (kernel.getKernelNumThreadsPerBlock() % ConfigureModel.warpSize != 0)
			numWarpsPerBlock = numWarpsPerBlock + 1;
		int numIssueCycles = numWarpsPerBlock / numWarpsIssuingInstsPerCyclePerCore;
		if (numWarpsPerBlock % numWarpsIssuingInstsPerCyclePerCore != 0)
			numIssueCycles = numIssueCycles + 1;
		
		//System.out.print(" ExecTime * numIssue  :"+executionTimePerThread + " * " + numIssueCycles + " ");
		executionTimePerThread *= numIssueCycles; // CETV3
		//System.out.println("Execution Time per thread:"+executionTimePerThread);
		@SuppressWarnings("unused")
		int numBlThisRound = 0, numWavesRounds = 0;
		int numBlOverall = 0;
		int numBlRemaining = ConfigureModel.numBlocks;
		int curSM = 0;
		int totalThreads;
		double  finalMaxBl, intMaxBl;
		executionTimeTotal = 0;
		totalThreads=ConfigureModel.numBlocks*ConfigureModel.numThreadsPerBlock;
		intMaxBl=totalThreads/(ConfigureModel.maxThreadPerSm*ConfigureModel.numSMs);
		finalMaxBl=ConfigureModel.finalMaxBl;
		
		// CETV2
		for (int i = 0; i < kernel.getKernelNumBlocks(); i++) {
			if (numBlRemaining == 0)
				break;
			if (numBlocksExecuted[curSM] == finalMaxBl) {
				curSM = 0;
				
				executionTimeTotal += UtilsMisc.maxInArray(numBlocksExecuted, ConfigureModel.numSMs) *
						executionTimePerThread;
				
				numBlThisRound = UtilsMisc.sumArray(numBlocksExecuted, ConfigureModel.numSMs);
				numWavesRounds++;
				numBlOverall += numBlThisRound;
				numBlRemaining -= numBlThisRound;
				numBlThisRound = 0;
				//System.out.println("\nCurrent SM :  "+numBlocksExecuted[curSM]+" Execution Time Total : "+executionTimeTotal);
				executionTimeTotal += UtilsMisc.maxInArray(numBlocksExecuted, ConfigureModel.numSMs) *
						executionTimePerThread;
				for (int j = 0; j < ConfigureModel.numSMs; j++) {
					numBlocksExecuted[j] = 0;
						
				}
				//System.out.println("\n \nWaves:"+numWavesRounds);
				
			} else {
				//System.out.println("\nElse : Current SM :  "+numBlocksExecuted[curSM]+" Execution Time Total : "+executionTimeTotal);
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
		//System.out.println("Number of threads Executed  :  "+ConfigureModel.numThreadsPerBlock*ConfigureModel.numBlocks);
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
		System.out.println("ExecutionTimePerThread : "+executionTimePerThread);
	    //executionTimeTotal+=ConfigureModel.kernelLaunchOverhead;
	
		
	}
	
	private void show() {
		;
	}
	
	public ComputeExecTime() {
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
		
		double launchoverhead=0;
		int totalThreads;
		//System.out.print("executionTimeTotal : ");
		totalThreads=ConfigureModel.numBlocks*ConfigureModel.numThreadsPerBlock;
		readModel();
		launchoverhead=coeff2*totalThreads + coeff1;
		executionTimeTotal+=launchoverhead;
		return executionTimeTotal;

	}
	public void readModel(){
		try {
			String path=Preparation.path_home+"/KernelLaunchOverheadModel.txt";
			BufferedReader buf=new BufferedReader(new FileReader(path));
			String line;
			int lineNo=0;
			while((line=buf.readLine())!=null){
				if(lineNo==0) coeff1=Double.parseDouble(line);
				else coeff2=Double.parseDouble(line);
				lineNo++;
			}
		} catch (FileNotFoundException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}

}