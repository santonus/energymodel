package edu.bitsgoa.programAnalyzer.singlekernel.basicblock;

import java.util.ArrayList;
import java.util.HashMap;

import edu.bitsgoa.driver.ConfigureModel;
import edu.bitsgoa.exectimeestimate.ComputeExecTime;
import edu.bitsgoa.programAnalyzer.PTXUtil;
import edu.bitsgoa.programAnalyzer.instructiontypes.AllInstData;
import edu.bitsgoa.programAnalyzer.instructiontypes.DataComputeInst;
import edu.bitsgoa.programAnalyzer.instructiontypes.DataMemoryInst;
import edu.bitsgoa.programAnalyzer.instructiontypes.DataMiscInst;
import edu.bitsgoa.programAnalyzer.instructiontypes.EnumAllInsts;
import edu.bitsgoa.programAnalyzer.instructiontypes.EnumComputeInsts;
import edu.bitsgoa.programAnalyzer.instructiontypes.EnumMemoryInsts;
import edu.bitsgoa.programAnalyzer.instructiontypes.EnumMiscInsts;

public class BasicBlock {

	private int blockNum;
	private DataFlow dfgBasicBlock;
	public ArrayList<String> allInstructionsList;
	public double allInstructionsCount, executedInstructionsCount;
	public DataComputeInst[] computeDetails;
	public int numComputeInstTypes;
	public DataMemoryInst[] memoryDetails;
	public int numMemoryInstTypes;
	public DataMiscInst[] miscDetails;
	public int numMiscInstTypes;
	public int unknownCount;
	
	private HashMap<EnumComputeInsts, Integer> computeInstTypeCount;
	private HashMap<EnumMemoryInsts, Integer> memoryInstTypeCount;
	private HashMap<EnumMiscInsts, Integer> miscInstTypeCount;
	private String startLabel, branchToLabel;
	private boolean labelIn, branchOut, loopDetected, returnBlock;
	private double ILP;
	private int TLP;
	private double parallelism;
	
	public void calculateILP() {
		ILP = dfgBasicBlock.calculateILP(allInstructionsCount);
	}
	
	public void calculateTLP(int kernelNumThreadsPerBlock, int kernelNumBlocks) {
		int threadsPerBlock = kernelNumThreadsPerBlock;
		int maxBlSM = (int)ConfigureModel.finalMaxBl;
		double warpsPerBlock = 1.0 * threadsPerBlock / ConfigureModel.warpSize; // if 16/32
		double maxWarpsSM = maxBlSM * warpsPerBlock;
		System.out.println("maxBLSM is: "+maxBlSM);
		System.out.println("warpSize is: "+ConfigureModel.warpSize);
		System.out.println("threadsPerBlock are: "+threadsPerBlock);
		System.out.println("warpsPerBlock is: "+warpsPerBlock);
		System.out.println("maxWarpSM is: "+maxWarpsSM);
		TLP = (int) Math.ceil(maxWarpsSM);
	}
	
	public void calculateParallelism(int kernelNumThreadsPerBlock, int kernelNumBlocks) {
		calculateILP();
		calculateTLP(kernelNumThreadsPerBlock, kernelNumBlocks);
		//System.out.println("Block No :"+blockNum+" TLP:" +TLP+" and  ILP:"+ILP);
		System.out.println("TLP is: "+TLP);
		System.out.println("ILP is: "+ILP);
		parallelism = TLP * ILP;
	}
	
	public void countInstructionTypesInBlock() {
		allInstructionsCount = allInstructionsList.size();
		executedInstructionsCount = 0;
		computeInstTypeCount.clear();
		memoryInstTypeCount.clear();
		miscInstTypeCount.clear();
		
		for (int i = 0; i < numComputeInstTypes; i++) {
			computeInstTypeCount.put(EnumComputeInsts.values()[i], 0);
		}
		for (int i = 0; i < numMemoryInstTypes; i++) {
			memoryInstTypeCount.put(EnumMemoryInsts.values()[i], 0);
		}
		for (int i = 0; i < numMiscInstTypes; i++) {
			miscInstTypeCount.put(EnumMiscInsts.values()[i], 0);
		}
		
		EnumAllInsts instType = null;
		EnumComputeInsts computeType = null;
		EnumMemoryInsts memType = null;
		EnumMiscInsts miscType = null;
		String curInst = null;
		int value = 0;
		
		for (int i = 0; i < allInstructionsCount; i++) {
			curInst = allInstructionsList.get(i);
			instType = PTXUtil.decodeInstruction(curInst);
			
			if (instType == EnumAllInsts.Computation) {
				computeType = PTXUtil.decodeComputationType(curInst);
				if (computeInstTypeCount.containsKey(computeType))
					value = computeInstTypeCount.get(computeType) + 1;
				else
					value = 1;
				computeInstTypeCount.put(computeType, value);
			}
			
			else if (instType == EnumAllInsts.MemAccess) {
				memType = PTXUtil.decodeMemoryType(curInst);
				if (memoryInstTypeCount.containsKey(memType))
					value = memoryInstTypeCount.get(memType) + 1;
				else
					value = 1;
				memoryInstTypeCount.put(memType, value);
			}
			
			else if (instType == EnumAllInsts.Unknown)
				unknownCount++;
			
			else {
				miscType = PTXUtil.decodeMiscInstType(curInst);
				if (miscInstTypeCount.containsKey(miscType))
					value = miscInstTypeCount.get(miscType) + 1;
				else
					value = 1;
				miscInstTypeCount.put(miscType, value);
			}
			
			if (PTXUtil.isExecutableInst(curInst))
				executedInstructionsCount++;
		}
	}
	
	

	public void constructDFG() {
		allInstructionsCount = allInstructionsList.size();
		dfgBasicBlock = new DataFlow((int)allInstructionsCount);
		String curInst = "";
		for (int i = 0; i < allInstructionsCount; i++) {
			curInst = allInstructionsList.get(i);
			dfgBasicBlock.createAndAddDFGNode(curInst);
		}
		dfgBasicBlock.constructDependencies();
		calculateILP();
	}
	
	
	public String getInstCountsOfBlockAsString() {
		StringBuilder resultString = new StringBuilder();
		resultString.append("\nInstruction Count Within Block:\n");
		resultString.append("\nInstruction\t\t\tCount\n");
		int value = 0, len = 0;
		EnumMemoryInsts memInstType = null;
		EnumComputeInsts computeInstType = null;
		EnumMiscInsts miscInstType = null;
		for (int i = 0; i < numComputeInstTypes; i++) {
			
			computeInstType = computeDetails[i].getInstType();
			len = ("" + computeInstType).length();
			resultString.append(computeInstType + "                      ".substring(len));
			
			value = getComputeInstTypeCount(computeInstType);
			len = ("" + value).length();
			if (value != 0) resultString.append(value + "");
			else resultString.append("-");
			resultString.append("\n");
		}
		resultString.append("\n");
		for (int i = 0; i < numMemoryInstTypes; i++) {
			
			memInstType = memoryDetails[i].getInstType();
			len = ("" + memInstType).length();
			resultString.append(memInstType + "                      ".substring(len));

			value = getMemoryInstTypeCount(memInstType);
			len = ("" + value).length();
			if (value != 0) resultString.append(value + "");
			else resultString.append("-");
			resultString.append("\n");
		}
		resultString.append("\n");
		
		for (int i = 0; i < numMiscInstTypes; i++) {
			
			miscInstType = miscDetails[i].getInstType();
			len = ("" + miscInstType).length();
			resultString.append(miscInstType + "                      ".substring(len));

			value = getMiscInstTypeCount(miscInstType);
			len = ("" + value).length();
			if (value != 0) resultString.append(value + "");
			else resultString.append("-");
			resultString.append("\n");
		}
		
		len = "Unknown".length();
		resultString.append("\nUnknown" + "                      ".substring(len));
		if (unknownCount != 0) resultString.append(unknownCount + "\n");
		else resultString.append("-\n");
		
		return resultString.toString();
	}
	
	public String getBasicBlockAsString() {
		StringBuilder resultString = new StringBuilder();
		resultString.append("\nBlock Number " + blockNum + ":\n\n");
		if (labelIn)
			resultString.append("LabelIn: " + startLabel + "\n\n");
		if (branchOut)
			resultString.append("BranchOut: " + branchToLabel + "\n\n");
		if (loopDetected)
			resultString.append("Loop detected: This block consists of a loop body.\n\n");
		if (returnBlock)
			resultString.append("This is the last block to be executed.\n\n");
		for (int i = 0; i < allInstructionsCount; i++) {
			resultString.append(allInstructionsList.get(i) + "\n");
		}
		resultString.append("\n");
		return resultString.toString();
	}
	
	public int getComputeInstTypeCount(EnumComputeInsts instType) {
		if (computeInstTypeCount.get(instType) == null)
			return 0;
		return computeInstTypeCount.get(instType);
	}
	
	public int getMemoryInstTypeCount(EnumMemoryInsts instType) {
		if (memoryInstTypeCount.get(instType) == null)
			return 0;
		return memoryInstTypeCount.get(instType);
	}
	
	public int getMiscInstTypeCount(EnumMiscInsts instType) {
		if (miscInstTypeCount.get(instType) == null)
			return 0;
		return miscInstTypeCount.get(instType);
	}
	
	public int getUnknownCount() {
		return unknownCount;
	}
	
	public BasicBlock() {
		dfgBasicBlock = null;
		computeDetails = new DataComputeInst[AllInstData.numComputeInstTypes];
		memoryDetails = new DataMemoryInst[AllInstData.numMemoryInstTypes];
		miscDetails = new DataMiscInst[AllInstData.numMiscInstTypes];
		numComputeInstTypes = AllInstData.numComputeInstTypes;
		for (int i = 0; i < numComputeInstTypes; i++) {
			computeDetails[i] = new DataComputeInst();
			computeDetails[i] = AllInstData.computeDetails[i];
		}
		numMemoryInstTypes = AllInstData.numMemoryInstTypes;
		for (int i = 0; i < numMemoryInstTypes; i++) {
			memoryDetails[i] = new DataMemoryInst();
			memoryDetails[i] = AllInstData.memoryDetails[i];
		}
		numMiscInstTypes = AllInstData.numMiscInstTypes;
		for (int i = 0; i < numMiscInstTypes; i++) {
			miscDetails[i] = new DataMiscInst();
			miscDetails[i] = AllInstData.miscDetails[i];
		}
		allInstructionsList = new ArrayList<String>();
		computeInstTypeCount = new HashMap<EnumComputeInsts, Integer>();
		memoryInstTypeCount = new HashMap<EnumMemoryInsts, Integer>();
		miscInstTypeCount = new HashMap<EnumMiscInsts, Integer>();
		reset();
	}
	
	public void reset() {
		blockNum = 0;
		ILP = 0.0;
		TLP = 0;
		parallelism = 0.0;
		unknownCount = 0;
		allInstructionsList.clear();
		computeInstTypeCount.clear();
		memoryInstTypeCount.clear();
		miscInstTypeCount.clear();
		for (int i = 0; i < numComputeInstTypes; i++) {
			computeDetails[i].reset();
			computeInstTypeCount.put(EnumComputeInsts.values()[i], 0);
		}
		for (int i = 0; i < numMemoryInstTypes; i++) {
			memoryDetails[i].reset();
			memoryInstTypeCount.put(EnumMemoryInsts.values()[i], 0);
		}
		for (int i = 0; i < EnumMiscInsts.values().length; i++) {
			miscDetails[i].reset();
			miscInstTypeCount.put(EnumMiscInsts.values()[i], 0);
		}
		allInstructionsCount = executedInstructionsCount = 0;
		labelIn = branchOut = loopDetected = returnBlock = false;		
		startLabel = branchToLabel = "";
	}
	
	public void putInstruction(String instruction) {
		allInstructionsList.add(instruction.trim());
	}
	
	public void setStartLabel(String startLabel) {
		if (startLabel == null || startLabel.equals("")) return;
		labelIn = true;
		this.startLabel = startLabel;
	}
	
	public String getInstructionNumber(int num) {
		if (num < 0 || num >= allInstructionsCount)
			return null;
		return allInstructionsList.get(num);
	}
	
	public void setBranchOut(String branchToLabel) {
		if (branchToLabel == null || branchToLabel.equals("")) return;
		branchOut = true;
		this.branchToLabel = branchToLabel;
	}
	
	public int getBlockNum() {
		return blockNum;
	}
	
	public void setBlockNum(int num) {
		this.blockNum = num;
	}
	
	public double getAllInstructionsCount() {
		return allInstructionsCount;
	}
	
	public double getExecutedInstructionsCount() {
		return executedInstructionsCount;
	}
	
	public boolean hasLoop() {
		return loopDetected;
	}
	
	public void setLoopDetected() {
		loopDetected = true;
	}
	
	public boolean isLastBlock() {
		return returnBlock;
	}
	
	public void setLastBlock() {
		returnBlock = true;
	}
	
	public boolean hasStartLabel() {
		return labelIn;
	}
	
	public String getStartLabel() {
		if (labelIn)
			return startLabel;
		return "";
	}
	
	public boolean hasBranchOut() {
		return branchOut;
	}
	
	public String getBranchToLabel() {
		if (branchOut)
			return branchToLabel;
		return "";
	}
	
	public double getILP() {
		return ILP;
	}
	
	public int getTLP() {
		return TLP;
	}
	
	public double getParallelism() {
		return parallelism;
	}
	
	public DFGNode getDFGNodeNumber(int num) {
		return dfgBasicBlock.getNodeAtIndex(num);
	}
	
	public int getNumOfDFGNodes() {
		return dfgBasicBlock.getNumDfgNodes();
	}
	
	public String getDFGDetailsAsString(int kernelNumThreadsPerBlock, int kernelNumBlocks) {
		DFGNode curNode = null;
		StringBuilder resultString = new StringBuilder();
		resultString.append("\nData Flow:\n\n");
		if (dfgBasicBlock == null) {
			resultString.append("Not constructed.\n\n");
			return resultString.toString();
		}
		for (int i = 0; i < dfgBasicBlock.getNumDfgNodes(); i++) {
			curNode = dfgBasicBlock.getNodeAtIndex(i);
			if (curNode == null)
				continue;
			resultString.append(curNode.getNodeInstNum() +  " - " + 
					curNode.getRegListAsString() + curNode.independenceMarker());
		}
		calculateParallelism(kernelNumThreadsPerBlock, kernelNumBlocks);
		resultString.append("\nILP: " + ILP + "\nTLP: " + TLP + "\nParallelism: " + parallelism + "\n");
		return resultString.toString();
	}
}