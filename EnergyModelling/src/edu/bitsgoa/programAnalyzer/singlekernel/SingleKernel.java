package edu.bitsgoa.programAnalyzer.singlekernel;

import java.util.ArrayList;
import java.util.HashMap;

import edu.bitsgoa.programAnalyzer.PTXUtil;
import edu.bitsgoa.programAnalyzer.instructiontypes.AllInstData;
import edu.bitsgoa.programAnalyzer.instructiontypes.EnumAllInsts;
import edu.bitsgoa.programAnalyzer.instructiontypes.EnumComputeInsts;
import edu.bitsgoa.programAnalyzer.instructiontypes.EnumMemoryInsts;
import edu.bitsgoa.programAnalyzer.instructiontypes.EnumMiscInsts;
import edu.bitsgoa.programAnalyzer.singlekernel.basicblock.BasicBlock;
import edu.bitsgoa.utilities.UtilsMisc;



public class SingleKernel {
	
	private String kernelName;
	private int kernelNumber;
	
	private int kernelNumThreadsPerBlock, kernelNumBlocks;
	@SuppressWarnings("unused") private int maxNumActiveThreadsPerSM, maxNumActiveWarpsPerSM, maxNumActiveBlocksPerSM;
	@SuppressWarnings("unused") private double theoreticalMaxOccupancy;

	private ArrayList<String> overallInstructionList;
	private HashMap<String, Integer> overallInstTypeCount; // get from basic blocks
	private double overallInstExecCount;
	
	private ArrayList<BasicBlock> basicBlockList;
	private int numBasicBlocks;
		
	private ControlFlow controlFlowGraph;
	
	private double overallDelay;
	
	public String kernelNumString,
		allInstsString, perBlockCountString, controlFlowString, instsExecCountString,
		execTimeString, verifyString;
	
	public void constructAllDFGs() {
		BasicBlock curBlock = null;
		for (int i = 0; i < numBasicBlocks; i++) {
			curBlock = basicBlockList.get(i);
			curBlock.constructDFG();
		}
	}
	
	public void constructBasicBlocks() {
		numBasicBlocks = 0;
		overallInstExecCount = 0;
		basicBlockList.clear();
		int currentBlockNum = 0, currentBlockInstCount = 0;
		EnumAllInsts instType = null; String instruction = null;
		
		int numInstructions = overallInstructionList.size();
		for (int i = 0; i < numInstructions; i++) {
			instruction = overallInstructionList.get(i);
			instType = PTXUtil.decodeInstruction(instruction);
			if (PTXUtil.canBeSkipped(instruction)) continue;
			
			if (instType == EnumAllInsts.Branch || instType == EnumAllInsts.Return) {
				if (currentBlockInstCount == 0) {
					BasicBlock newBlock = new BasicBlock();
					newBlock.setBlockNum(currentBlockNum + 1);
					basicBlockList.add(currentBlockNum, newBlock);
				}
				basicBlockList.get(currentBlockNum).putInstruction(instruction);
				if (instType == EnumAllInsts.Branch)
					basicBlockList.get(currentBlockNum).setBranchOut(PTXUtil.getLabelName(instruction));
				if (instType == EnumAllInsts.Return)
					basicBlockList.get(currentBlockNum).setLastBlock();
				currentBlockInstCount++;
				currentBlockNum++;
				numBasicBlocks++;
				currentBlockInstCount = 0;
			}
			else if (instType == EnumAllInsts.Label) {
				if (currentBlockInstCount != 0) {
					currentBlockNum++;
					numBasicBlocks++;
					currentBlockInstCount = 0;
				}
				BasicBlock newBlock = new BasicBlock();
				newBlock.setBlockNum(currentBlockNum + 1);
				basicBlockList.add(currentBlockNum, newBlock);
				basicBlockList.get(currentBlockNum).putInstruction(instruction);
				basicBlockList.get(currentBlockNum).setStartLabel(PTXUtil.getLabelName(instruction));
				currentBlockInstCount++;
			}
			else {
				if (currentBlockInstCount == 0) {
					BasicBlock newBlock = new BasicBlock();
					newBlock.setBlockNum(currentBlockNum + 1);
					basicBlockList.add(currentBlockNum, newBlock);
				}
				basicBlockList.get(currentBlockNum).putInstruction(instruction);
				currentBlockInstCount++;
			}
		}
		
		currentBlockInstCount = 0;
		numBasicBlocks = basicBlockList.size();
		BasicBlock curBlock = null;
		
		// detect loops
		for (int i = 0; i < numBasicBlocks; i++) {
			curBlock = basicBlockList.get(i);
			if (curBlock.getStartLabel() != null && !curBlock.getStartLabel().equals("") && curBlock.getStartLabel().equals(curBlock.getBranchToLabel()))
				curBlock.setLoopDetected();
		}
	}
	
	
	public void countInstsPerBlock() {
		numBasicBlocks = basicBlockList.size();
		BasicBlock curBlock = null;
		for (int i = 0; i < numBasicBlocks; i++) {
			curBlock = basicBlockList.get(i);
			curBlock.countInstructionTypesInBlock();
		}
		countOverallInsts();
	}
	
	public void putInstruction(String instruction) {
		if (PTXUtil.canBeSkipped(instruction))
			return;
		overallInstructionList.add(instruction.trim());
	}
	
	
	public void countOverallInsts() {
		BasicBlock curBlock = null;
		EnumComputeInsts instType = null;
		int count = 0;
		for (int i = 0; i < numBasicBlocks; i++) {
			curBlock = basicBlockList.get(i);
			for (int j = 0; j < AllInstData.numComputeInstTypes; j++) {
				instType = AllInstData.computeDetails[j].getInstType();
//				if (instType == InstTypesEnum.Label)
//					continue;
				count = curBlock.getComputeInstTypeCount(instType);
				if (overallInstTypeCount.containsKey(instType + ""))
					overallInstTypeCount.put(instType + "", overallInstTypeCount.get(instType + "") + count);
				else
					overallInstTypeCount.put(instType + "", count);
			}
		}
	}

	public int getCountOfPTXInst(String instType) {
		return overallInstTypeCount.get(instType);
	}
	
	public void constructControlFlowAndCount() {
		controlFlowGraph.constructControlFlow(basicBlockList);
		overallInstExecCount = controlFlowGraph.countInstsStartingAtAllBlocks();
	}

	public void getDetailsAsStrings() {
		int numThreads = kernelNumThreadsPerBlock * kernelNumBlocks;
		
		kernelNumString = UtilsMisc.blankSpace +
				"\nKernel number: " + kernelNumber + "\t\t\tKernel name: " + kernelName + "\n";
		
		StringBuilder allInstsBuilder = new StringBuilder();
		allInstsBuilder.append("\nInstructions:\n");
		for (int i = 0; i < overallInstructionList.size(); i++) {
			allInstsBuilder.append(overallInstructionList.get(i) + "\n");
		}
		allInstsString = allInstsBuilder.toString();
		
		StringBuilder instCountsBuilder = new StringBuilder();
		instCountsBuilder.append("\nInstructions Executed Per Block:\n\n");
		BasicBlock curBlock = null;
		
		EnumComputeInsts computeInstType = null; 
		EnumMemoryInsts memInstType = null;
		EnumMiscInsts miscInstType = null;

		int count = 0;
		
		for (int i = 0; i < numBasicBlocks; i++) {
			curBlock = basicBlockList.get(i);
			instCountsBuilder.append("Block" + curBlock.getBlockNum() + " - ");
			for (int j = 0; j < AllInstData.numComputeInstTypes; j++) {
				computeInstType = AllInstData.computeDetails[j].getInstType();
				count = curBlock.getComputeInstTypeCount(computeInstType);
				if (count != 0)
					instCountsBuilder.append(computeInstType + ": " + count + "; ");
			}
			for (int j = 0; j < AllInstData.numMemoryInstTypes; j++) {
				memInstType = AllInstData.memoryDetails[j].getInstType();
				count = curBlock.getMemoryInstTypeCount(memInstType);
				if (count != 0)
					instCountsBuilder.append(memInstType + ": " + count + "; ");
			}
			for (int j = 0; j < AllInstData.numMiscInstTypes; j++) {
				miscInstType = AllInstData.miscDetails[j].getInstType();
				count = curBlock.getMiscInstTypeCount(miscInstType);
				if (count != 0)
					instCountsBuilder.append(miscInstType + ": " + count + "; ");
			}
			instCountsBuilder.append("\n\t\t(Total: " + curBlock.getExecutedInstructionsCount() + ")" + "\n");
		}
		instCountsBuilder.append("\nTotal kernel delay: " + overallDelay + "\n");
		perBlockCountString = instCountsBuilder.toString();
		
		controlFlowString = "\nControl Flow Graph:\n" + controlFlowGraph.getCFGAsString();
		
		instCountsBuilder = new StringBuilder();
		instCountsBuilder.append("\nInstructions Executed if Starting at Each Block (intermediate data for CFG instruction count):\n");
		instCountsBuilder.append(controlFlowGraph.instsStartingAtBlock());
		instCountsBuilder.append("\n\nTot insts. executed (per thread): " + overallInstExecCount);
		instCountsBuilder.append("\nTot insts. (all " + numThreads + " threads): " + (overallInstExecCount*numThreads) + "\n");
		instsExecCountString = instCountsBuilder.toString();
	}
	
	public void setKernelLaunchParameters(int kernelNumBlocks, int kernelNumThreadsPerBlock) {
		this.kernelNumBlocks = kernelNumBlocks;
		this.kernelNumThreadsPerBlock = kernelNumThreadsPerBlock;
	}
	
	public void setKernelNameAndNumber(String kernelName, int kernelNum) {
		this.kernelName = kernelName;
		this.kernelNumber = kernelNum;
	}
	
	public void resetCount() {
		for (int i = 0; i < AllInstData.numComputeInstTypes; i++) {
			overallInstTypeCount.put(AllInstData.computeDetails[i].getInstName(), 0);
		}
		overallInstructionList.clear();
		for (int i = 0; i < basicBlockList.size(); i++)
			basicBlockList.get(i).reset();
		basicBlockList.clear();
		numBasicBlocks = 0;
		overallDelay = 0;
		kernelNumString = allInstsString = instsExecCountString = controlFlowString = execTimeString = verifyString = "";
	}
	
	public SingleKernel(int blocks, int threadsPerBlock) {
		kernelNumBlocks = blocks;
		kernelNumThreadsPerBlock = threadsPerBlock;
		
		overallInstExecCount = 0;
		overallInstTypeCount = new HashMap<String, Integer>();
		overallInstructionList = new ArrayList<String>();
		basicBlockList = new ArrayList<BasicBlock>();
		kernelName = "";
		kernelNumber = -1;
		numBasicBlocks = 0;
		overallDelay = 0.0;
		controlFlowGraph = new ControlFlow();
		kernelNumString = allInstsString = instsExecCountString = controlFlowString = execTimeString = verifyString = "";
		resetCount();
	}
		
	public int getKernelNumThreadsPerBlock() {
		return kernelNumThreadsPerBlock;
	}
	
	public int getKernelNumBlocks() {
		return kernelNumBlocks;
	}
	
	public BasicBlock getBasicBlockFromList(int index) {
		if (index >= 0 && index < numBasicBlocks)
			return basicBlockList.get(index);
		return null;
	}
	
	public int getNumBasicBlocks() {
		return numBasicBlocks;
	}
	
	public int getKernelNumber() {
		return kernelNumber;
	}
	
	public String getKernelName() {
		return kernelName;
	}
	
}