package edu.bitsgoa.programAnalyzer.singlekernel;

import java.util.ArrayList;

import edu.bitsgoa.driver.ConfigureModel;
import edu.bitsgoa.programAnalyzer.singlekernel.basicblock.BasicBlock;



public class ControlFlow {
	
	private ArrayList<BasicBlock> basicBlockList;
	private ArrayList<BlockData> cfgBlockList;
	private int numBasicBlocks;
	
	private boolean constructionComplete;
	
	private boolean[] blockCounted;
	private double[] blockInstCount;
	
	public double countInstsStartingAtAllBlocks() {
		if (!constructionComplete)
			return 0.0;
		double countInstExec = 0;
		
		blockCounted = new boolean[numBasicBlocks];
		blockInstCount = new double[numBasicBlocks];
		
		for (int i = 0; i < numBasicBlocks; i++) {
			blockCounted[i] = false;
			blockInstCount[i] = 0.0;
		}
		
		for (int i = 0; i < numBasicBlocks; i++) {
			ArrayList<BasicBlock> visitedWhileTraversing = new ArrayList<BasicBlock>();
			ArrayList<Double> countEachBlockVisited = new ArrayList<Double>();
			blockInstCount[i] = countInstsStartingAtBlock(i+1, visitedWhileTraversing, countEachBlockVisited);
			countInstExec += blockInstCount[i];
		}
		
		return countInstExec;
	}
	
	public double countInstsStartingAtBlock(int block, ArrayList<BasicBlock> visitedWhileTraversing, ArrayList<Double> countEachBlockVisited) {
		BlockData curBlock = getBlockDataWithNumber(block);
		if (curBlock == null)
			return 0.0;
		
		double count = 0.0, addFromBranch1 = 0.0, addFromBranch2 = 0.0;
		int index = 0, size = 0;
		
		if (visitedWhileTraversing.contains(getBasicBlockWithNumber(block))) {
			size = visitedWhileTraversing.size();
			index = visitedWhileTraversing.indexOf(getBasicBlockWithNumber(block));
			for (int i = index; i < size; i++) {
				count += visitedWhileTraversing.get(i).getExecutedInstructionsCount();
			}
			count = count * ConfigureModel.numLoopIterations;
			return count;
		}
		
		if (block > 0 && blockCounted[block - 1] == true)
			return blockInstCount[block - 1];
				
		count = curBlock.getBlock().getExecutedInstructionsCount();
		
		if (curBlock.isAnyBranchValid()) {
			visitedWhileTraversing.add(curBlock.getBlock());
			countEachBlockVisited.add(curBlock.getBlock().getExecutedInstructionsCount());
		}
		
		if (curBlock.isBranch1Valid()) {
			addFromBranch1 = countInstsStartingAtBlock(curBlock.blockBranch1, visitedWhileTraversing, countEachBlockVisited);
			if (curBlock.isBranch2Valid())
				addFromBranch1 = ConfigureModel.branchProbability * addFromBranch1;
		}
		if (curBlock.isBranch2Valid()) {
			addFromBranch2 = countInstsStartingAtBlock(curBlock.blockBranch2, visitedWhileTraversing, countEachBlockVisited);
			if (curBlock.isBranch1Valid())
				addFromBranch2 = ConfigureModel.branchProbability * addFromBranch2;
		}
		count = count + addFromBranch1 + addFromBranch2;
		
		blockCounted[block - 1] = true;
		blockInstCount[block - 1] = count;
		return count;
	}
	
	public void constructControlFlow(ArrayList<BasicBlock> basicBlockListIn) {
		if (basicBlockListIn == null || basicBlockListIn.size() == 0)
			return;
		this.basicBlockList = basicBlockListIn;
		numBasicBlocks = basicBlockList.size();
		
		cfgBlockList = new ArrayList<BlockData>();
		
		BlockData temp = null;
		BasicBlock curBlock = null;
		int nextBlock1 = 0, nextBlock2 = 0;
		for (int i = 0; i < numBasicBlocks; i++) {
			curBlock = basicBlockList.get(i);
			temp = new BlockData(curBlock);
			if (curBlock.isLastBlock()) {
				cfgBlockList.add(temp);
				continue;
			}
			if (i == numBasicBlocks - 1)
				nextBlock1 = -1;
			else
				nextBlock1 = basicBlockList.get(i+1).getBlockNum();
			if (curBlock.hasBranchOut()) {
				nextBlock2 = getBlockWithStartLabel(curBlock.getBranchToLabel());
				if (nextBlock2 != nextBlock1)
					temp.setBranch2(nextBlock2);
			}
			temp.setBranch1(nextBlock1);
			cfgBlockList.add(temp);
		}
		constructionComplete = true;
	}
	
	public String instsStartingAtBlock() {
		if (!constructionComplete)
			return "";
		StringBuilder resultString = new StringBuilder();
		for (int i = 0; i < numBasicBlocks; i++) {
			resultString.append("\nStarting with block " + (i+1) + ": " + blockInstCount[i]);
		}
		return resultString.toString();
	}
	
	public String getCFGAsString() {
		if (!constructionComplete)
			return "";
		BlockData temp = null;
		StringBuilder resultString = new StringBuilder();
		for (int i = 0; i < numBasicBlocks; i++) {
			temp = cfgBlockList.get(i);
			resultString.append(temp.getBlock().getBlockNum() + " : " +
					temp.branch1Valid + " (" + temp.blockBranch1 + "), " +
					temp.branch2Valid + " (" + temp.blockBranch2 + ")\n");
		}
		return resultString.toString();
	}
	
	public ControlFlow() {
		basicBlockList = null;
		cfgBlockList = null;
		numBasicBlocks = 0;
		constructionComplete = false;
		blockCounted = null;
		blockInstCount = null;
	}
	
	public BasicBlock getBasicBlockWithNumber(int num) {
		BasicBlock curBlock = null;
		for (int i = 0; i < numBasicBlocks; i++) {
			curBlock = basicBlockList.get(i); 
			if (curBlock.getBlockNum() == num)
				return curBlock;
		}
		return null;
	}
	
	public BlockData getBlockDataWithNumber(int num) {
		if (!constructionComplete)
			return null;
		BlockData curBlock = null;
		for (int i = 0; i < numBasicBlocks; i++) {
			curBlock = cfgBlockList.get(i); 
			if (curBlock.getBlock().getBlockNum() == num)
				return curBlock;
		}
		return null;
	}
	
	public int getBlockWithStartLabel(String startLabel) {
		BasicBlock curBlock = null;
		for (int i = 0; i < numBasicBlocks; i++) {
			curBlock = basicBlockList.get(i);
			if (curBlock.hasStartLabel() && curBlock.getStartLabel().equals(startLabel))
				return i+1;
		}
		return -1;
	}
	
	public int getBlockWithBranchToLabel(String branchToLabel) {
		BasicBlock curBlock = null;
		for (int i = 0; i < numBasicBlocks; i++) {
			curBlock = basicBlockList.get(i);
			if (curBlock.hasBranchOut() && curBlock.getBranchToLabel() == branchToLabel)
				return i+1;
		}
		return -1;
	}
	
	class BlockData {
		BasicBlock basicBlock;
		
		int blockBranch1;
		boolean branch1Valid;
		int blockBranch2;
		boolean branch2Valid;
		
		public BlockData() {
			basicBlock = null;
			blockBranch1 = blockBranch2 = -1;
			branch1Valid = branch2Valid = false;
		}
		
		public BlockData(BasicBlock block) {
			basicBlock = block;
			blockBranch1 = blockBranch2 = -1;
			branch1Valid = branch2Valid = false;
		}
		
		public boolean areBothBranchesInvalid() {
			if (!isBranch1Valid() && !isBranch2Valid())
				return true;
			return false;
		}
		
		public boolean isAnyBranchValid() {
			if (isBranch1Valid() || isBranch2Valid())
				return true;
			return false;
		}
		
		public boolean isAnyBranchInvalid() {
			if (!isBranch1Valid() || !isBranch2Valid())
				return true;
			return false;
		}
		
		public void setBranch(int block) {
			if (!isBranch1Valid())
				setBranch1(block);
			else if (!isBranch2Valid())
				setBranch2(block);
			else
				System.out.println("CFG: Something wrong in branching.");
		}
		
		public void setBlock(BasicBlock block) {
			basicBlock = block;
		}
		
		public BasicBlock getBlock() {
			return basicBlock;
		}
		
		public void setBranch1(int block) {
			blockBranch1 = block;
			branch1Valid = true;
		}
		
		public void setBranch2(int block) {
			blockBranch2 = block;
			branch2Valid = true;
		}
		
		public boolean isBranch1Valid() { return branch1Valid; }
		public boolean isBranch2Valid() { return branch2Valid; }
		public void clearBranch1() { branch1Valid = false; }
		public void clearBranch2() { branch2Valid = false; }
	}
	
}