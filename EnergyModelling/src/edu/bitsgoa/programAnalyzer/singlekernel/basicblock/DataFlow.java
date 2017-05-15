package edu.bitsgoa.programAnalyzer.singlekernel.basicblock;

import edu.bitsgoa.driver.ConfigureModel;
import edu.bitsgoa.programAnalyzer.PTXUtil;
import edu.bitsgoa.programAnalyzer.instructiontypes.EnumAllInsts;

public class DataFlow {
	
	private int maxNodesDfg;
	private int numDfgNodes;
	private DFGNode[] dfgNodesList;
	
	public void createAndAddDFGNode(String instruction) {
		DFGNode newNode = new DFGNode();
		newNode.constructNode(numDfgNodes, instruction);
		dfgNodesList[numDfgNodes] = newNode;
		numDfgNodes++;
	}
	
	public double calculateILP(double numInstructionsTotal) {
		
		double ILP = 1.0;
		int totalParallelInstCycles = 0, i, j;
		double effectiveInsts = numInstructionsTotal;
		DFGNode node = null;
		boolean breakFlag = false;
		
		i = 0;
		while (i < numDfgNodes) {
			
			if (PTXUtil.decodeInstruction(dfgNodesList[i].getStrInst()) == EnumAllInsts.Label) {
				effectiveInsts--;
				i += 1;
				continue;
			}
			
			breakFlag = false;
			for (j = 0; j < ConfigureModel.numIndependentInsts - 1 && (i + j + 1) < numDfgNodes; j++) {
				node = dfgNodesList[i + j + 1];
				if (!node.getIndependence()) {
					breakFlag = true;
					break;
				}
			}
			if (breakFlag)
				i += j + 1;
			else if (!breakFlag && j != ConfigureModel.numIndependentInsts - 1)
				i += 1;
			else
				i += ConfigureModel.numIndependentInsts;
			
			totalParallelInstCycles++;
			
		}
		
		ILP = 1.0 * effectiveInsts / totalParallelInstCycles;
		return ILP;
	}
	
	public void constructDependencies() {
		int i, j;
		DFGNode curNode = null, compareNode = null;
		for (i = 0; i < numDfgNodes; i++) {
			curNode = dfgNodesList[i];
			for (j = i - 1; j >= 0; j--) {
				compareNode = dfgNodesList[j];
				if (PTXUtil.isControlInst(curNode.getStrInst()) || PTXUtil.isControlInst(compareNode.getStrInst())
						|| !areIndependent(curNode, compareNode)) {
					addDependency(curNode, compareNode);
				}
			}
		}
		
		boolean flag = false;
		for (i = 1; i < numDfgNodes; i++) { // exclude the first instruction
			curNode = dfgNodesList[i];
			flag = false;
			for (j = i - 1; j >= i - 1 - ConfigureModel.numIndependentInsts && j >= 0; j--) {
				compareNode = dfgNodesList[j];
				if (PTXUtil.isControlInst(curNode.getStrInst()) || PTXUtil.isControlInst(compareNode.getStrInst())
						|| !areIndependent(curNode, compareNode)) {
					flag = true;
					break;
				}
			}
			curNode.setIndependence(!flag);
		}
	}
	
	private boolean areIndependent(DFGNode node1, DFGNode node2) {
		if (node1 == null || node2 == null || !node1.isValid() || !node2.isValid())
			return true;
		String regName = "";
		for (int i = 0; i < node2.getNumRegs(); i++) {
			regName = node2.getRegName(i);
			if (skipDuringIndependenceCheck(regName))
				continue;
			if (node1.checkIfInRegisterList(regName))
				return false;
		}
		return true;
	}
	
	private void addDependency(DFGNode fromNode, DFGNode toNode) {
		if (fromNode == null || toNode == null || !fromNode.isValid() || !toNode.isValid())
			return;
		fromNode.addToPrevNodes(toNode.getNodeInstNum());
		toNode.addToNextNodes(fromNode.getNodeInstNum());
	}
	
	private boolean skipDuringIndependenceCheck(String regName) {
		if (regName == null || regName.equals("") ||
				regName.contains("ctaid") || regName.contains("tid"))
			return true;
		return false;
	}
	
	public DFGNode getNodeAtIndex(int index) {
		if (index >= numDfgNodes)
			return null;
		return dfgNodesList[index];
	}
	
	public DFGNode getNodeWithInstNum(int num) {
		for (int i = 0; i < numDfgNodes; i++) {
			if (dfgNodesList[i].getNodeInstNum() == num)
				return dfgNodesList[i];
		}
		return null;
	}
	
	public int getNumDfgNodes() {
		return numDfgNodes;
	}
	
	public DataFlow(int maxNodes) {
		numDfgNodes = 0;
		this.maxNodesDfg = maxNodes;
		dfgNodesList = new DFGNode[maxNodesDfg];
		for (int i = 0; i < maxNodesDfg; i++) {
			dfgNodesList[i] = new DFGNode();
		}
	}
	
}