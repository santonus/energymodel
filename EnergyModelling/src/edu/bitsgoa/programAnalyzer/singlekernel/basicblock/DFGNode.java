package edu.bitsgoa.programAnalyzer.singlekernel.basicblock;

import java.util.ArrayList;

import edu.bitsgoa.driver.ConfigureModel;
import edu.bitsgoa.programAnalyzer.PTXUtil;
import edu.bitsgoa.programAnalyzer.instructiontypes.EnumAllInsts;
import edu.bitsgoa.utilities.UtilsMisc;



public class DFGNode {

	private int nodeInstNumber;
	private EnumAllInsts instType;
	private String instruction;
	
	private String[] registers;
	private int numRegs;
	
	private ArrayList<Integer> prevNodes;
	private ArrayList<Integer> nextNodes;
	
	private boolean valid;
	private boolean independent;
	
	public boolean getIndependence() {
		return independent;
	}
	
	public void setIndependence(boolean val) {
		independent = val;
	}
	
	public String independenceMarker() {
		if (independent)
			return UtilsMisc.independentStr;
		return "\n";
	}
	
	public boolean checkIfInRegisterList(String regName) {
		for (int i = 0; i < numRegs; i++) {
			if (registers[i].equals(regName))
				return true;
		}
		return false;
	}
	
	private void makeRegList() {
		if (instruction == null)
			return;
		int start = 0, end = 0, i;
		char curChar = ' '; String regTemp = "";
		boolean insideRegName = false;
		
		numRegs = 0; i = 0;
		while (i < instruction.length()) {
			curChar = instruction.charAt(i);
			if (curChar == '%') {
				start = i + 1;
				insideRegName = true;
			}
			else if (insideRegName && !UtilsMisc.isNumberOrLetter(curChar)) {
				end = i;
				insideRegName = false;
				regTemp = instruction.substring(start, end);
				registers[numRegs] = regTemp;
				numRegs++;
			}
			i++;
		}
	}
	
	public void constructNode(int instNum, String inst) {
		setNodeInstNum(instNum);
		setStrInst(inst);
		makeRegList();
		valid = true;
	}
	
	public void setNodeInstNum(int num) {
		this.nodeInstNumber = num;
	}
	
	public void setStrInst(String inst) {
		this.instruction = inst;
		instType = PTXUtil.decodeInstruction(instruction);
	}
	
	public void addToPrevNodes(int num) {
		if (!prevNodes.contains(num))
			prevNodes.add(num);
	}
	
	public void addToNextNodes(int num) {
		if (!nextNodes.contains(num))
			nextNodes.add(num);
	}
	
	public int getNodeInstNum() {
		return nodeInstNumber;
	}
	
	public String getStrInst() {
		return instruction;
	}
	
	public EnumAllInsts getInstType() {
		return instType;
	}
	
	public ArrayList<Integer> getPrevNodesList() {
		return prevNodes;
	}
	
	public ArrayList<Integer> getNextNodesList() {
		return nextNodes;
	}
	
	public String getRegName(int index) {
		if (index >= numRegs)
			return "";
		return registers[index];
	}
	
	public String getRegListAsString() {
		StringBuilder resultString = new StringBuilder();
		for (int i = 0; i < numRegs; i++) {
			resultString.append(getRegName(i) + " ");
		}
		return resultString.toString();
	}
	
	public int getNumRegs() {
		return numRegs;
	}
	
	public boolean isValid() {
		return valid;
	}
	
	public DFGNode() {
		nodeInstNumber = 0;
		instruction = "";
		instType = null;
		numRegs = 0;
		independent = false;
		prevNodes = new ArrayList<Integer>();
		nextNodes = new ArrayList<Integer>();
		registers = new String[ConfigureModel.maxRegistersPerInstruction];
		for (int i = 0; i < ConfigureModel.maxRegistersPerInstruction; i++) {
			registers[i] = "";
		}
		valid = false;
	}
	
}