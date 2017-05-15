package edu.bitsgoa.driver;

import edu.bitsgoa.exectimeestimate.ComputeExecTime;
import edu.bitsgoa.exectimeestimate.ComputeExecTimeModified;
import edu.bitsgoa.exectimeestimate.ComputeOverhead;
import edu.bitsgoa.powerModeller.PowerPredictor;
import edu.bitsgoa.programAnalyzer.InputProg;
import edu.bitsgoa.programAnalyzer.instructiontypes.AllInstData;
import edu.bitsgoa.programAnalyzer.singlekernel.SingleKernel;
import edu.bitsgoa.views.DisplayCustomConsole;

public class MainRunner {

	private static ComputeOverhead computeOverhead;
	@SuppressWarnings("unused")
	private static ComputeExecTimeModified computeExecModified;
	private static ComputeExecTime computeExecTime;
	private static PowerPredictor powerPredict;
	private static InputProg inputProg;

	public static void initializeAll() {
		ConfigureModel.getPropValues();
		AllInstData.initializeAll();
		computeOverhead = new ComputeOverhead();
		computeExecModified = new ComputeExecTimeModified();
		computeExecTime = new ComputeExecTime();
		powerPredict = new PowerPredictor();
		inputProg = new InputProg();
	}

	public static void runModel() {
		inputProg.loadComputePrint();
		computeOverhead.calculate(ConfigureModel.transferSize);
		SingleKernel curKernel = null;
		for (int i = 0; i < inputProg.getFoundNumberOfKernels(); i++) {
			curKernel = inputProg.getKernelWithNumber(i);
			computeExecTime.calculate(curKernel);
			DisplayCustomConsole.display("Kernel " + curKernel.getKernelNumber() + "\t\tName: " + curKernel.getKernelName(),true); 
			DisplayCustomConsole.display("Predicted exec time:\t" + computeExecTime.getExecTimeTotal() + " ms",true);
			DisplayCustomConsole.display("Predicted power:\t\t" + powerPredict.getCalculatedPower(ConfigureModel.achievedOccupancy) + " W",true);
			DisplayCustomConsole.display("Predicted energy:\t\t" + computeExecTime.getExecTimeTotal() * powerPredict.getCalculatedPower(ConfigureModel.achievedOccupancy) + " mJ\n",true);
		}	
	}

	/*
	 * Any TLP related errors = check ComputeExecTime and calculateParallelism() in BasicBlock
	 */
}