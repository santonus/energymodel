package edu.bitsgoa.programAnalyzer;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.PrintWriter;
import java.util.Scanner;

import edu.bitsgoa.driver.ConfigureModel;
import edu.bitsgoa.programAnalyzer.instructiontypes.EnumAllInsts;
import edu.bitsgoa.programAnalyzer.singlekernel.SingleKernel;
import edu.bitsgoa.utilities.UtilsMisc;
public class InputProg {
	
	private int foundNumberOfKernels;	
	private SingleKernel[] kernelList;
	
	public void loadComputePrint() {
		loadPTX();
		for (int i = 0; i < foundNumberOfKernels; i++) {
			kernelList[i].constructBasicBlocks();
			kernelList[i].countInstsPerBlock();
			kernelList[i].constructAllDFGs();
			kernelList[i].constructControlFlowAndCount();
		}
		printToFile();
	}
	
	public void loadPTX() {
		String fileName = ConfigureModel.inputProgPath + ConfigureModel.inputProgPTX;
		Scanner reader = null;
		try {
			reader = new Scanner(new File(fileName));
		} catch (FileNotFoundException e) {
			System.out.println("File " + fileName + " not found.");
			return;
		}
		
		String line = null;
		EnumAllInsts instType = null;
		boolean insideKernel = false, kernelDetected = false;
		String kernelName = null;
		int currentKernel = 0;
		int countNonKernelBraces = 0;
		foundNumberOfKernels = 0;
		kernelList = new SingleKernel[ConfigureModel.maxNumberOfKernels];
		for (int i = 0; i < ConfigureModel.maxNumberOfKernels; i++) {
			kernelList[i] = new SingleKernel(ConfigureModel.numBlocks, ConfigureModel.numThreadsPerBlock);
		}
		while (reader.hasNextLine()) {
			line = reader.nextLine().trim();
			instType = PTXUtil.decodeInstruction(line);
			if (line.contains("entry")) { //TODO: Modify for clock, qrgQRNG
				kernelDetected = true;
				kernelName = line.substring(line.indexOf('_'), UtilsMisc.maxInt(line.indexOf('('), line.length() - 1));
			}
			if (PTXUtil.canBeSkipped(line))
				continue;
			else if (instType == EnumAllInsts.KernelStart && kernelDetected) {
				foundNumberOfKernels++;
				insideKernel = true;
				kernelDetected = false;
				kernelList[currentKernel].setKernelNameAndNumber(kernelName, currentKernel);
			}
			else if (instType == EnumAllInsts.KernelStart && insideKernel) {
				// required to handle open braces that are not actually indicative of a kernel start
				countNonKernelBraces++;
			}
			else if (instType == EnumAllInsts.KernelEnd && insideKernel && countNonKernelBraces > 0) {
				countNonKernelBraces--;
			}
			else if (instType == EnumAllInsts.KernelEnd && insideKernel && countNonKernelBraces == 0) {
				insideKernel = false;
				kernelDetected = false;
				currentKernel++;
			}
			else if (insideKernel) {
				kernelList[currentKernel].putInstruction(line);
			}
		}
		
		if (reader != null) {
			reader.close();
			reader = null;
		}
	}
	
	public void printToFile() {
		String path = ConfigureModel.inputProgPath + ConfigureModel.outputFileName;
		PrintWriter writer = null;
		try {
			writer = new PrintWriter(new File(path), "UTF-8");
		} catch (Exception e) {
			System.out.println("Error printing output to file.");
			if (writer != null)
				writer.close();
			return;
		}
		for (int i = 0; i < foundNumberOfKernels; i++) {
			kernelList[i].getDetailsAsStrings();
			
			writer.write(kernelList[i].kernelNumString);
			
			writer.write(kernelList[i].allInstsString);
			writer.write(UtilsMisc.dashedLine);
			
			writer.write(kernelList[i].perBlockCountString);
			writer.write(UtilsMisc.smallerLine);
			
			writer.write(kernelList[i].controlFlowString);
			writer.write(UtilsMisc.smallerLine);
			
			writer.write(kernelList[i].instsExecCountString);
			writer.write(UtilsMisc.smallerLine);
			
			writer.write(kernelList[i].execTimeString);
			
			writer.write(UtilsMisc.dashedLine);
			
			writer.write("\n\nBASIC BLOCKS\n\n");
			for (int j = 0; j < kernelList[i].getNumBasicBlocks(); j++) {
				writer.write(kernelList[i].getBasicBlockFromList(j).getBasicBlockAsString());
				writer.write(kernelList[i].getBasicBlockFromList(j).getDFGDetailsAsString
						(kernelList[i].getKernelNumThreadsPerBlock(), kernelList[i].getKernelNumBlocks()));
				writer.write(kernelList[i].getBasicBlockFromList(j).getInstCountsOfBlockAsString());
				writer.write(UtilsMisc.dashedLine);
			}
		}
		
		if (writer != null)
			writer.close();
	}
	
	public SingleKernel[] getKernelList() {
		return kernelList;
	}
	
	public SingleKernel getKernelWithNumber(int i) {
		if (i < foundNumberOfKernels)
			return kernelList[i];
		return null;
	}
	
	public int getFoundNumberOfKernels() {
		return foundNumberOfKernels;
	}
	
	public InputProg() {
		foundNumberOfKernels = 0;
	}
	
}