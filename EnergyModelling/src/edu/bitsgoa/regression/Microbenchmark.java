package edu.bitsgoa.regression;

import edu.bitsgoa.driver.Preparation;
import edu.bitsgoa.startup.StartCheck;
import edu.bitsgoa.utilities.BetterRunProcess;
import edu.bitsgoa.views.DisplayCustomConsole;

public class Microbenchmark implements Runnable {
	boolean[] existenceArray;
	
	public Microbenchmark(boolean[] existenceArray){
		this.existenceArray=existenceArray;
	}
	public void run(){
		calculateMicrobencmarksLatencies();
	}
	public static void runInstructions(String cmd,String path_executables,String fileName){
		String[] command=new String[1];
		BetterRunProcess process=new BetterRunProcess();
		command[0]=cmd;
		for(int i=0;i<50;i++){
			if(i==0)	process.runProcessBuilderInDifferentDirectory(command,path_executables,0,1,0,fileName);
			else	process.runProcessBuilderInDifferentDirectory(command,path_executables,0,1,1,fileName);
		}
		maxFreqCommandRun(fileName);	//find the most frequently occurring latency
	}
	public void calculateMicrobencmarksLatencies(){
		DisplayCustomConsole.display("Running microbenchmarks for latencies...",false);
		String path_executables=StartCheck.path_devicequeryExec.substring(StartCheck.path_devicequeryExec.indexOf('/'),StartCheck.path_devicequeryExec.lastIndexOf('/'))+"/";
		
		if(!existenceArray[11]){
			if(!existenceArray[3])	runInstructions("./instLatency_addf32", path_executables,"times_addf32.txt");
			else	maxFreqCommandRun("times_addf32.txt");
		}
		
		if(!existenceArray[12]){
			if(!existenceArray[4])	runInstructions("./instLatency_andb32", path_executables,"times_andb32.txt");
			else	maxFreqCommandRun("times_andb32.txt");
		}
		
		if(!existenceArray[13]){
			if(!existenceArray[5])	runInstructions("./instLatency_divf32", path_executables,"times_divf32.txt");
			else	maxFreqCommandRun("times_divf32.txt");
		}
	
		if(!existenceArray[14]){
			if(!existenceArray[6])	runInstructions("./instLatency_divs32", path_executables,"times_divs32.txt");
			else	maxFreqCommandRun("times_divs32.txt");
		}
		
		if(!existenceArray[15]){
			if(!existenceArray[7])	runInstructions("./instLatency_mad", path_executables,"times_mad.txt");
			else	maxFreqCommandRun("times_mad.txt");
		}
	
		if(!existenceArray[16]){
			if(!existenceArray[8])	runInstructions("./instLatency_mulf32", path_executables,"times_mulf32.txt");
			else	maxFreqCommandRun("times_mulf32.txt");
		}
		
		if(!existenceArray[17]){
			if(!existenceArray[9])	runInstructions("./instLatency_sqrtcvt", path_executables,"times_sqrtcvt.txt");
			else	maxFreqCommandRun("times_sqrtcvt.txt");
		}

		if(!existenceArray[18]){
			if(!existenceArray[10])	runInstructions("./instLatency_subs32", path_executables,"times_subs32.txt");
			else	maxFreqCommandRun("times_subs32.txt");
		}
		DisplayCustomConsole.display("Done",true);
	}
	public static void maxFreqCommandRun(String fileName){
		String[] cmd=new String[3];	//cmd array contains the command to run
		BetterRunProcess process=new BetterRunProcess();
		cmd[0]="/bin/sh";
		cmd[1]="-c";
		cmd[2]="sort "+fileName+"| uniq -c | sort -r | head -1 | xargs";
		process.runProcessBuilderInDifferentDirectory(cmd,Preparation.path_home,0,1,0,"av_"+fileName);
	}


}
