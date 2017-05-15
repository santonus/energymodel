package edu.bitsgoa.utilities;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStreamReader;

import javax.swing.JOptionPane;

import org.eclipse.swt.widgets.Display;
import edu.bitsgoa.views.CustomConsole;
import edu.bitsgoa.views.DisplayCustomConsole;

public class BetterRunProcessThreaded implements Runnable {
	String[] cmd;
	String path;
	int printToConsole;
	int printToExternalFile;
	int append;
	String filename;
	String startMsg;
	boolean changeLine;
	String endMsg;
	
	public BetterRunProcessThreaded(String[] cmd,String path,int printToConsole,int printToExternalFile,int append,String filename,String startMsg,boolean changeLine,String endMsg){
		this.cmd=cmd;
		this.path=path;
		this.printToConsole=printToConsole;
		this.printToExternalFile=printToExternalFile;
		this.append=append;
		this.filename=filename;
		this.startMsg=startMsg;
		this.changeLine=changeLine;
		this.endMsg=endMsg;
	}
	@Override
	public void run() {
		if(startMsg!=null){
			Display.getDefault().asyncExec(new Runnable() {
				@Override
				public void run() {
					if(CustomConsole.used==0) return;	//If custom console has not been opened, then there is no point in displaying on it. Simply, return.
					CustomConsole.text.append(startMsg);
					if(changeLine)
						CustomConsole.text.append("\n");
				}
			});
		}
		
		ProcessBuilder builder;
		if(cmd.length==1)	builder=new ProcessBuilder(cmd[0]);
		else if(cmd.length==2)	builder=new ProcessBuilder(cmd[0],cmd[1]);
		else if(cmd.length==3)	builder=new ProcessBuilder(cmd[0],cmd[1],cmd[2]);
		else if(cmd.length==4)	builder=new ProcessBuilder(cmd[0],cmd[1],cmd[2],cmd[3]);
		else	builder=new ProcessBuilder(cmd[0],cmd[1],cmd[2],cmd[3],cmd[4]);
		builder.directory(new File(path));
		try {
			Process pr=builder.start();
			if(printToConsole==1) printToConsole(pr);
			if(printToExternalFile==1) printToExternalFile(pr,filename,append);
		} catch (IOException e) {
			e.printStackTrace();
		}
		if(endMsg!=null){
			Display.getDefault().asyncExec(new Runnable() {
				@Override
				public void run() {
					if(CustomConsole.used==0) return;	//If custom console has not been opened, then there is no point in displaying on it. Simply, return.
					CustomConsole.text.append(endMsg);
					CustomConsole.text.append("\n");
				}
			});
		
		}
	}
	/**
	 * Writes the result of running a process to the custom console using DisplayCustomConsole.display()
	 * @param pr a process object as Process proc=processBuilderObject.start();
	 * @return
	 */
	public void printToConsole(Process pr){
		BufferedReader br_output=new BufferedReader(new InputStreamReader(pr.getInputStream()));
		BufferedReader br_err=new BufferedReader(new InputStreamReader(pr.getErrorStream()));
		String line;
		try {
			while((line=br_output.readLine())!=null){
				DisplayCustomConsole.display(line+"\n",true);
			}
			br_output.close();
			while((line=br_err.readLine())!=null){
				DisplayCustomConsole.display(line+"\n",true);
			}
			br_err.close();
			br_err.close();
		} catch (IOException e) {
			e.printStackTrace();
		}
	}
	/**
	 * Writes the results of running a process to external text file 
	 * @param proc	a process object as Process proc=processBuilderObject.start();
	 * @param filename	name of the file in which to write data
	 * @param append	1 if the file already exists, then just append data to it. If 0, then create a new text file and write to it
	 * @return
	 */
	public void printToExternalFile(Process proc,String filename,int append){
		String path="/home/"+System.getProperty("user.name")+"/.eclipse"+"/models/"+filename;
		BufferedReader br_output = new BufferedReader(new InputStreamReader(proc.getInputStream()));
		BufferedReader br_err=new BufferedReader(new InputStreamReader(proc.getErrorStream()));
		String line;
		try {
			if(append==0){	//implies that a new file is to be created and written into
				FileWriter fw = new FileWriter(path,false);
				while ( (line = br_output.readLine()) != null){
					fw.write(line+"\n");
				}
				while((line=br_err.readLine())!=null){
					fw.write(line);
				}
				fw.close();
			}
			else{	//implies that data has to be written into an existing file
				FileWriter fw = new FileWriter(path,true);
				while ( (line = br_output.readLine()) != null){
					fw.write(line+"\n");
				}
				while((line=br_err.readLine())!=null){
					fw.write(line);
				}
				fw.close();
			}
		} catch (IOException e) {
			e.printStackTrace();
		}
	}

}
