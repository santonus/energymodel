package edu.bitsgoa.utilities;
/**
 * This class allows execution of external processes (for example an exe or a shell script). 
 * @author santonu_sarkar
 *
 */
import java.io.BufferedReader;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.Writer;
import edu.bitsgoa.views.DisplayCustomConsole;

public class BetterRunProcess {
	class ConsumeStreamOp extends Thread	{
		private InputStream _is;
		private String _type;
		private Writer _opw;

		ConsumeStreamOp(InputStream is, String type) {
			this(is, type, null);
		}
		ConsumeStreamOp(InputStream is, String type, Writer redirect) {
			_is = is;
			_type = type;
			_opw = redirect;
		}
		public final String toString() {
			return (new String("Stream Type:" + _type)); 
		}
		public void run() {
			try {
				InputStreamReader isr = new InputStreamReader(_is);
				BufferedReader br = new BufferedReader(isr);
				String line=null;
				while ( (line = br.readLine()) != null) {
					if (_opw != null)
						_opw.write(line + "\n");
					else
						DisplayCustomConsole.display(line,true);    
				}
				if (_opw != null)
					_opw.flush();
			} catch (IOException ioe) {
				ioe.printStackTrace();  
			}
		}
		public final Writer getOutput() {
			return _opw;
		}
	}
	private String _err;
	private String _oup;

	/**
	 * 
	 * @param cmd a string array containing the command
	 * @param path	path from where to run the process. This is the path where the executable is present
	 * @param printToConsole	1 if the results of running the process have to be written to the custom console
	 * @param printToExternalFile	1 if the results of running the process have to be written to an external text file
	 * @param append	1 if the text file already exists, and results have to be appended to it
	 * @param fileName	name of the text file in which to write data. Pass as "" if you do not want to write to an external file
	 * @return
	 */
	public void runProcessBuilderInDifferentDirectory(String[] cmd,String path,int printToConsole,int printToExternalFile,int append,String fileName){
		ProcessBuilder builder;
		if(cmd.length==1){
			builder=new ProcessBuilder(cmd[0]);
		}
		else if(cmd.length==2){
			builder=new ProcessBuilder(cmd[0],cmd[1]);
		}
		else if(cmd.length==3){
			builder=new ProcessBuilder(cmd[0],cmd[1],cmd[2]);
		}
		else if(cmd.length==4){
			builder=new ProcessBuilder(cmd[0],cmd[1],cmd[2],cmd[3]);
		}
		else{
			builder=new ProcessBuilder(cmd[0],cmd[1],cmd[2],cmd[3],cmd[4]);
		}
		builder.directory(new File(path));
		try {
			Process pr=builder.start();
			if(printToConsole==1) printToConsole(pr);
			if(printToExternalFile==1) printToExternalFile(pr,fileName,append);
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
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
	public final String getError() {
		return _err;
	}

	public final String getOutput() {
		return _oup;
	}
}