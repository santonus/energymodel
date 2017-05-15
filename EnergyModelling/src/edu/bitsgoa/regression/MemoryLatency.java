package edu.bitsgoa.regression;
/**
 * This class calculated memory latency data is absent, then fits an Akima spline through it. Memory latency data is computed by running memoryLatency.out.
 * For fitting a spline through the data, it uses the Apache Commons Math library.
 */

import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;

import org.apache.commons.math3.analysis.interpolation.AkimaSplineInterpolator;
import org.apache.commons.math3.analysis.polynomials.PolynomialSplineFunction;

import edu.bitsgoa.driver.Preparation;
import edu.bitsgoa.startup.StartCheck;
import edu.bitsgoa.utilities.BetterRunProcess;
import edu.bitsgoa.views.DisplayCustomConsole;


public class MemoryLatency implements Runnable {
	static BufferedReader br;
	static InputStream stdin;
	static InputStreamReader isr;
	static FileWriter fw;
	static String path_result;	//path where all the files, including memLatency.txt are stored.
	//psf is the final object returned after fitting a spline. Using psf, memory latency for any number of threads can be found
	public static PolynomialSplineFunction psf;	

	/**If memory latency data does not exist, then this method calculates it first. Otherwise, it just fits an Akima spline through it
	 * @param
	 * @return
	 */
	public static void runExecutable(){
		boolean memLatencyData=new File(Preparation.path_home,"memLatency.txt").exists();	//check if memLatency.txt exists at path_result.
		path_result=Preparation.path_home+"/memLatency.txt";
		if(!memLatencyData){	
			//memLatency.txt does not exist. Hence, calculate it first.
			DisplayCustomConsole.display("Gathering Memory Latency Data...",false);
			String path_executable=StartCheck.path_memorylatencyExec.substring(StartCheck.path_memorylatencyExec.indexOf('/'),StartCheck.path_memorylatencyExec.lastIndexOf('/')+1);
			String[] cmd=new String[1];
			cmd[0]="./memLatency.out";
			BetterRunProcess process=new BetterRunProcess();
			process.runProcessBuilderInDifferentDirectory(cmd, path_executable,0,1,0,"memLatency.txt");
			DisplayCustomConsole.display("Done",true);
			fitCurve();
		}
		else fitCurve();

	}
	/**
	 * Fits an Akima spline through memLatency.txt
	 * @param
	 * @return
	 */
	public static void fitCurve(){
		DisplayCustomConsole.display("Calculating Memory Latency Model...",false);
		int lineNo=0;
		String line;
		double x,y;
		String[] arr=new String[2];

		try {
			BufferedReader buf=new BufferedReader(new FileReader(path_result));
			while((line=buf.readLine())!=null){
				lineNo+=1;
			}
			double[] X=new double[lineNo];
			double[] Y=new double[lineNo];
			lineNo=0;
			buf.close();
			buf=new BufferedReader(new FileReader(path_result));
			while((line=buf.readLine())!=null){
				arr=line.split("\\s+");
				x=Double.parseDouble(arr[0]);
				y=Double.parseDouble(arr[1]);
				X[lineNo]=x;
				Y[lineNo]=y;
				lineNo+=1;
			}
			AkimaSplineInterpolator asi = new AkimaSplineInterpolator();
			psf = asi.interpolate(X,Y);
			DisplayCustomConsole.display("Done",true);
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		} 

	}
	@Override
	public void run() {
		runExecutable();		
	}
}
