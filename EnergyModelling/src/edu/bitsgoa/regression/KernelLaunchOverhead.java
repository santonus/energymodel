package edu.bitsgoa.regression;
/**
 * This class finds the kernel launch overhead parameter for the user's hardware. This is done by running "empty" executable with different number
 * of threads as arguments. Then, we fit a straight line through the data using linear regression.
 */

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import edu.bitsgoa.driver.Preparation;
import edu.bitsgoa.startup.StartCheck;
import edu.bitsgoa.utilities.LinearRegressionModel;
import edu.bitsgoa.utilities.RegressionModel;
import edu.bitsgoa.views.DisplayCustomConsole;

public class KernelLaunchOverhead implements Runnable {
	private static int[] arr;
	private static BufferedReader br;
	private static InputStream stdin;
	private static InputStreamReader isr;
	private static FileWriter fw;
	private static String path_result;
	private static double[] x;
	private static double[] y;
	private static int lineNo=0;
	private static double[] coefficients;
	private static int arrayFilled=0;
	
	int bvar1;
	int bvar2;
	
	public KernelLaunchOverhead(int bvar1,int bvar2){
		this.bvar1=bvar1;
		this.bvar2=bvar2;
	}

	/**
	 * Generate kernel launch overhead data and its regression model
	 * @param bvar1	bvar=1 implies KernelLaunchOverhead.txt exists. Then, if it is absent, then we only need to find KernelLaunchOverheadModel.txt.
	 * @param bvar2	bvar=0 implies KernelLaunchOverheadModel.txt is absent, and needs to be calculated
	 * @return 
	 */
	public static void runExecutable(int bvar1,int bvar2){
		initializeArray();
		path_result=Preparation.path_home+"/kernelLaunchOverhead.txt";
		if(bvar1==0){	//if KernelLaunchOverhead data does not exist
			DisplayCustomConsole.display("Generating Kernel Launch Overhead Data...",false);
			String path_executable=StartCheck.path_kernellaunchoverheadExec.substring(StartCheck.path_kernellaunchoverheadExec.indexOf('/'),StartCheck.path_kernellaunchoverheadExec.lastIndexOf('/')+1);
			try {
				fw = new FileWriter(path_result);
				for(int i=0;i<arr.length;i++){
					ProcessBuilder builder=new ProcessBuilder("./empty",Integer.toString(arr[i]));
					builder.directory(new File(path_executable));
					int av=0;
					float sum=0;
					//For a particular number of threads, run "empty" 10 times, and take the average. Write that average to an external text file
					while(av<10){
						Process pr=builder.start();
						stdin = pr.getInputStream();
						isr = new InputStreamReader(stdin);
						br = new BufferedReader(isr);
						sum=sum+Float.parseFloat(br.readLine());
						av++;
					}
					//write to KernelLaunchOverhead.txt stored at user.home/.eclipse/models
					fw.write(arr[i]+"	"+Float.toString(sum/10));
					fw.write("\n");
				}
				fw.close();

			} catch (IOException e1) {
				// TODO Auto-generated catch block
				e1.printStackTrace();
			}
			DisplayCustomConsole.display("Done",true);
			//Populate the arrays. These arrays will be used as input for generating the regression model
			FillArrays(path_result);
			//Fit a best-fit line
			BestLineFit();
			//Write the model to an external file
			saveModel(Preparation.path_home+"/KernelLaunchOverheadModel.txt");
		}
		if(bvar2==0 && bvar1==1){	//if only KernelLaunchOverheadModel.txt is absent, then calculate it
			if(arrayFilled==0)	FillArrays(path_result);
			BestLineFit();
			saveModel(Preparation.path_home+"/KernelLaunchOverheadModel.txt");
		}
	}
	/**
	 * Fill two arrays. One array holds the number of threads. Other array holds the corresponding kernel launch overhead
	 * @param path_result path to memLatency.txt. memLatency.txt contains the memory latency data
	 * return
	 */
	public static void FillArrays(String path_result){
		arrayFilled=1;
		String[] arr=new String[2];
		try (BufferedReader br = new BufferedReader(new FileReader(path_result))) {
			String line;
			while ((line = br.readLine()) != null) {
				arr=line.split("\\s+");
				x[lineNo]=Double.parseDouble(arr[0]);
				y[lineNo]=Double.parseDouble(arr[1]);
				lineNo++;
			}

		} catch (IOException e) {
			e.printStackTrace();
		}
	}	
	/**
	 * This method is used to fit a best-fit line
	 * @param	
	 * @return
	 */
	public static void BestLineFit(){
		DisplayCustomConsole.display("Generating Kernel Launch Overhead Model...",false);
		RegressionModel model=new LinearRegressionModel(x, y);
		model.compute();
		coefficients=model.getCoefficients();
		DisplayCustomConsole.display("Done",true);
	}
	/**
	 * This method is used to write the generated model to an external file
	 * @param pathResult path to the file into which the model is written
	 * @return
	 */
	public static void saveModel(String pathResult){
		try {
			fw=new FileWriter(pathResult);
			fw.write(Double.toString(coefficients[0])+"\n");
			fw.write(Double.toString(coefficients[1]));
			fw.close();
		} catch (IOException e) {
			e.printStackTrace();
		}
	}
	/**
	 * This method is used to  initialize variables
	 * @param
	 * @return
	 */
	public static void initializeArray(){
		coefficients=new double[2];
		x=new double[20];
		y=new double[20];
		arr=new int[20];
		arr[0]=1;
		arr[1]=32;
		arr[2]=1024;
		for(int i=2;i<=19;i++){
			arr[i]=2*arr[i-1];
		}
	}
	@Override
	public void run() {
		runExecutable(bvar1, bvar2);
	}
}
