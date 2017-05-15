/**
 * This class checks for the existence of certain files, and if absent, generates them by running executables using the BetterRunProcess class.
 *Micro-benchmarking of user's hardware is also done in this class. This class is a handler class for the menu command "Prepare". That is, when
 *the user click on "Prepare" while using the plug-in, this class is called.
 */
package edu.bitsgoa.driver;

import java.io.File;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.Executor;
import java.util.concurrent.Executors;

import javax.swing.JOptionPane;

import org.eclipse.core.commands.AbstractHandler;
import org.eclipse.core.commands.ExecutionEvent;
import org.eclipse.core.commands.ExecutionException;
import org.eclipse.core.commands.IHandler;
import edu.bitsgoa.properties.Parameters;
import edu.bitsgoa.regression.KernelLaunchOverhead;
import edu.bitsgoa.regression.MemoryLatency;
import edu.bitsgoa.regression.Microbenchmark;
import edu.bitsgoa.startup.StartCheck;
import edu.bitsgoa.utilities.BetterRunProcess;
import edu.bitsgoa.views.DisplayCustomConsole;

public class Preparation extends AbstractHandler implements IHandler {

	public static boolean return_val=false;	//enable or disable handler
	private static boolean[] existenceArray;	//true is the corresponding file is present, false otherwise
	public static String path_home="/home/"+System.getProperty("user.name")+"/.eclipse"+"/models/";	//path to where all the files are generated
	private static int bvar1=0;	//variable used in KernelLaunchOverhead.java
	private static int bvar2=0;	//variable used in KernelLaunchOverhead.java
	public static int used=0;	//1 if this class has been used at least once
	private static BetterRunProcess process;
	
	public Object execute(ExecutionEvent event) throws ExecutionException {
		String path_executables=StartCheck.path_devicequeryExec.substring(StartCheck.path_devicequeryExec.indexOf('/'),StartCheck.path_devicequeryExec.lastIndexOf('/'))+"/";
		used=1;	//Handler fired, turn used=1
		mkdir();	//make a new directory at path_home
		initialize();	//initialize variables. The method filesExist() is called from initialize()
		String[] cmd=new String[1];	//this array stores the command to run using BetterRunProcess.java
		Executor executor=Executors.newSingleThreadExecutor();
		try{
			if(!existenceArray[1]){	//implies deviceQuery.txt is not present at path_home. So, generate the file
				cmd[0]="./deviceQuery.out";
				edu.bitsgoa.utilities.BetterRunProcessThreaded exp=new edu.bitsgoa.utilities.BetterRunProcessThreaded(cmd,path_executables,0,1,0,"deviceQuery.txt","Gathering GPU configuration data...",false,"Done");
				executor.execute(exp);
			}
			if(!existenceArray[0]){	//implies bandWidth.txt is not present at path_home. So, generate the file
				cmd[0]="./bandWidth.out";
				edu.bitsgoa.utilities.BetterRunProcessThreaded exp=new edu.bitsgoa.utilities.BetterRunProcessThreaded(cmd,path_executables,0,1,0,"bandWidth.txt","Gathering GPU bandwidth data...",false,"Done");
				executor.execute(exp);
			}
			if(!existenceArray[2]){
				/*if bvar1=1 and bvar2=0, then KernelLaunchOverhead.txt exists, but KernelLaunchOverheadModel.txt does not. So, generate only 
					KernelLaunchOverheadModel.txt. In any other case,generate both KernelLaunchOverheadModel.txt and KernelLaunchOverhead.txt.
					We need the latter to generate the former
				 */
				KernelLaunchOverhead klo=new KernelLaunchOverhead(bvar1, bvar2);
				executor.execute(klo);
			}
			MemoryLatency ml=new MemoryLatency();	//Calculate memory latencies. The logic is in class MemoryLatency.java
			executor.execute(ml);
			
			Microbenchmark mb=new Microbenchmark(existenceArray);
			executor.execute(mb);

		}catch(Exception e){
			e.printStackTrace();
		}
		
		//now that all the files have been generated, the user may fill-in the parameter values.
		if(Parameters.timesused==0){
			executor.execute(new Runnable() {
				@Override
				public void run() {
					DisplayCustomConsole.display("Fill in the input paramters by going to Project->Properties->PTXAnalysis",true);
					}
			});
		}
		else{
			executor.execute(new Runnable() {
				@Override
				public void run() {
					DisplayCustomConsole.display("You can now perform energy estimation",true);				
				}
			});
		}

		return null;
	}
	/**
	 * Check if certain files exist at path_home. If yes, then make the corresponding entry in existenceArray as true.
	 * @param null
	 * @return null
	 */
	public static void filesExist(){
		boolean checkBanwidth = new File(path_home,"bandWidth.txt").exists();	//Does bandWidth.txt exist?
		boolean checkDeviceQuery = new File(path_home,"deviceQuery.txt").exists();	//Does deviceQuery.txt exist?
		boolean kernelLaunchOverhead = new File(path_home,"kernelLaunchOverhead.txt").exists();	//Does KernelLaunchOverhead.txt exist?
		boolean kernelLaunchOverheadModel=new File(path_home,"KernelLaunchOverheadModel.txt").exists();	//Does KernelLaunchOverheadModel.txt exist?
		//check for the presence of latencies of native instructions
		boolean addf32=new File(path_home,"times_addf32.txt").exists();
		boolean andb32=new File(path_home,"times_andb32.txt").exists();
		boolean divf32=new File(path_home,"times_divf32.txt").exists();
		boolean divs32=new File(path_home,"times_divs32.txt").exists();
		boolean mad=new File(path_home,"times_mad.txt").exists();
		boolean mulf32=new File(path_home,"times_mulf32.txt").exists();
		boolean sqrtcvt=new File(path_home,"times_sqrtcvt.txt").exists();
		boolean subs32=new File(path_home,"times_subs32.txt").exists();
		boolean av_addf32=new File(path_home,"av_times_addf32.txt").exists();
		boolean av_andb32=new File(path_home,"av_times_andb32.txt").exists();
		boolean av_divf32=new File(path_home,"av_times_divf32.txt").exists();
		boolean av_divs32=new File(path_home,"av_times_divs32.txt").exists();
		boolean av_mad=new File(path_home,"av_times_mad.txt").exists();
		boolean av_mulf32=new File(path_home,"av_times_mulf32.txt").exists();
		boolean av_sqrtcvt=new File(path_home,"av_times_sqrtcvt.txt").exists();
		boolean av_subs32=new File(path_home,"av_times_subs32.txt").exists();

		if(checkBanwidth) existenceArray[0]=true;	//If yes, then take action
		if(checkDeviceQuery) existenceArray[1]=true;	//If yes, then take action
		if(kernelLaunchOverhead && !kernelLaunchOverheadModel){
			existenceArray[2]=false;
			bvar1=1;
			bvar2=0;
		}
		else if(!kernelLaunchOverhead && kernelLaunchOverheadModel){
			existenceArray[2]=false;
			bvar1=0;
			bvar2=0;
		}
		else if(!kernelLaunchOverhead && !kernelLaunchOverheadModel){
			existenceArray[2]=false;
			bvar1=0;
			bvar2=0;
		}
		else existenceArray[2]=true;	//Implies KernelLaunchOverhead.txt and KernelLaunchOverheadModel.txt both exist
		
		if(addf32) existenceArray[3]=true;
		if(andb32) existenceArray[4]=true;
		if(divf32) existenceArray[5]=true;
		if(divs32) existenceArray[6]=true;
		if(mad) existenceArray[7]=true;
		if(mulf32) existenceArray[8]=true;
		if(sqrtcvt) existenceArray[9]=true;
		if(subs32) existenceArray[10]=true;
		if(av_addf32) existenceArray[11]=true;
		if(av_andb32) existenceArray[12]=true;
		if(av_divf32) existenceArray[13]=true;
		if(av_divs32) existenceArray[14]=true;
		if(av_mad) existenceArray[15]=true;
		if(av_mulf32) existenceArray[16]=true;
		if(av_sqrtcvt) existenceArray[17]=true;
		if(av_subs32) existenceArray[18]=true;
	}
	/**
	 * Initialize different variables and call filesExist() to check if certain files exist at path_home
	 * @param null
	 * @return null
	 */
	public static void initialize(){
		process=new BetterRunProcess();
		existenceArray=new boolean[19];
		for(int i=0;i<19;i++)	existenceArray[i]=false;
		filesExist();
	}
	/** Make a directory at path_home
	 * @param null
	 * @return null
	 */
	public static void mkdir(){
		new File(path_home).mkdir();
	}
	

	public boolean isEnabled(){
		return return_val;
	}
}