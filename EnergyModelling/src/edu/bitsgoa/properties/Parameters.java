package edu.bitsgoa.properties;
/**
 * This class reads from the generated text files (deviceQuery.txt and bandWidth.txt) to automatically populate certain parameters used in energy
 * estimation calculation. The user can change the values as he wishes. 
 */
import java.awt.Container;
import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import org.eclipse.swt.SWT;
import org.eclipse.swt.widgets.Composite;
import org.eclipse.swt.widgets.Control;
import org.eclipse.swt.widgets.Label;
import org.eclipse.swt.widgets.Text;
import org.eclipse.ui.dialogs.PropertyPage;
import edu.bitsgoa.driver.Preparation;
import edu.bitsgoa.views.DisplayCustomConsole;

public class Parameters extends PropertyPage {
	
	public static int timesused=0;	//the number of times this Property Page has been opened. This variable is useful in saving user-entered values b/w multiple calls to this class
	private static Text maxNoThreadPerSM;	//max no of threads per SM
	private static Text noSM;	//no of SMs present in the card
	private static Text warpsize;	//warp size
	private static Text noBanks;	//no of banks
	private static Text devTohostBW;	//dev. to host bw
	private static Text hostToDevBW;	//host to dev. bw
	private static Text globalMemBW;	//global mem. bw
	private static Text noCoresPerSM;	//no of cores per sm
	private static Text globalMemLineSize;	//global mem. line size
	private static Text noThreadsPerBlock;	//no. of threads per block
	private static Text noBlocks;	//no. of blocks
	private static Text noOfLoopIterations;	//no. of loop iterations
	private static Text GPUclock;	//freq. of the processor
	private static Text transferBWPeak;		//transfer bw. peak
	private static Text transferSize;	//transfer size
	private static Text branchProb;	//branch probablity
	private static Text maxNoofKernels;	//max. no. of kernels
	private static Text maxRegPerInst;	//max. no. of reg. per inst.
	private static Text noIndptInst;	//no. of independent inst.
	private static Text maxActvWarpPerInst;	//max. active warps per inst.
	private static Text accFactorBadCoal;	//access factor bad coales.
	private static Text noOfBankConf;	//no. of bank conflicts
	private static Text sharedBytesTrans;	//shared bytes transferred	
	private static String version;	//cuda compute capability
	private String[] arr;
	
	/**
	 * This method is called when the "Restore Default Values" button is pressed. Effectively, it fills all the text-boxes with default values
	 * @param
	 * @return
	 */
	@Override
	protected void performDefaults() {
		super.performDefaults();
		timesused=0;
		defaultValues();
	}
	/**
	 * This method is called when the "Apply" button is pressed. The values in the text-boxes are committed
	 * @param
	 * @return
	 */
	@Override
	protected void performApply() {
		super.performApply();
		initializeValues();
	}
	/**
	 * This method is called by createParthControl() method to create all the labels and buttons
	 * @param container
	 * @return
	 */
	public void createLables(Composite container){
		Label lblPleaseFillinThe = new Label(container, SWT.PUSH);
		lblPleaseFillinThe.setBounds(10, 3, 262,22);
		lblPleaseFillinThe.setText("Please fill-in the following parameters:");

		Label noOfSM = new Label(container, SWT.NONE);
		noOfSM.setBounds(10, 31, 150, 22);
		noOfSM.setText("No. of SMs");

		Label lblNewLabel = new Label(container, SWT.NONE);
		lblNewLabel.setBounds(10, 52, 150, 22);
		lblNewLabel.setText("Max. threads per SM");

		Label lblNewLabel_1 = new Label(container, SWT.NONE);
		lblNewLabel_1.setBounds(10, 73, 175, 22);
		lblNewLabel_1.setText("Device to host bandwidth");

		Label label = new Label(container, SWT.NONE);
		label.setBounds(10,115, 173, 22);
		label.setText("Host to device bandwidth");

		Label label_1 = new Label(container, SWT.NONE);
		label_1.setBounds(10, 136, 184, 22);
		label_1.setText("Global memory bandwidth");

		Label label_2 = new Label(container, SWT.NONE);
		label_2.setBounds(10, 157,133, 22);
		label_2.setText("No. of cores per SM");		

		Label lblSadad = new Label(container, SWT.NONE);
		lblSadad.setBounds(10, 94, 133, 22);
		lblSadad.setText("No. of banks");

		Label lblNewLabel_2 = new Label(container, SWT.NONE);
		lblNewLabel_2.setBounds(10,178,193, 22);
		lblNewLabel_2.setText("Global memory line size");

		Label warpSize=new Label(container,SWT.NONE);
		warpSize.setBounds(10,199,82, 22);
		warpSize.setText("Warp size");

		Label gpuclock=new Label(container,SWT.NONE);
		gpuclock.setBounds(10,220,107,22);
		gpuclock.setText("GPU clock (Hz)");

		Label lblNewLabel_4 = new Label(container, SWT.NONE);
		lblNewLabel_4.setBounds(320,52, 82, 22);
		lblNewLabel_4.setText("No. blocks");

		Label lblNewLabel_3 = new Label(container, SWT.NONE);
		lblNewLabel_3.setBounds(320,31, 173, 22);
		lblNewLabel_3.setText("No. of threads per block");

		Label lblNewLabel_5 = new Label(container, SWT.NONE);
		lblNewLabel_5.setBounds(320,73, 157, 22);
		lblNewLabel_5.setText("No. of loop iterations");

		Label label_4 = new Label(container, SWT.SEPARATOR | SWT.VERTICAL);
		label_4.setBounds(300, 23, 11, 259);

		Label lblTransferBandwidthPeak = new Label(container, SWT.NONE);
		lblTransferBandwidthPeak.setBounds(10, 242, 175, 17);
		lblTransferBandwidthPeak.setText("Transfer bandwidth peak");


		Label lblTranserSize = new Label(container, SWT.NONE);
		lblTranserSize.setBounds(320, 96, 107, 17);
		lblTranserSize.setText("Transer size");

		Label lblBranchProbability = new Label(container, SWT.NONE);
		lblBranchProbability.setBounds(320, 115, 124, 17);
		lblBranchProbability.setText("Branch probability");

		Label lblMaxNoOf = new Label(container, SWT.NONE);
		lblMaxNoOf.setBounds(320, 136, 143, 17);
		lblMaxNoOf.setText("Max. no. of kernels");

		Label lblMaxRegistersPer = new Label(container, SWT.NONE);
		lblMaxRegistersPer.setBounds(320, 157, 173, 17);
		lblMaxRegistersPer.setText("Max. reg. per instruction");

		Label lblNoOfIndpendent = new Label(container, SWT.NONE);
		lblNoOfIndpendent.setBounds(320, 178, 157, 17);
		lblNoOfIndpendent.setText("No. of indpendent inst");

		Label lblMaxActiveWarps = new Label(container, SWT.NONE);
		lblMaxActiveWarps.setBounds(320, 199, 173, 17);
		lblMaxActiveWarps.setText("Max. active warps per inst");

		Label lblAccessFactorBad = new Label(container, SWT.NONE);
		lblAccessFactorBad.setBounds(320, 220, 173, 17);
		lblAccessFactorBad.setText("Access factor bad coales");

		Label lblNoOfBank = new Label(container, SWT.NONE);
		lblNoOfBank.setBounds(320,245, 173, 17);
		lblNoOfBank.setText("No. of bank conflicts");

		Label lblSharedBytesTransferred = new Label(container, SWT.NONE);
		lblSharedBytesTransferred.setBounds(320, 268, 173, 17);
		lblSharedBytesTransferred.setText("Shared bytes transferred");

		noSM=new Text(container,SWT.BORDER);	//no of SMs
		noSM.setBounds(211,31,75, 22);

		maxNoThreadPerSM = new Text(container, SWT.BORDER);	//max no of threads per SM
		maxNoThreadPerSM.setBounds(211, 52, 75, 22);

		GPUclock=new Text(container,SWT.BORDER);
		GPUclock.setBounds(211,217,75, 22);

		noBanks = new Text(container, SWT.BORDER);	//No of banks
		noBanks.setBounds(211, 94, 75, 22);

		devTohostBW = new Text(container, SWT.BORDER);	//device to host bw
		devTohostBW.setBounds(211,73, 75, 22);

		hostToDevBW = new Text(container, SWT.BORDER);	//host to device bw
		hostToDevBW.setBounds(211,115, 75, 22);

		globalMemBW = new Text(container, SWT.BORDER);	//global mem bw
		globalMemBW.setBounds(211,136, 75, 22);

		warpsize=new Text(container,SWT.BORDER);	//warp size
		warpsize.setBounds(211,199,73,22);

		noCoresPerSM = new Text(container, SWT.BORDER);	//no of cores per SM
		noCoresPerSM.setBounds(211,157, 75, 22);

		globalMemLineSize = new Text(container, SWT.BORDER);	//global memory line size
		globalMemLineSize.setBounds(211,178,73,22);

		noThreadsPerBlock = new Text(container, SWT.BORDER);	//no of threads per block
		noThreadsPerBlock.setBounds(499,31,78,22);

		noBlocks = new Text(container, SWT.BORDER);	//no of blocks
		noBlocks.setBounds(499,52, 78, 22);

		noOfLoopIterations = new Text(container, SWT.BORDER);	//No. of loop iterations
		noOfLoopIterations.setBounds(499,73, 78,22);

		transferBWPeak = new Text(container, SWT.BORDER);
		transferBWPeak.setBounds(211, 240, 75, 22);

		transferSize = new Text(container, SWT.BORDER);
		transferSize.setBounds(499, 94, 78, 22);

		branchProb = new Text(container, SWT.BORDER);
		branchProb.setBounds(499, 115, 78, 22);

		maxNoofKernels = new Text(container, SWT.BORDER);
		maxNoofKernels.setBounds(499, 136, 78, 22);

		maxRegPerInst = new Text(container, SWT.BORDER);
		maxRegPerInst.setBounds(499, 157, 78, 22);

		noIndptInst = new Text(container, SWT.BORDER);
		noIndptInst.setBounds(499, 178, 78, 22);

		maxActvWarpPerInst = new Text(container, SWT.BORDER);
		maxActvWarpPerInst.setBounds(499, 199, 78, 22);

		accFactorBadCoal = new Text(container, SWT.BORDER);
		accFactorBadCoal.setBounds(499, 220, 78, 22);

		noOfBankConf = new Text(container, SWT.BORDER);
		noOfBankConf.setBounds(499, 242, 78, 22);

		sharedBytesTrans = new Text(container, SWT.BORDER);
		sharedBytesTrans.setBounds(499, 265, 78, 22);

	}
	
	/**
	 * This method is first called when a PropertyPage is created
	 * @param parent
	 * @return
	 */
	@Override
	public Control createContents(Composite parent) {	
		Composite container=new Composite(parent, SWT.NULL);
		if(timesused==0){	//if this is the first time the user is opening this page
			createLables(container);	//create labels and buttons

			String sm="";
			String ret=lines2(11);
			for(int i=0;i<ret.length();i++){
				char ch=ret.charAt(i);
				if(ch=='0'||ch=='1'||ch=='2'||ch=='3'||ch=='4'||ch=='5'||ch=='6'||ch=='7'||ch=='8'||ch=='9'){
					sm=sm+ch;
				}
				if(ch==')') break;
			}
			noSM.setText(sm);

			arr=lines2(23).split("\\s+");	
			maxNoThreadPerSM.setText(arr[arr.length-1]);

			arr=lines2(12).split("\\s+");
			GPUclock.setText(arr[arr.length-4]+"000");

			arr=lines2(9).split("\\s+");
			version=arr[arr.length-1];

			if(version.charAt(0)=='1') noBanks.setText("16");
			else if(version.charAt(0)=='2' || version.charAt(0)=='3') noBanks.setText("32");

			arr=lines(15).split("\\s+");
			devTohostBW.setText(arr[arr.length-1]);

			arr=lines(10).split("\\s+");
			hostToDevBW.setText(arr[arr.length-1]);

			arr=lines2(14).split("\\s+");
			int memBusBW=Integer.parseInt(arr[arr.length-1].substring(0,arr[arr.length-1].indexOf('-')));
			arr=lines2(13).split("\\s+");
			int memClkrate=Integer.parseInt(arr[arr.length-2]);
			double globalMemoryBw=(memClkrate*memBusBW*2)/8000;
			globalMemBW.setText(Double.toString(globalMemoryBw));

			arr=lines2(22).split("\\s+");
			warpsize.setText(arr[arr.length-1]);

			String sm1="";
			String ret1=lines2(11);
			int n=0;
			for(int i=0;i<ret1.length();i++){
				char ch=ret1.charAt(i);
				if(ch==',' || n==1){
					n=1;
					if(ch=='0'||ch=='1'||ch=='2'||ch=='3'||ch=='4'||ch=='5'||ch=='6'||ch=='7'||ch=='8'||ch=='9'){
						sm1=sm1+ch;
					}
					else if(ch==')') break;
					else continue;
				}
				else continue;
			}
			noCoresPerSM.setText(sm1);
			globalMemLineSize.setText("128");
			noThreadsPerBlock.setText("256");
			noBlocks.setText("64");
			noOfLoopIterations.setText("0");
			
			float devtohost=Float.parseFloat(devTohostBW.getText());
			float hosttodev=Float.parseFloat(hostToDevBW.getText());
			float average=(devtohost+hosttodev)/2;
			transferBWPeak.setText(Float.toString(average));
			
			transferSize.setText("13631488");
			branchProb.setText("0.5");
			maxNoofKernels.setText("10");
			maxRegPerInst.setText("5");
			noIndptInst.setText("2");
			maxActvWarpPerInst.setText("64");
			accFactorBadCoal.setText("32");
			noOfBankConf.setText("0");
			sharedBytesTrans.setText("4");
			timesused=1;	//turn timesused=1 since the page has been used at-least 1 time
		}
		else
		{	//Since the user has already opened this page and is opening again, fill-in the text-boxes with the saved values
			createLables(container);
			noSM.setText(ParametersValue.noSM);
			maxNoThreadPerSM.setText(ParametersValue.maxNoThreadPerSM);
			GPUclock.setText(ParametersValue.GPUclock);
			noBanks.setText(ParametersValue.noBanks);
			devTohostBW.setText(ParametersValue.devTohostBW);
			hostToDevBW.setText(ParametersValue.hostToDevBW);
			globalMemLineSize.setText(ParametersValue.globalMemLineSize);
			globalMemBW.setText(ParametersValue.globalMemBW);
			warpsize.setText(ParametersValue.warpsize);
			noCoresPerSM.setText(ParametersValue.noCoresPerSM);
			noThreadsPerBlock.setText(ParametersValue.noThreadsPerBlock);
			noBlocks.setText(ParametersValue.noBlocks);
			noOfLoopIterations.setText(ParametersValue.noOfLoopIterations);
			transferBWPeak.setText(ParametersValue.transferBWPeak);
			transferSize.setText(ParametersValue.transferSize);
			branchProb.setText(ParametersValue.branchProb);
			maxNoofKernels.setText(ParametersValue.maxNoofKernels);
			maxRegPerInst.setText(ParametersValue.maxActvWarpPerInst);
			noIndptInst.setText(ParametersValue.noIndptInst);
			maxActvWarpPerInst.setText(ParametersValue.maxActvWarpPerInst);
			accFactorBadCoal.setText(ParametersValue.accFactorBadCoal);
			noOfBankConf.setText(ParametersValue.noOfBankConf);
			sharedBytesTrans.setText(ParametersValue.sharedBytesTrans);
			
		}
		if(maxNoThreadPerSM.getText()!=null && noBanks.getText()!=null && devTohostBW.getText()!=null && noCoresPerSM.getText()!=null && globalMemBW.getText()!=null && globalMemLineSize.getText()!=null && noThreadsPerBlock.getText()!=null && noBlocks.getText()!=null && noOfLoopIterations.getText()!=null){
			if(Preparation.used==0) DisplayCustomConsole.display("Now, click on the Prepare option in the menu bar",true);
			else DisplayCustomConsole.display("Energy Estimation can now be performed by going to the menu bar->Energy Estimation->Perform Energy Estimationm",true);
			initializeValues();
		}
		return container;
	}
	/**
	 * This method is called to fill-in the default values in the text-boxes. To do so, we just read from the text files again.
	 * @param
	 * @return
	 */
	public void defaultValues(){

		String sm="";
		String ret=lines2(11);
		for(int i=0;i<ret.length();i++){
			char ch=ret.charAt(i);
			if(ch=='0'||ch=='1'||ch=='2'||ch=='3'||ch=='4'||ch=='5'||ch=='6'||ch=='7'||ch=='8'||ch=='9'){
				sm=sm+ch;
			}
			if(ch==')') break;
		}
		noSM.setText(sm);
		
		arr=lines2(23).split("\\s+");	
		maxNoThreadPerSM.setText(arr[arr.length-1]);

		arr=lines2(12).split("\\s+");
		GPUclock.setText(arr[arr.length-4]+"000");

		arr=lines2(9).split("\\s+");
		version=arr[arr.length-1];
		if(version.charAt(0)=='1') noBanks.setText("16");
		if(version.charAt(0)=='2' || version.charAt(0)=='3') noBanks.setText("32");

		arr=lines(15).split("\\s+");
		devTohostBW.setText(arr[arr.length-1]);

		arr=lines(10).split("\\s+");
		hostToDevBW.setText(arr[arr.length-1]);

		arr=lines2(14).split("\\s+");
		int memBusBW=Integer.parseInt(arr[arr.length-1].substring(0,arr[arr.length-1].indexOf('-')));
		arr=lines2(13).split("\\s+");
		int memClkrate=Integer.parseInt(arr[arr.length-2]);
		double globalMemoryBw=(memClkrate*memBusBW*2)/8000;
		globalMemBW.setText(Double.toString(globalMemoryBw));

		arr=lines2(22).split("\\s+");
		warpsize.setText(arr[arr.length-1]);

		String sm1="";
		String ret1=lines2(11);
		int n=0;
		for(int i=0;i<ret1.length();i++){
			char ch=ret1.charAt(i);
			if(ch==',' || n==1){
				n=1;
				if(ch=='0'||ch=='1'||ch=='2'||ch=='3'||ch=='4'||ch=='5'||ch=='6'||ch=='7'||ch=='8'||ch=='9'){
					sm1=sm1+ch;
				}
				else if(ch==')') break;
				else continue;
			}
			else continue;
		}
		noCoresPerSM.setText(sm1);
		globalMemLineSize.setText("128");
		noThreadsPerBlock.setText("256");
		noBlocks.setText("64");
		noOfLoopIterations.setText("10");
		
		float devtohost=Float.parseFloat(devTohostBW.getText());
		float hosttodev=Float.parseFloat(hostToDevBW.getText());
		float average=(devtohost+hosttodev)/2;
		transferBWPeak.setText(Float.toString(average));
		
		transferSize.setText("13631488");
		branchProb.setText("0.5");
		maxNoofKernels.setText("10");
		maxRegPerInst.setText("5");
		noIndptInst.setText("2");
		maxActvWarpPerInst.setText("64");
		accFactorBadCoal.setText("0");
		noOfBankConf.setText("0");
		sharedBytesTrans.setText("128");
	}
	/**
	 * This method is called to "save" the current values in the text fields. If this method is not called, then as soon as the property page 
	 * is closed, the values are lost
	 * @param
	 * @return
	 */
	public static void initializeValues(){
		ParametersValue.version=version;
		ParametersValue.maxNoThreadPerSM=maxNoThreadPerSM.getText();
		ParametersValue.noSM=noSM.getText();
		ParametersValue.warpsize=warpsize.getText();
		ParametersValue.noBanks=noBanks.getText();
		ParametersValue.devTohostBW=devTohostBW.getText();
		ParametersValue.noCoresPerSM=noCoresPerSM.getText();
		ParametersValue.hostToDevBW=hostToDevBW.getText();
		ParametersValue.globalMemBW=globalMemBW.getText();
		ParametersValue.globalMemLineSize=globalMemLineSize.getText();
		ParametersValue.noThreadsPerBlock=noThreadsPerBlock.getText();
		ParametersValue.noBlocks=noBlocks.getText();
		ParametersValue.noOfLoopIterations=noOfLoopIterations.getText();
		ParametersValue.GPUclock=GPUclock.getText();
		ParametersValue.transferBWPeak=transferBWPeak.getText();
		ParametersValue.transferSize=transferSize.getText();
		ParametersValue.branchProb=branchProb.getText();
		ParametersValue.maxNoofKernels=maxNoofKernels.getText();
		ParametersValue.maxRegPerInst=maxRegPerInst.getText();
		ParametersValue.noIndptInst=noIndptInst.getText();
		ParametersValue.maxActvWarpPerInst=maxActvWarpPerInst.getText();
		ParametersValue.accFactorBadCoal=accFactorBadCoal.getText();
		ParametersValue.noOfBankConf=noOfBankConf.getText();
		ParametersValue.sharedBytesTrans=sharedBytesTrans.getText();
	}
	/**
	 * This method is called to read a particular line in the file "bandWidth.txt"
	 * @param line_no number of the line to read from deviceQuery.txt
	 * @return
	 */
	public static String lines(int line_no){
		String path=Preparation.path_home+"/bandWidth.txt";
		try {
			int counter =0;
			BufferedReader br = new BufferedReader(new FileReader(path));  
			String line=null;  
			while ((line = br.readLine()) != null) {   
				counter++;
				if(counter==line_no) return line;
			}  
			br.close();
		} catch(FileNotFoundException e){ 
			DisplayCustomConsole.display("Can't Find the Required Files, Prepare First",true);
			
		}
		catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}  
		return null;
	}
	/**
	 * This method is called to read a particular line in the file "deviceQuery.txt"
	 * @param line_no number of the line to read from deviceQuery.txt
	 * @return
	 */
	public static String lines2(int line_no){
		String path=Preparation.path_home+"/deviceQuery.txt";
		try {
			int counter =0;
			BufferedReader br = new BufferedReader(new FileReader(path));  
			String line=null;  
			while ((line = br.readLine()) != null) {   
				counter++;
				if(counter==line_no) 
					return line;
			}  
			br.close();
		}catch(FileNotFoundException e){ 
			DisplayCustomConsole.display("Can't Find the Required Files, Prepare First",true);
			
		}
		catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		return null;
	}
}
