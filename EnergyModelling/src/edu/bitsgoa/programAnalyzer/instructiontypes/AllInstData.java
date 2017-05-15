package edu.bitsgoa.programAnalyzer.instructiontypes;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;

import com.opencsv.CSVReader;

import edu.bitsgoa.driver.ConfigureModel;
import edu.bitsgoa.driver.Preparation;
import edu.bitsgoa.properties.ParametersValue;
import edu.bitsgoa.regression.MemoryLatency;

public final class AllInstData {

	public static DataComputeInst[] computeDetails;
	public static int numComputeInstTypes;
	public static DataMemoryInst[] memoryDetails;
	public static int numMemoryInstTypes;
	public static DataMiscInst[] miscDetails;
	public static int numMiscInstTypes;
	//instructions
	static int fadd_mul_muladd_16;
	static int fadd_mul_muladd_32;
	static int fadd_mul_muladd_64;
	static int add_expadd_sub_expsub_32;
	static int mul_muladd_expmuladd_32;
	static int intmul_24;

	public static void initializeAll() {
		initializeCompute();
		initializeMemory();
		initializeMisc();
	}

	public static void initializeCompute() {
		numComputeInstTypes = EnumComputeInsts.values().length; //number of native instructions
		computeDetails = new DataComputeInst[numComputeInstTypes];

		for (int i = 0; i < numComputeInstTypes; i++) {
			computeDetails[i] = new DataComputeInst();
		}
		// latency, throughput, peak-warps, delay
		
		computeDetails[0].setData(EnumComputeInsts.fmadd, "fmadd",(int)Float.parseFloat(readLatenciesFromFile("av_times_addf32.txt")),instructionThroughput()[1], 16, 0.0); 	// 18, 32, 16	//present
		computeDetails[1].setData(EnumComputeInsts.fadd, "fadd",(int)Float.parseFloat(readLatenciesFromFile("av_times_addf32.txt")), 32, 16, 0.0); 		// 16, 32, 16	//present
		computeDetails[2].setData(EnumComputeInsts.madd, "madd", 18, 16, 11, 0.0); 		// 22, 16, 11	//not present
		computeDetails[3].setData(EnumComputeInsts.mad, "mad",(int)Float.parseFloat(readLatenciesFromFile("av_times_mad.txt")), 16, 11, 0.0);	 	// 22, 16, 11	//present	
		computeDetails[4].setData(EnumComputeInsts.add, "add", 10, 32, 16, 0.0); 		//16, 32, 16	//which one. 2 adds are present
		computeDetails[5].setData(EnumComputeInsts.sub, "sub",(int)Float.parseFloat(readLatenciesFromFile("av_times_subs32.txt")), 32, 16, 0.0); 		//16, 32, 16	//subs32 present
		computeDetails[6].setData(EnumComputeInsts.fmul, "fmul",(int)Float.parseFloat(readLatenciesFromFile("av_times_mulf32.txt")), 32, 16, 0.0);		// 16, 32, 16	//muf32
		computeDetails[7].setData(EnumComputeInsts.mul, "mul", 9, 16, 16, 0.0); 		// 20, 16, 16	//muls
		computeDetails[8].setData(EnumComputeInsts.fdiv, "fdiv",(int)Float.parseFloat(readLatenciesFromFile("av_times_divf32.txt")), 0.75, 4, 0.0); 	// 711, 0.75, 4	//present divf32
		computeDetails[9].setData(EnumComputeInsts.div, "div",(int)Float.parseFloat(readLatenciesFromFile("av_times_divs32.txt")), 1.8, 5, 0.0);		// 317, 1.8, 5	//present divs32
		computeDetails[10].setData(EnumComputeInsts.and, "and", 9, 32, 16, 0.0); 		// 16, 32, 16	//andb32 present
		computeDetails[11].setData(EnumComputeInsts.sqrt, "sqrt",(int)Float.parseFloat(readLatenciesFromFile("av_times_sqrtcvt.txt")), 1.6, 5, 0.0); 	// 269, 1.6, 5	//sqrtcvt			
		computeDetails[12].setData(EnumComputeInsts.mov, "mov", 10, 32, 16, 0.0);		// 16, 32, 16	//not present
		computeDetails[13].setData(EnumComputeInsts.setp, "setp", 10, 32, 16, 0.0);		// 16, 32, 16	//not present
		computeDetails[14].setData(EnumComputeInsts.fma, "fma", 18, 32, 16, 0.0); 		// 16, 32, 16	//not present
		computeDetails[15].setData(EnumComputeInsts.cvt, "cvt", 9, 32, 16, 0.0); 		// 16, 32, 16	//not present
	}

	public static void initializeMemory() {
		numMemoryInstTypes = EnumMemoryInsts.values().length;
		memoryDetails = new DataMemoryInst[numMemoryInstTypes];
		for (int i = 0; i < numMemoryInstTypes; i++) {
			memoryDetails[i] = new DataMemoryInst();
		}
		int totalThreads;
		double latency;
		totalThreads=ConfigureModel.numBlocks*ConfigureModel.numThreadsPerBlock;
		latency=MemoryLatency.psf.value(totalThreads);
		// latency, peakwarps, delay
		memoryDetails[0].setData(EnumMemoryInsts.GlobalLoad, "ld.global", latency, 8, 0.0); // 305, 8
		memoryDetails[1].setData(EnumMemoryInsts.GlobalStore, "st.global",latency, 8, 0.0); // 305, 8
		memoryDetails[2].setData(EnumMemoryInsts.SharedLoad, "ld.shared", 36, 2, 0.0); // 20, 2
		memoryDetails[3].setData(EnumMemoryInsts.SharedStore, "st.global", 36, 8, 0.0); // 36, 8
		memoryDetails[4].setData(EnumMemoryInsts.ParamLoad, "ld.param", latency, 8, 0.0);
		memoryDetails[5].setData(EnumMemoryInsts.ParamStore, "st.param", latency, 8, 0.0); //481
	}

	public static void initializeMisc() {
		numMiscInstTypes = EnumMiscInsts.values().length;
		miscDetails = new DataMiscInst[numMiscInstTypes];
		for (int i = 0; i < numMiscInstTypes; i++) {
			miscDetails[i] = new DataMiscInst();
		}
		// latency, throughput, peak-warps, delay
		miscDetails[0].setData(EnumMiscInsts.Sync, "sync", 10, 0, 0, 0.0);
		miscDetails[1].setData(EnumMiscInsts.Label, "label", 0, 0, 0, 0.0);
		miscDetails[2].setData(EnumMiscInsts.Branch, "branch", 10, 0, 0, 0.0);
		miscDetails[3].setData(EnumMiscInsts.Return, "return", 10, 0, 0, 0.0);
	}

	public static String readLatenciesFromFile(String fileName){
		String path=Preparation.path_home+fileName;
		try {
			BufferedReader br = new BufferedReader(new FileReader(path));
			String[] str=br.readLine().split("\\s+");
			return str[1];
		} catch (FileNotFoundException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		return null;
	}
	public static int[] instructionThroughput(){
		int[] instTp=new int[16];
		int tp_fmadd,tp_fadd,tp_madd,tp_mad,tp_add,tp_sub,tp_fma,tp_mul,tp_fmul,tp_fdiv,tp_div,tp_and,tp_sqrt,tp_mov,tp_setp,tp_cvt;
		if(ParametersValue.version=="2"){
			tp_fmadd=readThroughput(2,2);
			tp_fadd=readThroughput(2,2);
			tp_madd=readThroughput(6,2);
			tp_mad=readThroughput(6,2);
			tp_add=readThroughput(5,2);
			tp_sub=readThroughput(5,2);
			tp_fmul=readThroughput(2,2);
			tp_mul=readThroughput(6,2);
			tp_fdiv=readThroughput(1,2);
			tp_div=readThroughput(1,2);
			tp_and=readThroughput(11,2);
			tp_sqrt=readThroughput(4,2);
			tp_mov=readThroughput(1,2);
			tp_setp=readThroughput(1,2);
			tp_fma=readThroughput(1,2);
			tp_cvt=readThroughput(1,2);
		}
		else if(ParametersValue.version=="2.1"){
			tp_fmadd=readThroughput(2,3);
			tp_fadd=readThroughput(2,3);
			tp_madd=readThroughput(6,3);
			tp_mad=readThroughput(6,3);
			tp_add=readThroughput(5,3);
			tp_sub=readThroughput(5,3);
			tp_fmul=readThroughput(2,3);
			tp_mul=readThroughput(6,3);
			tp_fdiv=readThroughput(1,3);
			tp_div=readThroughput(1,3);
			tp_and=readThroughput(11,3);
			tp_sqrt=readThroughput(4,3);
			tp_mov=readThroughput(1,3);
			tp_setp=readThroughput(1,3);
			tp_fma=readThroughput(1,3);
			tp_cvt=readThroughput(1,3);
		}
		else if(ParametersValue.version=="3.0" || ParametersValue.version=="3.2"){
			tp_fmadd=readThroughput(2,4);
			tp_fadd=readThroughput(2,4);
			tp_madd=readThroughput(6,4);
			tp_mad=readThroughput(6,4);
			tp_add=readThroughput(5,4);
			tp_sub=readThroughput(5,4);
			tp_fmul=readThroughput(2,4);
			tp_mul=readThroughput(6,4);
			tp_fdiv=readThroughput(1,4);
			tp_div=readThroughput(1,4);
			tp_and=readThroughput(11,4);
			tp_sqrt=readThroughput(4,4);
			tp_mov=readThroughput(1,4);
			tp_setp=readThroughput(1,4);
			tp_fma=readThroughput(1,4);
			tp_cvt=readThroughput(1,4);
		}
		else if(ParametersValue.version=="3.5" || ParametersValue.version=="3.7"){
			tp_fmadd=readThroughput(2,5);
			tp_fadd=readThroughput(2,5);
			tp_madd=readThroughput(6,5);
			tp_mad=readThroughput(6,5);
			tp_add=readThroughput(5,5);
			tp_sub=readThroughput(5,5);
			tp_fmul=readThroughput(2,5);
			tp_mul=readThroughput(6,5);
			tp_fdiv=readThroughput(1,5);
			tp_div=readThroughput(1,5);
			tp_and=readThroughput(11,5);
			tp_sqrt=readThroughput(4,5);
			tp_mov=readThroughput(1,5);
			tp_setp=readThroughput(1,5);
			tp_fma=readThroughput(1,5);
			tp_cvt=readThroughput(1,5);
		}
		else if(ParametersValue.version=="5.0" || ParametersValue.version=="5.2"){
			tp_fmadd=readThroughput(2,6);
			tp_fadd=readThroughput(2,6);
			tp_madd=readThroughput(6,6);
			tp_mad=readThroughput(6,6);
			tp_add=readThroughput(5,6);
			tp_sub=readThroughput(5,6);
			tp_fmul=readThroughput(2,6);
			tp_mul=readThroughput(6,6);
			tp_fdiv=readThroughput(1,6);
			tp_div=readThroughput(1,6);
			tp_and=readThroughput(11,6);
			tp_sqrt=readThroughput(4,6);
			tp_mov=readThroughput(1,6);
			tp_setp=readThroughput(1,6);
			tp_fma=readThroughput(1,6);
			tp_cvt=readThroughput(1,6);

		}
		else if(ParametersValue.version=="5.3"){
			tp_fmadd=readThroughput(2,7);
			tp_fadd=readThroughput(2,7);
			tp_madd=readThroughput(6,7);
			tp_mad=readThroughput(6,7);
			tp_add=readThroughput(5,7);
			tp_sub=readThroughput(5,7);
			tp_fmul=readThroughput(2,7);
			tp_mul=readThroughput(6,7);
			tp_fdiv=readThroughput(1,7);
			tp_div=readThroughput(1,7);
			tp_and=readThroughput(11,7);
			tp_sqrt=readThroughput(4,7);
			tp_mov=readThroughput(1,7);
			tp_setp=readThroughput(1,7);
			tp_fma=readThroughput(1,7);
			tp_cvt=readThroughput(1,7);
		}
		else if(ParametersValue.version=="6"){
			tp_fmadd=readThroughput(2,8);
			tp_fadd=readThroughput(2,8);
			tp_madd=readThroughput(6,8);
			tp_mad=readThroughput(6,8);
			tp_add=readThroughput(5,8);
			tp_sub=readThroughput(5,8);
			tp_fmul=readThroughput(2,8);
			tp_mul=readThroughput(6,8);
			tp_fdiv=readThroughput(1,8);
			tp_div=readThroughput(1,8);
			tp_and=readThroughput(11,8);
			tp_sqrt=readThroughput(4,8);
			tp_mov=readThroughput(1,8);
			tp_setp=readThroughput(1,8);
			tp_fma=readThroughput(1,8);
			tp_cvt=readThroughput(1,8);
		}
		else if(ParametersValue.version=="6.1"){
			tp_fmadd=readThroughput(2,9);
			tp_fadd=readThroughput(2,9);
			tp_madd=readThroughput(6,9);
			tp_mad=readThroughput(6,9);
			tp_add=readThroughput(5,9);
			tp_sub=readThroughput(5,9);
			tp_fmul=readThroughput(2,9);
			tp_mul=readThroughput(6,9);
			tp_fdiv=readThroughput(1,9);
			tp_div=readThroughput(1,9);
			tp_and=readThroughput(11,9);
			tp_sqrt=readThroughput(4,9);
			tp_mov=readThroughput(1,9);
			tp_setp=readThroughput(1,9);
			tp_fma=readThroughput(1,9);
			tp_cvt=readThroughput(1,9);
		}
		else {
			tp_fmadd=readThroughput(2,10);
			tp_fadd=readThroughput(2,10);
			tp_madd=readThroughput(6,10);
			tp_mad=readThroughput(6,10);
			tp_add=readThroughput(5,10);
			tp_sub=readThroughput(5,10);
			tp_fmul=readThroughput(2,10);
			tp_mul=readThroughput(6,10);
			tp_fdiv=readThroughput(1,10);
			tp_div=readThroughput(1,10);
			tp_and=readThroughput(11,10);
			tp_sqrt=readThroughput(4,10);
			tp_mov=readThroughput(1,10);
			tp_setp=readThroughput(1,10);
			tp_fma=readThroughput(1,10);
			tp_cvt=readThroughput(1,10);
		}
		instTp[0]=tp_fmadd;
		instTp[1]=tp_fadd;
		instTp[2]=tp_madd;
		instTp[3]=tp_mad;
		instTp[4]=tp_add;
		instTp[5]=tp_sub;
		instTp[6]=tp_fma;
		instTp[7]=tp_mul;
		instTp[8]=tp_fmul;
		instTp[9]=tp_fdiv;
		instTp[10]=tp_div;
		instTp[11]=tp_and;
		instTp[12]=tp_sqrt;
		instTp[13]=tp_mov;
		instTp[14]=tp_setp;
		instTp[15]=tp_cvt;
		
		return instTp;
	}
	public static int readThroughput(int line,int colNo){
		int throughput=0;
		try {
			CSVReader reader = new CSVReader(new FileReader("/home/limafoxtrottango/cuda-workspace/edu.bitsgoa.EnergyModelling/Executables"+"/throughput.csv"), ',' , '"' ,0);
			String[] nextLine;
			int lineNo=0;
			while ((nextLine = reader.readNext()) != null) {
				if (nextLine != null) {
					//Verifying the read data here
					if(lineNo==line){
						throughput=Integer.parseInt(nextLine[colNo-1]);
					}
				}
				lineNo++;
			}
		} catch (FileNotFoundException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch(NumberFormatException e){
			return -1;
		}
		return throughput;

	}
		
	private AllInstData() {
		throw new RuntimeException("Do not instantiate this class: " + getClass());
	}

}