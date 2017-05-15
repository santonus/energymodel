package edu.bitsgoa.programAnalyzer;

import edu.bitsgoa.programAnalyzer.instructiontypes.AllInstData;
import edu.bitsgoa.programAnalyzer.instructiontypes.EnumAllInsts;
import edu.bitsgoa.programAnalyzer.instructiontypes.EnumComputeInsts;
import edu.bitsgoa.programAnalyzer.instructiontypes.EnumMemoryInsts;
import edu.bitsgoa.programAnalyzer.instructiontypes.EnumMiscInsts;
import edu.bitsgoa.utilities.UtilsMisc;

public final class PTXUtil {

	public static EnumAllInsts decodeInstruction(String instruction) {
		if (instruction == null)
			return EnumAllInsts.Empty;
		
		instruction = instruction.trim();
		
		if (instruction.equals(""))
			return EnumAllInsts.Empty;
		else if (instruction.charAt(0) == '.')
			return EnumAllInsts.Directive;
		else if (instruction.length() >= 2 && instruction.charAt(0) == '/' && instruction.charAt(1) == '/')
			return EnumAllInsts.Comment;
		else if (instruction.contains("{") && instruction.length() < 3)
			return EnumAllInsts.KernelStart;
		else if (instruction.contains("}") && instruction.length() < 3)
			return EnumAllInsts.KernelEnd;
		else if (instruction.charAt(instruction.length() - 1) == ':')
			return EnumAllInsts.Label;
		else if (instruction.contains("sync"))
			return EnumAllInsts.Sync;
		else if (instruction.contains("bra"))
			return EnumAllInsts.Branch;
		else if (instruction.contains("ret") && instruction.length() < 5)
			return EnumAllInsts.Return;
		for (int i = 0; i < AllInstData.numComputeInstTypes; i++) {
			if (instruction.contains(AllInstData.computeDetails[i].getInstName())) {
				return EnumAllInsts.Computation;
			}
		}
		for (int i = 0; i < AllInstData.numMemoryInstTypes; i++) {
			if (instruction.contains(AllInstData.memoryDetails[i].getInstName())) {
				return EnumAllInsts.MemAccess;
			}
		}
		return EnumAllInsts.Unknown;
	}
	
	public static EnumComputeInsts decodeComputationType(String instruction) {
		EnumAllInsts instType = decodeInstruction(instruction);
		if (instType != EnumAllInsts.Computation)
			return null;
		
		if (instruction.contains("fmadd")) return EnumComputeInsts.fmadd;
		else if (instruction.contains("fadd")) return EnumComputeInsts.fadd;
		else if (instruction.contains("madd")) return EnumComputeInsts.madd;
		else if (instruction.contains("mad")) return EnumComputeInsts.mad;
		else if (instruction.contains("add")) return EnumComputeInsts.add;
		else if (instruction.contains("sub")) return EnumComputeInsts.sub;
		else if (instruction.contains("fmul")) return EnumComputeInsts.fmul;
		else if (instruction.contains("mul")) return EnumComputeInsts.mul;
		else if (instruction.contains("fdiv")) return EnumComputeInsts.fdiv;
		else if (instruction.contains("div")) return EnumComputeInsts.div;
		else if (instruction.contains("and")) return EnumComputeInsts.and;
		else if (instruction.contains("sqrt")) return EnumComputeInsts.sqrt;
		else if (instruction.contains("mov")) return EnumComputeInsts.mov;
//		else if (instruction.contains("ld.param")) return EnumComputeInsts.ldstparam;
		else if (instruction.contains("setp")) return EnumComputeInsts.setp;
//		else if (instruction.contains("bra")) return EnumComputeInsts.bra;
		else if (instruction.contains("fma")) return EnumComputeInsts.fma;
		else if (instruction.contains("cvt")) return EnumComputeInsts.cvt;
//		else if (instruction.contains("ret")) return EnumComputeInsts.ret;
		
		return null;
	}
	
	public static EnumMemoryInsts decodeMemoryType(String instruction) {
		EnumAllInsts instType = decodeInstruction(instruction);
		if (instType != EnumAllInsts.MemAccess)
			return null;
		
		if (instruction.contains("ld.global"))
			return EnumMemoryInsts.GlobalLoad;
		else if (instruction.contains("st.global"))
			return EnumMemoryInsts.GlobalStore;
		else if (instruction.contains("ld.shared"))
			return EnumMemoryInsts.SharedLoad;
		else if (instruction.contains("st.shared"))
			return EnumMemoryInsts.SharedStore;
		else if (instruction.contains("ld.param"))
			return EnumMemoryInsts.ParamLoad;
		else if (instruction.contains("st.param"))
			return EnumMemoryInsts.ParamStore;
		return null;
	}
	
	public static EnumMiscInsts decodeMiscInstType(String instruction) {
		EnumAllInsts instType = decodeInstruction(instruction);
		
		if (instType == EnumAllInsts.Sync)
			return EnumMiscInsts.Sync;
		else if (instType == EnumAllInsts.Branch)
			return EnumMiscInsts.Branch;
		else if (instType == EnumAllInsts.Label)
			return EnumMiscInsts.Label;
		else if (instType == EnumAllInsts.Return)
			return EnumMiscInsts.Return;
		return null;
	}
	
	/* KernelStart, KernelEnd, Label are the only three instructions
	 * that are neither executable nor can they be skipped while reading. */
	
	public static boolean isExecutableInst(EnumAllInsts instType) {
		if (instType == null)
			return false;
		if (instType == EnumAllInsts.Computation || instType == EnumAllInsts.MemAccess ||
				instType == EnumAllInsts.Sync || instType == EnumAllInsts.Branch || instType == EnumAllInsts.Return)
			return true;
		return false;
	}
	
	public static boolean isExecutableInst(String instruction) {
		return isExecutableInst(decodeInstruction(instruction));
	}
	
	public static boolean isControlInst(EnumAllInsts instType) {
		if (instType == null)
			return false;
		if (instType == EnumAllInsts.Sync || instType == EnumAllInsts.Branch || instType == EnumAllInsts.Return)
			return true;
		return false;
	}
	
	public static boolean isControlInst(String instruction) {
		return isControlInst(decodeInstruction(instruction));
	}
	
	public static boolean canBeSkipped(EnumAllInsts instType) {
		if (instType == null || instType == EnumAllInsts.Empty || instType == EnumAllInsts.Comment || instType == EnumAllInsts.Directive)
			return true;
		return false;
		// Dont skip unknown. instType == EnumAllInsts.Unknown
	}
	
	public static boolean canBeSkipped(String instruction) {
		return canBeSkipped(decodeInstruction(instruction));
	}
	
	public static String getLabelName(String instruction) {
		if (instruction == null || instruction.equals("")) return "";
		EnumAllInsts instType = PTXUtil.decodeInstruction(instruction);
		if (instType == EnumAllInsts.Branch) {
			int start = UtilsMisc.maxInt(instruction.lastIndexOf('\t'), instruction.lastIndexOf(' '));
			int end = instruction.indexOf(';');
			return instruction.substring(start, end).trim();
		}
		else if (instType == EnumAllInsts.Label) {
			int end = instruction.indexOf(':');
			return instruction.substring(0, end).trim();
		}
		return "";
	}
	
	
//	public static boolean isBranch(String instruction) {
//		if (instruction == null || instruction.equals(""))
//			return false;
//		instruction = instruction.trim();
//		EnumAllInsts instType = decodeInstruction(instruction);
//		if (instType == EnumAllInsts.Branch)
//			return true;
//		return false;
//	}
//	
//	public static boolean isReturn(String instruction) {
//		if (instruction == null || instruction.equals(""))
//			return false;
//		instruction = instruction.trim();
//		EnumAllInsts instType = decodeInstruction(instruction);
//		if (instType == EnumAllInsts.Return)
//			return true;
//		return false;
//	}
//	
//	public static boolean isLabel(String instruction) {
//		if (instruction == null || instruction.equals(""))
//			return false;
//		instruction = instruction.trim();
//		EnumAllInsts instType = decodeInstruction(instruction);
//		if (instType == EnumAllInsts.Label)
//			return true;
//		return false;
//	}
	
}