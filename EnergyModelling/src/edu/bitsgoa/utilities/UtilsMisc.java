package edu.bitsgoa.utilities;

import java.lang.Character;

public final class UtilsMisc {
	
	public static String dashedLine = "\n-----------------------------------------------------------------------------------------------------\n";
	public static String smallerLine = "\n-----------------------------------------------------------\n";
	public static String blankSpace = "\n\n"; //"\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n";
	public static String independentStr = "\t\t\t**Independent**\n";
	
	public static boolean isNumberOrLetter(char character) {
		if (isNumber(character))
			return true;
		if (isLetter(character))
			return true;
		return false;
	}
	
	public static boolean isNumber(char character) {
		if (character == '0' || character == '1' || character == '2' || character == '3' || character == '4'
		|| character == '5' || character == '6' || character == '7' || character == '8' || character == '9')
			return true;
		return false;
	}
	
	public static boolean isLetter(char character) {
		character = Character.toLowerCase(character);
		switch (character) {
			case 'a': case 'b': case 'c': case 'd': case 'e': case 'f':
			case 'g': case 'h': case 'i': case 'j': case 'k': case 'l':
			case 'm': case 'n': case 'o': case 'p': case 'q': case 'r':
			case 's': case 't': case 'u': case 'v': case 'w': case 'x':
			case 'y': case 'z': return true;
			default: return false;
		}
	}

	public static boolean divideByZeroCheck(double arg, String varName) {
		if (arg == 0) {
			System.out.println("Divide by zero! Check: " + varName);
			return true;
		}
		return false;
	}
	
	public static int maxInArray(int[] arrayName, int arraySize) {
		int maximum = arrayName[0];
		for (int i = 1; i < arraySize; i++) {
			if (arrayName[i] > maximum)
				maximum = arrayName[i];
		}
		return maximum;
	}
	
	public static int sumArray(int[] arrayName, int arraySize) {
		int sum = arrayName[0];
		for (int i = 1; i < arraySize; i++) {
			sum += arrayName[i];
		}
		return sum;
	}
	
	public static double minDouble(double arg1, double arg2) {
		if (arg1 <= arg2)
			return arg1;
		return arg2;
	}
	
	public static long minLong(long arg1, long arg2) {
		if (arg1 <= arg2)
			return arg1;
		return arg2;
	}
	
	public static double minDouble(double arg1, double arg2, double arg3) {
		double result;
		result = UtilsMisc.minDouble(arg1, arg2);
		result = UtilsMisc.minDouble(result, arg3);
		return result;
	}
	
	public static int minInt(int arg1, int arg2) {
		if (arg1 <= arg2)
			return arg1;
		return arg2;
	}
	
	public static int maxInt(int arg1, int arg2) {
		if (arg1 >= arg2)
			return arg1;
		return arg2;
	}
	
	private UtilsMisc() {
		throw new RuntimeException("Do not instantiate UtilsMisc class! It has static methods only.");
	}
		
}