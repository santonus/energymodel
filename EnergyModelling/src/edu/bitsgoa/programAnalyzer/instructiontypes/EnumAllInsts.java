package edu.bitsgoa.programAnalyzer.instructiontypes;

public enum EnumAllInsts {
	KernelStart, KernelEnd, Label,
	Computation, MemAccess, Sync, Branch, Return,
	Comment, Directive, Empty, Unknown
}