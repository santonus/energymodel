package edu.bitsgoa.driver;
/**
 * This class obtains the path to the currently opened .cu file. Then, it generates a .ptx file, and calls the energy estimator program
 */
import org.eclipse.core.commands.AbstractHandler;
import org.eclipse.core.commands.ExecutionEvent;
import org.eclipse.core.commands.ExecutionException;
import org.eclipse.core.commands.IHandler;
import org.eclipse.core.resources.IFile;
import org.eclipse.core.resources.ResourcesPlugin;
import org.eclipse.ui.IWorkbenchPart;
import org.eclipse.ui.PlatformUI;
import edu.bitsgoa.properties.Parameters;
import edu.bitsgoa.utilities.BetterRunProcess;


public class EnergyEstimator extends AbstractHandler implements IHandler{

	public static boolean return_val=false;
	public static String pathToCurrentFile=null;
	public static String fileName;
	private String builderDirectory;

	public Object execute(ExecutionEvent event) throws ExecutionException {
		//Run a new process: nvcc -ptx file.cu
		pathToCurrentFile=getPath();
		fileName=pathToCurrentFile.substring(pathToCurrentFile.lastIndexOf('/')+1,pathToCurrentFile.length());
		builderDirectory=pathToCurrentFile.substring(0,pathToCurrentFile.lastIndexOf('/'));
		BetterRunProcess process=new BetterRunProcess();
		String[] cmd=new String[3];
		cmd[0]="nvcc";
		cmd[1]="-ptx";
		cmd[2]=fileName;
		process.runProcessBuilderInDifferentDirectory(cmd,builderDirectory,1,0,0,"");
		//fire the energy estimation program
		MainRunner.initializeAll();
		MainRunner.runModel();
		return null;
	}
	/**
	 * This methods returns the path to the currently opened .cu file in the editor.
	 * @param
	 * @return	path to the currently opened .cu file in the editor.
	 */
	public String getPath(){
		IWorkbenchPart workbenchPart = PlatformUI.getWorkbench().getActiveWorkbenchWindow().getActivePage().getActivePart(); 
		IFile file = (IFile) workbenchPart.getSite().getPage().getActiveEditor().getEditorInput().getAdapter(IFile.class);
		String path=ResourcesPlugin.getWorkspace().getRoot().getLocation().toString()+file.getFullPath();
		return path;
	}	
	public boolean isEnabled(){
		if(Preparation.used==1 && Parameters.timesused>0) return true;
		else return false;
	}


}
