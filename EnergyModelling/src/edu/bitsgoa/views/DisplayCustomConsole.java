package edu.bitsgoa.views;
import org.eclipse.swt.widgets.Display;
/**
 * This class is used for displaying on the custom console "Energy Estimation Results"
 */

public class DisplayCustomConsole {
	/**
	 * Display on the custom console. This method is used by any class that wishes to display any data to the user
	 * @param message the message to display on the custom console
	 * @return
	 */
	public static void display(final String message,final boolean changeLine){
		Display.getDefault().asyncExec(new Runnable() {
			@Override
			public void run() {
				if(CustomConsole.used==0) return;	//If custom console has not been opened, then there is no point in displaying on it. Simply, return.
				CustomConsole.text.append(message);
				if(changeLine)
					CustomConsole.text.append("\n");
			}
		});
	}

}
