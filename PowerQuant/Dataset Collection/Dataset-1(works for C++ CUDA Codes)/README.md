# How to Use PowerAPI

To use PowerAPI in your project, follow these steps:

1. Add PowerAPI to the path by executing the following command in your terminal:
    ```
    LD_LIBRARY_PATH="/path/of/Folder/PowerAPI"
    ```

2. Import PowerAPI in your CUDA file by adding this at start of .cu file.
    ```
    #include "GPUDevice.h"
    ```

3. Create a new GPUDevice Object and wrap your kernel with the startReading and stopReading functions as shown in the code below:
    ```
    GPUDevice g1 = GPUDevice(<GPU Device ID>,<Kernel Name>,<Grid Size>,<Block Size>);
    g1.startReading();
    <<Cuda Kernel Calls>>
    g1.stopReading();
    ```

4. Run the file, and the results will be saved in a new text file with the name of the kernel.Sample output will look like this.
Unit of Power used is Watt.
Unit of Energy used is milliJoules.
    ```
    KernelName,GridSize,BlockSize,MaxPower,MinPower,AvgPower,Time,Energy,
    reluKernel, 86436, 256, 84, 64, 76 , 67603944.000000, 5154, 
    ```
