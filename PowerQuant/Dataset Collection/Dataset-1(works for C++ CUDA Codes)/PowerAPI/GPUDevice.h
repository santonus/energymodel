#ifndef GPUDEVICE_H
#define GPUDEVICE_H
#include <nvml.h>
#include <cuda_runtime.h>
#include <iostream>
#include <stdlib.h>
#include<fstream> 
#include <sys/time.h>
#include<unistd.h>
#include<pthread.h>
class GPUDevice {
public:
    // Prints NVML initialization error along with the stage it occurred in
    void Error(nvmlReturn_t err, const std::string& stage) ;
    // Constructor: Initializes NVML and retrieves GPU handle by index default is 0
    GPUDevice(int deviceIndex,char* kernelName,int gs,int bs);
    // Returns current power usage in Watts
    unsigned int getPower();

    // Returns energy consumed by GPU since start in milli Joules
    unsigned long long getEnergy();

    //Returns current temperature of GPU
    unsigned int getTemp();

    //Starts Power(int Watts) Measurement of GPU 
    void startReading();

    //Stops Power(int Watts) Measurement of GPU and saved it to Power_data.txt along with time taken for execution
    void stopReading();

private:
    // Variable to store NVML return code
    nvmlReturn_t nvmlResult;
    // Variable to store device ID of the GPU being inspected
    nvmlDevice_t nvmlDeviceID;
    //start energy Reading 
    unsigned long long startEnergy;
    //stop energy Reading 
    unsigned long long stopEnergy;
    //time stuct
struct timespec begin, end;
    //Kernel Name
    std::string KernelName;
    //Grid Size
    int gs;
    //Block Size
    int bs;
    //Max Power
    unsigned int maxPower;
    //Min Power 
    unsigned int minPower;
    //Avg Power
    long long int avgPower;
    //Tread ID of GPU Max/Min Power Function
    pthread_t ptid;
    //Finds the maximum and minimum power by continously reading power values
    friend void* findMaxMinPower(void *);
    
}; 
#endif