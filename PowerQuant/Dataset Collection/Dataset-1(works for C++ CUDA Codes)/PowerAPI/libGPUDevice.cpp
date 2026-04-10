#include <nvml.h>
#include <cuda_runtime.h>
#include <iostream>
#include <stdlib.h>
#include <fstream>
#include <sys/time.h>
#include "GPUDevice.h"
#include <chrono>
#include <unistd.h>
#include <string>
#include <pthread.h>
// Prints NVML initialization error along with the stage it occurred in
void GPUDevice::Error(nvmlReturn_t err, const std::string &stage)
{
    printf("NVML Init fail: %s\n at %s", nvmlErrorString(err), stage.c_str());
}

// Constructor: Initializes NVML and retrieves GPU handle by index default is 0
GPUDevice::GPUDevice(int deviceIndex = 0, char *kernelName = "Not_Specified", int GS = -1, int BS = -1)
{
    nvmlResult = nvmlInit_v2();
    gs = GS;
    bs = BS;
    KernelName = kernelName;
    maxPower = 0;
    minPower = 300;
    avgPower = 0;
    if (NVML_SUCCESS == nvmlResult)
    {
        nvmlResult = nvmlDeviceGetHandleByIndex(deviceIndex, &nvmlDeviceID);
        if (NVML_SUCCESS == nvmlResult)
        {
            printf("NVML Initialization successful with GPU %d.\n", deviceIndex);
        }
        else
        {
            Error(nvmlResult, "nvmlDeviceGetHandleByIndex");
        }
    }
    else
    {
        Error(nvmlResult, "nvmlInit");
    }
}

// Returns current power usage in Watts
unsigned int GPUDevice::getPower()
{
    unsigned int power;
    nvmlResult = nvmlDeviceGetPowerUsage(nvmlDeviceID, &power);
    if (NVML_SUCCESS == nvmlResult)
    {
        return power / 1000;
    }
    else
    {
        Error(nvmlResult, "nvmlDeviceGetPowerUsage");
        return 0;
    }
}

// Returns energy consumed by GPU since start in Joules
unsigned long long GPUDevice::getEnergy()
{
    unsigned long long energy;
    nvmlResult = nvmlDeviceGetTotalEnergyConsumption(nvmlDeviceID, &energy);
    if (NVML_SUCCESS == nvmlResult)
    {
        return energy;
    }
    else
    {
        Error(nvmlResult, "nvmlDeviceGetTotalEnergyConsumption");
        return 0;
    }
}

// Returns current temperature of GPU
unsigned int GPUDevice::getTemp()
{
    unsigned int temp;
    nvmlResult = nvmlDeviceGetTemperature(nvmlDeviceID, NVML_TEMPERATURE_GPU, &temp);
    return temp;
}

// Finds the maximum and minimum power by continously reading power values
void *findMaxMinPower(void *obj_param)
{
    GPUDevice *thr = ((GPUDevice *)obj_param);
    long long avg = 0;
    long long count = 0;
    while (true)
    {
        sleep(0.000001);
        unsigned int p = thr->getPower();
        avg += p;
        count++;
        if (p > thr->maxPower)
        {
            thr->maxPower = p;
        }
        if (p < thr->minPower)
        {
            thr->minPower = p;
        }
        thr->avgPower = (avg / count);
    }
}

// Starts Power(int Watts) Measurement of GPU and saved it to Power_data.txt along with time taken for execution
void GPUDevice::startReading()
{
    pthread_create(&ptid, NULL, &findMaxMinPower, this);
    startEnergy = getEnergy();
    clock_gettime(CLOCK_MONOTONIC, &begin);
};

// Stops Power(int Watts) Measurement of GPU and saved it to Power_data.txt along with time taken for execution
void GPUDevice::stopReading()
{
    // // This call waits for all of the submitted GPU work to complete
    // cudaDeviceSynchronize();
    // get the power value after execution of program
    clock_gettime(CLOCK_MONOTONIC, &end);
    uint64_t time = 1e9 * (end.tv_sec - begin.tv_sec) + (end.tv_nsec -
                                                         begin.tv_nsec);
    stopEnergy = getEnergy();
    pthread_cancel(ptid);
    unsigned long long energyConsumed = (stopEnergy - startEnergy);
    nvmlResult = nvmlShutdown();
    if (NVML_SUCCESS != nvmlResult)
    {
        printf("Failed to shut down NVML: %s\n", nvmlErrorString(nvmlResult));
        exit(0);
    }
    std::string fileName = "Power_data_" + KernelName + ".txt";
    FILE *fp = fopen(fileName.c_str(), "a");
    if (fp == NULL)
    {
        fp = fopen(fileName.c_str(), "w");
    }
    else
    {
        // The output file stores power in Watts and execution time in milli-seconds.
        fprintf(fp, "%s, %d, %d, %d, %d, %lld , %Lf, %lld, \n", KernelName.c_str(), gs, bs, (maxPower), (minPower), (avgPower), (long double)(time), energyConsumed);
    }
    fclose(fp);
};
