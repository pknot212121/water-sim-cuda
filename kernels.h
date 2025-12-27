#include "common.cuh"
void summonTestKernel(Particles p,int number);
void summonOccupancyCheckInit(Particles p,int number,size_t cellsPerPage,bool* occupancy);