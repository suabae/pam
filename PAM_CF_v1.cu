// written by Sua Bae (3/09/2021)

#include <cuda.h>
#include <cuda_runtime.h>

__constant__ float 			trans_aElePos[256]; // transducer element position
__constant__ unsigned int 		trans_nNumEle; // num of elements

__constant__ unsigned int 		rf_nSample; 			// num of RF samples
__constant__ unsigned int 		rf_nChannel;			// num of channels of RF data
__constant__ float 			rf_nOffsetDelay_m; 	// [m] offset of input data (to be added to tx delay)
__constant__ float 			rf_nMeter2Pixel;  	// [pixel/m]  = sampling frequency / nSoundSpeed; 

__constant__ unsigned int 		g_nXdim; 			// num of x points 
__constant__ unsigned int 		g_nZdim; 			// num of z points 
__constant__ float 			g_dx; 				// [m] pixel size in x
__constant__ float 			g_dz; 				// [m] pixel size in z
__constant__ float 			g_nXstart; 			// [m] x coordinate of the first lateral pixel
__constant__ float 			g_nZstart; 			// [m] z coordinate of the first axial pixel
__constant__ unsigned int 		g_nTdim; 			// num of time points to be integrated


//***	THREADBLOCK DIMENSION ***
//	ThreadBlockSize =[N,1,1];
//	GridSize = [ceil(g_nTdim/N),g_nXdim,g_nZdim];
__global__ void _PAM_CF(float* CavMap_tzx, float* RfData_sc)		
{
    int tidx = blockIdx.x * blockDim.x + threadIdx.x;
    int xidx = blockIdx.y;
	int zidx = blockIdx.z;
	
	if (tidx < g_nTdim) {	
        float nX = (float)xidx * g_dx + g_nXstart; // x position of image grid [m]
        float nZ = (float)zidx * g_dz + g_nZstart + 1e-20; // z position of image grid [m]				

        float nTxDelay_m; // [m] distance from FUS transducer to the imaging point (RX delay)
        float nRxDelay_m; // [m] distance from imaging point to element (RX delay)
        float nDelay_px; // [pixel] round trip delay 
        float nDelay_int; // [pixel] integer part of round trip delay 
        float nDelay_frc; // [pixel] fractional part of round trip delay 
        int nAdd; // address of the rf datum
        float nInterpVal; // interpolated value
        float nCompenVal; // spherical spreading compensated value
        float nCF; // coherence factor

        nTxDelay_m = nZ;			

        float nChannelSum = 0; // refresh
        float nSumOfSquared = 0;
        #pragma unroll	
        for (int cidx = 0; cidx < rf_nChannel; cidx++) { // channel index
            nRxDelay_m = sqrt((nX-trans_aElePos[cidx])*(nX-trans_aElePos[cidx]) + nZ*nZ); // [m] distance from imaging point to element
            nDelay_px = (nTxDelay_m + nRxDelay_m + rf_nOffsetDelay_m)*rf_nMeter2Pixel; // [sample]						

            nDelay_int = (int)(nDelay_px);
            nDelay_frc = nDelay_px - nDelay_int;
            nAdd = cidx*rf_nSample + nDelay_int + tidx; 
            nInterpVal =  RfData_sc[nAdd]*(1-nDelay_frc) + RfData_sc[nAdd+1]*nDelay_frc; // interpolated sample for each channel and time point
            nCompenVal = sqrt(nRxDelay_m)*nInterpVal;
            nChannelSum = nChannelSum + nCompenVal; // stack onto the channel sum
            nSumOfSquared = nSumOfSquared + nCompenVal*nCompenVal;
        }

        nCF = (nChannelSum*nChannelSum)/(rf_nChannel*nSumOfSquared);
        CavMap_tzx[tidx + zidx*g_nTdim + xidx*g_nTdim*g_nZdim] = (nCF*nChannelSum)*(nCF*nChannelSum);

	}
}

