#include <cstdio>
#include <cstdlib>
#include <vector>

__global__ void init(int *bucket, int range){
	int i = blockIdx.x * blockDim.x + threadIdx.x;
 	if(i>=range) return;
	bucket[i]=0;
}

__global__ void count(int *bucket, int *key, int n){
	int i = blockIdx.x * blockDim.x + threadIdx.x;
 	if(i>=n) return;
	atomicAdd(&bucket[key[i]], 1);
}

__global__ void sort(int *bucket, int *key, int range, int n){
	int i = blockIdx.x * blockDim.x + threadIdx.x;
 	if(i>=n) return;
	for(int j= range-1; j>=0; j--){
		if(bucket[j-1] <= i){
			key[i] = j;
			return;
		}
	}
}

__global__ void scan(int *bucket, int *tmp, int range){
	int i = blockIdx.x * blockDim.x + threadIdx.x;
 	if(i>=range) return;
	for(int j=1; j<range; j<<=1) {
 		tmp[i] = bucket[i];
 		__syncthreads();
 		bucket[i] += tmp[i-j];
 		__syncthreads();
	}
}

int main() {
  int n = 50;
  int range = 5;
	int *key; 
  cudaMallocManaged(&key, n*sizeof(int));
  for (int i=0; i<n; i++) {
    key[i] = rand() % range;
    printf("%d ",key[i]);
  }
  printf("\n");

	const int M = 1024;

  int *bucket; 
  cudaMallocManaged(&bucket, range*sizeof(int));
  int *tmp; 
  cudaMallocManaged(&tmp, range*sizeof(int));
	init<<<(range+M-1)/M,M>>>(bucket,range);
	cudaDeviceSynchronize();
	count<<<(n+M-1)/M,M>>>(bucket,key,n);
	cudaDeviceSynchronize();
	scan<<<(range+M-1)/M,M>>>(bucket,tmp,range);
	cudaDeviceSynchronize();
	sort<<<(n+M-1)/M,M>>>(bucket,key,range,n);
	cudaDeviceSynchronize();


  for (int i=0; i<n; i++) {
    printf("%d ",key[i]);
  }
  printf("\n");
}