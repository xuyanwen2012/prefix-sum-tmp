#include <algorithm>
#include <iostream>
#include <numeric>
#include <random>
#include <vector>

#include "helper.cuh"

#include <cub/cub.cuh>

// Single block scan
template <typename T> __global__ void scan(T *g_odata, T *g_idata, int n) {
  extern __shared__ T temp[]; // allocated on invocation

  int thid = threadIdx.x;
  int pout = 0, pin = 1; // Load input into shared memory.

  // This is exclusive scan, so shift right by one
  // and set first element to 0
  temp[pout * n + thid] = (thid > 0) ? g_idata[thid - 1] : 0;

  __syncthreads();

  // 1, 2, 4, 8, 16, ... 2^(log n)
  for (int offset = 1; offset < n; offset *= 2) {
    pout = 1 - pout; // swap double buffer indices
    pin = 1 - pout;
    if (thid >= offset)
      temp[pout * n + thid] += temp[pin * n + thid - offset];
    else
      temp[pout * n + thid] = temp[pin * n + thid];
    __syncthreads();
  }

  g_odata[thid] = temp[pout * n + thid]; // write output
}

#define NUM_BANKS 16

#define LOG_NUM_BANKS 4
#define CONFLICT_FREE_OFFSET(n) ((n) >> NUM_BANKS + (n) >> (2 * LOG_NUM_BANKS))

template <typename T> __global__ void prescan(T *g_odata, T *g_idata, int n) {
  extern __shared__ T temp[]; // allocated on invocation
  int thid = threadIdx.x;
  int offset = 1;

  // temp[2 * thid] = g_idata[2 * thid]; // load input into shared memory
  // temp[2 * thid + 1] = g_idata[2 * thid + 1];

  int ai = thid;
  int bi = thid + (n / 2);
  int bankOffsetA = CONFLICT_FREE_OFFSET(ai);
  int bankOffsetB = CONFLICT_FREE_OFFSET(bi);
  temp[ai + bankOffsetA] = g_idata[ai];
  temp[bi + bankOffsetB] = g_idata[bi];

  for (int d = n >> 1; d > 0; d >>= 1) // build sum in place up the tree
  {
    __syncthreads();
    if (thid < d) {
      int ai = offset * (2 * thid + 1) - 1;
      int bi = offset * (2 * thid + 2) - 1;
      temp[bi] += temp[ai];
    }
    offset *= 2;
  }

  // if (thid == 0) {
  //   temp[n - 1] = 0;
  // }

  if (thid == 0) {
    temp[n - 1 + CONFLICT_FREE_OFFSET(n - 1)] = 0;
  }

  for (int d = 1; d < n; d *= 2) // traverse down tree & build scan
  {
    offset >>= 1;
    __syncthreads();
    if (thid < d) {
      int ai = offset * (2 * thid + 1) - 1;
      int bi = offset * (2 * thid + 2) - 1;
      T t = temp[ai];
      temp[ai] = temp[bi];
      temp[bi] += t;
    }
  }
  __syncthreads();
  // g_odata[2 * thid] = temp[2 * thid]; // write results to device memory
  // g_odata[2 * thid + 1] = temp[2 * thid + 1];

  g_odata[ai] = temp[ai + bankOffsetA];
  g_odata[bi] = temp[bi + bankOffsetB];
}

constexpr auto kNumItemsPerThread = 1;

// A stateful callback functor that maintains a running prefix to be applied
// during consecutive scan operations.
struct BlockPrefixCallbackOp {
  // Running prefix
  int running_total;

  // Constructor
  __device__ BlockPrefixCallbackOp(int running_total)
      : running_total(running_total) {}

  // Callback operator to be entered by the first warp of threads in the block.
  // Thread-0 is responsible for returning a value for seeding the block-wide
  // scan.
  __device__ int operator()(int block_aggregate) {
    int old_prefix = running_total;
    running_total += block_aggregate;
    return old_prefix;
  }
};

__device__ void CTA_SYNC() { __syncthreads(); }

__global__ void ExampleKernel(int *d_idata, int *d_odata, int num_items) {
  // Specialize BlockScan for a 1D block of 128 threads
  typedef cub::BlockScan<int, 128> BlockScan;

  // Allocate shared memory for BlockScan
  __shared__ typename BlockScan::TempStorage temp_storage;

  // Initialize running total
  BlockPrefixCallbackOp prefix_op(0);

  // Have the block iterate over segments of items
  for (int block_offset = 0; block_offset < num_items; block_offset += 128) {
    // Load a segment of consecutive items that are blocked across threads
    int thread_data = d_idata[block_offset];

    // Collectively compute the block-wide exclusive prefix sum
    BlockScan(temp_storage).ExclusiveSum(thread_data, thread_data, prefix_op);
    CTA_SYNC();

    // Store scanned items to output segment
    d_odata[block_offset] = thread_data;
  }
}

__global__ void ExampleKernel2(int *d_idata, int *d_odata, int num_items) {
  constexpr auto kNumItemsPerThread = 4;
  constexpr auto kNumThreads = 128;
  constexpr auto kTileSize = kNumThreads * kNumItemsPerThread;

  // Specialize BlockLoad, BlockStore, and BlockScan for a 1D block of 128
  // threads, 4 ints per thread
  using BlockLoad = cub::BlockLoad<int, kNumThreads, kNumItemsPerThread,
                                   cub::BLOCK_LOAD_DIRECT>;
  using BlockStore = cub::BlockStore<int, kNumThreads, kNumItemsPerThread,
                                     cub::BLOCK_STORE_DIRECT>;
  using BlockScan = cub::BlockScan<int, kNumThreads>;

  // Allocate aliased shared memory for BlockLoad, BlockStore, and BlockScan
  __shared__ union {
    typename BlockLoad::TempStorage load;
    typename BlockScan::TempStorage scan;
    typename BlockStore::TempStorage store;
  } temp_storage;

  // Initialize running total
  BlockPrefixCallbackOp prefix_op(0);

  // Have the block iterate over segments of items

  // use this to control over the logical blocks
  for (int block_offset = 0; block_offset < num_items;
       block_offset += kTileSize) {
    // Load a segment of consecutive items that are blocked across threads
    int thread_data[4];
    BlockLoad(temp_storage.load).Load(d_idata + block_offset, thread_data);
    CTA_SYNC();

    // Collectively compute the block-wide exclusive prefix sum
    int block_aggregate;
    BlockScan(temp_storage.scan)
        .ExclusiveSum(thread_data, thread_data, prefix_op);
    CTA_SYNC();

    // Store scanned items to output segment
    BlockStore(temp_storage.store).Store(d_odata + block_offset, thread_data);
    CTA_SYNC();
  }
}

// tile size
__global__ void k_CountUnique(int *keys, int *num_unique_out, const int n) {
  int first = 0;
  int last = n;

  auto result = first;
  while (++first != last) {
    if (!(keys[result] == keys[first]) && ++result != first) {
      keys[result] = keys[first];
    }
  }

  *num_unique_out = ++result;
}

__global__ void k_CountUniqueCub(int *in, int *out, const int n) {
  using BlockLoad =
      cub::BlockLoad<int, 32, kNumItemsPerThread, cub::BLOCK_LOAD_DIRECT>;
  // using BlockDiscontinuity = cub::BlockDiscontinuity<int, 128>;
  using BlockStore =
      cub::BlockStore<int, 32, kNumItemsPerThread, cub::BLOCK_STORE_DIRECT>;

  __shared__ union {
    typename BlockLoad::TempStorage load_storage;
    // typename BlockDiscontinuity::TempStorage discontinuity_storage;
    typename BlockStore::TempStorage store_storage;
  } temp_storage;

  int thread_data[kNumItemsPerThread];

  BlockLoad(temp_storage.load_storage).Load(in, thread_data);

  BlockStore(temp_storage.store_storage).Store(out, thread_data);

  // BlockDiscontinuity(temp_storage.discontinuity_storage)
  //     .FlagHeads(head_flags, thread_data, cub::Inequality());
}

__global__ void emptyKernel() {}

int main() {
  constexpr auto N = 1000;

  // CPU
  // std::vector<int> cpu_input(N);
  // std::mt19937 gen(114514);
  // std::uniform_int_distribution<int> dis(0, N);
  // std::generate(cpu_input.begin(), cpu_input.end(), [&]() { return dis(gen);
  // }); std::sort(cpu_input.begin(), cpu_input.end());

  // GPU
  int *u_input;
  MallocManaged(&u_input, N);
  int *u_output;
  MallocManaged(&u_output, N);

  std::fill_n(u_input, N, 1);

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  emptyKernel<<<1, 1>>>();

  cudaEventRecord(start);

  ExampleKernel2<<<1, 128>>>(u_input, u_output, N);
  // checkCudaErrors(cudaDeviceSynchronize());

  cudaEventRecord(stop);
  cudaEventSynchronize(stop);

  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  std::cout << "Elapsed time (GPU): " << milliseconds << "ms" << std::endl;

  // // print input and output side by side
  // for (int i = 0; i < N; i++) {
  //   std::cout << i << ":\t" << u_input[i] << "\t" << u_output[i] <<
  //   std::endl;
  // }

  CUDA_FREE(u_input);
  CUDA_FREE(u_output);
  return 0;
}