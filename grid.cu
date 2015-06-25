#define CUB_STDERR

#include <algorithm>
#include <cstdio>
#include <iostream>
#include <iomanip>

#include <cub/cub.cuh>
#include <thrust/scan.h>
#include <thrust/device_vector.h>

#include "benchmark.H"

using namespace cub;

#define MAX_WG_SIZE 1024

double buildTime = 0.0;

bool clInfoStats = true;
bool clBenchmark = true;

static CachingDeviceAllocator g_allocator(true);

static int                    ptx_version;
static int                    dev(0);

//--------------------------------------------------------------------------------
typedef struct {
  short lox, loy, loz;
  short hix, hiy, hiz;
} T_AABB;

// outputs
unsigned N;
unsigned NO;

unsigned *G = NULL;
unsigned *O = NULL;

// temporaries
static int3                   M;
static T_AABB                 *d_boxes;
static float                  *d_Px, *d_Py, *d_Pz;
static float                  *d_planes;
texture<float4, 1>            t_planes;

//--------------------------------------------------------------------------------
struct GpuTimer
{
  cudaEvent_t start;
  cudaEvent_t stop;

  GpuTimer()
  {
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
  }

  ~GpuTimer()
  {
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
  }

  void Start()
  {
    cudaEventRecord(start, 0);
  }

  void Stop()
  {
    cudaEventRecord(stop, 0);
  }

  float ElapsedMillis()
  {
    float elapsed;
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed, start, stop);
    return elapsed;
  }
};

static void setGrid(int maxThreads, int numThreads, int& blocksPerGrid, int& threadsPerBlock)
{
  threadsPerBlock = maxThreads;
  blocksPerGrid = (numThreads + threadsPerBlock - 1) / threadsPerBlock;
}

//--------------------------------------------------------------------------------
__device__ static inline int coords(const int3 M, const int3 id) {
  return (((M.y * id.z) + id.y) * M.x) + id.x;
}
__device__ static inline int3 global_coords(const T_AABB& box, const int rid) {
  const int3 m = (int3){box.hix-box.lox+1,box.hiy-box.loy+1,box.hiz-box.loz+1};
  int x, y, z;
  z = rid / (m.x * m.y);
  y = (rid - z * m.x * m.y) / m.x;
  x = rid - m.x * (y + m.y * z);

  return (int3){box.lox+x, box.loy+y, box.loz+z};
}

//--------------------------------------------------------------------------------
__device__
static bool collides(const float4 P, const float3 lo, const float3 hi)
{
  float3 pos;      // farthest vertex in the direction of  N
  float3 neg;      // farthest vertex in the direction of -N

  if (P.x>0.0f)
    if (P.y>0.0f)
      if (P.z>0.0f) {
        pos = hi;
        neg = lo;
      } else {
        pos = (float3){hi.x, hi.y, lo.z};
        neg = (float3){lo.x, lo.y, hi.z};
      }
    else
      if (P.z>0.0f) {
        pos = (float3){hi.x, lo.y, hi.z};
        neg = (float3){lo.x, hi.y, lo.z};
      } else {
        pos = (float3){hi.x, lo.y, lo.z};
        neg = (float3){lo.x, hi.y, hi.z};
      }
  else
    if (P.y>0.0f)
      if (P.z>0.0f) {
        pos = (float3){lo.x, hi.y, hi.z};
        neg = (float3){hi.x, lo.y, lo.z};
      } else {
        pos = (float3){lo.x, hi.y, lo.z};
        neg = (float3){hi.x, lo.y, hi.z};
      }
    else
      if (P.z>0.0f) {
        pos = (float3){lo.x, lo.y, hi.z};
        neg = (float3){hi.x, hi.y, lo.z};
      } else {
        pos = lo;
        neg = hi;
      }

  if ((P.x*pos.x+P.y*pos.y+P.z*pos.z + P.w) < 0.0f)
    return false;
  if ((P.x*neg.x+P.y*neg.y+P.z*neg.z + P.w) > 0.0f)
    return false;
  return true;
}

//--------------------------------------------------------------------------------
__global__
static void count_cells(uint *V, const T_AABB *boxes, int length)
{
  int i = blockDim.x * blockIdx.x + threadIdx.x;

  if (i >= length)
    return;

  const T_AABB& box = boxes[i];
  V[i] = uint(1+box.hix-box.lox)*uint(1+box.hiy-box.loy)*uint(1+box.hiz-box.loz);
}

//--------------------------------------------------------------------------------
__global__
static void avedev_samples(float *results, const uint *samples, const float avgItemsPerNonEmptyCell, int nonEmpty)
{
  int i = blockDim.x * blockIdx.x + threadIdx.x;

  if (i >= nonEmpty)
    return;

  float diff = fabs((float)samples[i] - avgItemsPerNonEmptyCell);
  results[i] = diff;
}

__global__
static void intersected_cells(uint *V, const T_AABB *boxes, const float *Px, const float *Py, const float *Pz, int length)
{
  int i = blockDim.x * blockIdx.x + threadIdx.x;

  if (i >= length)
    return;

  const T_AABB& box = boxes[i];
  float4 P = tex1Dfetch(t_planes, i);

  uint cnt = 0;
  for (short x=box.lox; x<=box.hix; ++x) {
    for (short y=box.loy; y<=box.hiy; ++y) {
      for (short z=box.loz; z<=box.hiz; ++z) {
//        float3 lo = (float3){Px[x], Py[y], Pz[z]};
//        float3 hi = (float3){Px[x+1], Py[y+1], Pz[z+1]};
//        if (collides(P, lo, hi)) {
        cnt++;
//        }
      }
    }
  }
  V[i] = cnt;
}


//--------------------------------------------------------------------------------
extern "C" void cudaInit(void) {
  cudaError_t error = cudaSuccess;
  do {
    int deviceCount;
    error = CubDebug(cudaGetDeviceCount(&deviceCount));
    if (error) break;

    if (deviceCount == 0) {
      fprintf(stderr, "No devices supporting CUDA.\n");
      exit(1);
    }

    error = CubDebug(cudaSetDevice(dev));
    if (error) break;

    size_t free_physmem, total_physmem;
    CubDebugExit(cudaMemGetInfo(&free_physmem, &total_physmem));

    error = CubDebug(cub::PtxVersion(ptx_version));
    if (error) break;

    cudaDeviceProp deviceProp;
    error = CubDebug(cudaGetDeviceProperties(&deviceProp, dev));
    if (error) break;

    if (deviceProp.major < 1) {
      fprintf(stderr, "Device does not support CUDA.\n");
      exit(1);
    }
/*
    printf("Device %d: %s (PTX version %d, SM %d.%d, %d SMs, %ld free / %ld total MB RAM, ECC %s)\n",
    dev,
    deviceProp.name,
    ptx_version,
    deviceProp.major, deviceProp.minor,
    deviceProp.multiProcessorCount,
    free_physmem / (1024*1024),
    total_physmem / (1024*1024),
    (deviceProp.ECCEnabled) ? "on" : "off");
    fflush(stdout);
*/
  } while (0);
  CubDebugExit(error);
}

//--------------------------------------------------------------------------------
static void begin(unsigned nobjs, T_AABB *boxes, float *planes, float *Px, float *Py, float *Pz, int Mx, int My, int Mz) {
  M = (int3){Mx, My, Mz};

  d_boxes = NULL;
  CubDebugExit(g_allocator.DeviceAllocate((void**)&d_boxes, sizeof(T_AABB)*nobjs));
  CubDebugExit(cudaMemcpy(d_boxes, boxes, sizeof(T_AABB)*nobjs, cudaMemcpyHostToDevice));

  d_Px = NULL;
  CubDebugExit(g_allocator.DeviceAllocate((void**)&d_Px, sizeof(float)*(M.x+1)));
  CubDebugExit(cudaMemcpy(d_Px, Px, sizeof(float)*(M.x+1), cudaMemcpyHostToDevice));
  d_Py = NULL;
  CubDebugExit(g_allocator.DeviceAllocate((void**)&d_Py, sizeof(float)*(M.y+1)));
  CubDebugExit(cudaMemcpy(d_Py, Py, sizeof(float)*(M.y+1), cudaMemcpyHostToDevice));
  d_Pz = NULL;
  CubDebugExit(g_allocator.DeviceAllocate((void**)&d_Pz, sizeof(float)*(M.z+1)));
  CubDebugExit(cudaMemcpy(d_Pz, Pz, sizeof(float)*(M.z+1), cudaMemcpyHostToDevice));

  d_planes = NULL;
  CubDebugExit(g_allocator.DeviceAllocate((void**)&d_planes, sizeof(float)*4*nobjs));
  CubDebugExit(cudaMemcpy(d_planes, planes, sizeof(float)*4*nobjs, cudaMemcpyHostToDevice));

  cudaBindTexture(NULL, t_planes, d_planes, sizeof(float)*4*nobjs);
}

//--------------------------------------------------------------------------------
static void result(uint *d_G, uint *d_O) {
  G = new unsigned[N+1];
  O = new unsigned[NO];
  CubDebugExit(cudaMemcpy(G, d_G, sizeof(uint)*(N+1), cudaMemcpyDeviceToHost));
  CubDebugExit(cudaMemcpy(O, d_O, sizeof(uint)*(NO), cudaMemcpyDeviceToHost));

  cudaDeviceSynchronize();
}


//--------------------------------------------------------------------------------
static void finish(void) {
  if (d_boxes) CubDebugExit(g_allocator.DeviceFree(d_boxes));
  if (d_planes) CubDebugExit(g_allocator.DeviceFree(d_planes));
  if (d_Px) CubDebugExit(g_allocator.DeviceFree(d_Px));
  if (d_Py) CubDebugExit(g_allocator.DeviceFree(d_Py));
  if (d_Pz) CubDebugExit(g_allocator.DeviceFree(d_Pz));

  d_boxes = NULL;
  d_planes = NULL;
  d_Px = d_Py = d_Pz = NULL;
}


//--------------------------------------------------------------------------------
__global__
static void make_object_ids(uint *O, const uint *V, int length)
{
  int i = blockDim.x * blockIdx.x + threadIdx.x;

  if (i >= length)
    return;

  if (i > 0) {
    O[V[i]] = 1u;
  }
}

__global__
static void make_ones(uint *C, int length)
{
  int i = blockDim.x * blockIdx.x + threadIdx.x;

  if (i >= length)
    return;

  C[i] = 1u;
}

__global__
static void make_cell_ids(uint *C, uint *O, int3 M, const T_AABB *boxes, const float *Px, const float *Py, const float *Pz, uint N, int length)
{
  int i = blockDim.x * blockIdx.x + threadIdx.x;

  if (i >= length)
    return;

  uint j = O[i];
  uint k = C[i];

  const int3 id = global_coords(boxes[j], k);

//  float3 lo = (float3){Px[id.x], Py[id.y], Pz[id.z]};
//  float3 hi = (float3){Px[id.x+1], Py[id.y+1], Pz[id.z+1]};

//  float4 P = tex1Dfetch(t_planes, j);

//  if (collides(P, lo, hi)) {
    const uint idx = coords(M, id);
    C[i] = idx;
//  } else {
//    C[i] = N;
//  }
}

__global__
static void nonempty_cells(uint *G, const uint *C, const uint *d_counts, int nonEmpty)
{
  int i = blockDim.x * blockIdx.x + threadIdx.x;

  if (i >= nonEmpty)
    return;

  G[C[i]] = d_counts[i];
}

//--------------------------------------------------------------------------------
extern "C" void buildPG(unsigned nobjs, T_AABB *boxes, float *planes, float *Px, float *Py, float *Pz, int Mx, int My, int Mz) {
  begin(nobjs, boxes, planes, Px, Py, Pz, Mx, My, Mz);

  ////////////////////////////////////////////////////////////////////////////////
  double delta = bench.get_time();

  N = Mx*My*Mz;

  uint *d_V = NULL;
  CubDebugExit(g_allocator.DeviceAllocate((void**)&d_V, sizeof(uint)*(nobjs+1)));

  uint data = 0;
  CubDebugExit(cudaMemcpy(d_V + nobjs, &data, sizeof(uint), cudaMemcpyHostToDevice));

  ////////////////////////////////////////
  GpuTimer countCells;

  cudaDeviceSynchronize();

  countCells.Start();

  int grid, block;
  setGrid(MAX_WG_SIZE, nobjs, grid, block);
  count_cells<<<grid, block>>>(d_V, d_boxes, nobjs);

  void *d_tmp1 = NULL;
  size_t tmp1_bytes = 0;

  CubDebugExit(cub::DeviceScan::ExclusiveSum(d_tmp1, tmp1_bytes, d_V, d_V, nobjs+1));
  CubDebugExit(g_allocator.DeviceAllocate(&d_tmp1, tmp1_bytes));
  CubDebugExit(cub::DeviceScan::ExclusiveSum(d_tmp1, tmp1_bytes, d_V, d_V, nobjs+1));
  cudaDeviceSynchronize();

  countCells.Stop();

  CubDebugExit(cudaMemcpy(&NO, d_V + nobjs, sizeof(uint), cudaMemcpyDeviceToHost));

  DoubleBuffer<uint> d_cs;
  DoubleBuffer<uint> d_os;
  CubDebugExit(g_allocator.DeviceAllocate((void**)&d_cs.d_buffers[0], sizeof(uint)*NO));
  CubDebugExit(g_allocator.DeviceAllocate((void**)&d_os.d_buffers[0], sizeof(uint)*NO));

  uint *d_C, *d_O;
  d_C = d_cs.Current();
  d_O = d_os.Current();

  GpuTimer makePairs;
  cudaDeviceSynchronize();
  makePairs.Start();

  void *d_tmp2 = NULL;
  size_t tmp2_bytes = 0;

  CubDebugExit(cudaMemset(d_O, 0, sizeof(uint)*(NO)));
  setGrid(MAX_WG_SIZE, nobjs, grid, block);
  make_object_ids<<<grid, block>>>(d_O, d_V, nobjs);

  CubDebugExit(cub::DeviceScan::InclusiveSum(d_tmp2, tmp2_bytes, d_O, d_O, NO));
  CubDebugExit(g_allocator.DeviceAllocate(&d_tmp2, tmp2_bytes));
  CubDebugExit(cub::DeviceScan::InclusiveSum(d_tmp2, tmp2_bytes, d_O, d_O, NO));

  setGrid(MAX_WG_SIZE, NO, grid, block);
  make_ones<<<grid, block>>>(d_C, NO);

  thrust::device_ptr<uint> p_K(d_C);
  thrust::device_ptr<uint> p_L(d_O);
  thrust::exclusive_scan_by_key(p_L, p_L + NO, p_K, p_K);

  setGrid(MAX_WG_SIZE, NO, grid, block);
  make_cell_ids<<<grid, block>>>(d_C, d_O, M, d_boxes, d_Px, d_Py, d_Pz, N, NO);
  makePairs.Stop();

  CubDebugExit(g_allocator.DeviceAllocate((void**)&d_cs.d_buffers[1], sizeof(uint)*NO));
  CubDebugExit(g_allocator.DeviceAllocate((void**)&d_os.d_buffers[1], sizeof(uint)*NO));

  GpuTimer sort;
  cudaDeviceSynchronize();
  sort.Start();

  void *d_tmp3 = NULL;
  size_t tmp3_bytes = 0;

  CubDebugExit(DeviceRadixSort::SortPairs(d_tmp3, tmp3_bytes, d_cs, d_os, NO));
  CubDebugExit(g_allocator.DeviceAllocate(&d_tmp3, tmp3_bytes));
  CubDebugExit(DeviceRadixSort::SortPairs(d_tmp3, tmp3_bytes, d_cs, d_os, NO));

  d_C = d_cs.Current();
  d_O = d_os.Current();

  sort.Stop();

  uint *d_G = NULL;
  CubDebugExit(g_allocator.DeviceAllocate((void**)&d_G, sizeof(uint)*(N+1)));
  uint *d_counts = NULL;
  CubDebugExit(g_allocator.DeviceAllocate((void**)&d_counts, sizeof(uint)*N));
  int *d_num_segments = NULL;
  CubDebugExit(g_allocator.DeviceAllocate((void**)&d_num_segments, sizeof(int)));

  GpuTimer computeCells;
  cudaDeviceSynchronize();
  computeCells.Start();

  void *d_tmp4 = NULL;
  size_t tmp4_bytes = 0;

  CubDebugExit(cub::DeviceReduce::RunLengthEncode(d_tmp4, tmp4_bytes, d_C, d_C, d_counts, d_num_segments, NO));
  CubDebugExit(g_allocator.DeviceAllocate(&d_tmp4, tmp4_bytes));
  CubDebugExit(cub::DeviceReduce::RunLengthEncode(d_tmp4, tmp4_bytes, d_C, d_C, d_counts, d_num_segments, NO));

  int nonEmpty;
  CubDebugExit(cudaMemcpy(&nonEmpty, d_num_segments, sizeof(int), cudaMemcpyDeviceToHost));

  CubDebugExit(cudaMemset(d_G, 0, sizeof(uint)*(N+1)));

  setGrid(MAX_WG_SIZE, nonEmpty, grid, block);
  nonempty_cells<<<grid, block>>>(d_G, d_C, d_counts, nonEmpty);

  uint zero = 0u;
  CubDebugExit(cudaMemcpy(d_G + N, &zero, sizeof(uint), cudaMemcpyHostToDevice));

  computeCells.Stop();

  void *d_tmp5 = NULL;
  size_t tmp5_bytes = 0;

  CubDebugExit(cub::DeviceScan::ExclusiveSum(d_tmp5, tmp5_bytes, d_G, d_G, N+1));
  CubDebugExit(g_allocator.DeviceAllocate(&d_tmp5, tmp5_bytes));
  CubDebugExit(cub::DeviceScan::ExclusiveSum(d_tmp5, tmp5_bytes, d_G, d_G, N+1));

  cudaDeviceSynchronize();

  delta = bench.get_time() - delta;
  buildTime += delta;

  uint NO2;
  CubDebugExit(cudaMemcpy(&NO2, d_G + N, sizeof(uint), cudaMemcpyDeviceToHost));

  if (clInfoStats) {
    printf("NOt: %d NOc: %d\n", NO, NO2);
  }

  if (NO2 < NO) {
    /* some of the triangles ids shouldn't be here at all because they don't overlap some cells. fix it. */
    --nonEmpty;
    NO = NO2;
  }

  result(d_G, d_O);


  ////////////////////////////////////////////////////////////////////////////////
  if (d_tmp1) CubDebugExit(g_allocator.DeviceFree(d_tmp1));
  if (d_tmp2) CubDebugExit(g_allocator.DeviceFree(d_tmp2));
  if (d_tmp3) CubDebugExit(g_allocator.DeviceFree(d_tmp3));
  if (d_tmp4) CubDebugExit(g_allocator.DeviceFree(d_tmp4));
  if (d_tmp5) CubDebugExit(g_allocator.DeviceFree(d_tmp5));

  if (d_cs.d_buffers[0]) CubDebugExit(g_allocator.DeviceFree(d_cs.d_buffers[0]));
  if (d_cs.d_buffers[1]) CubDebugExit(g_allocator.DeviceFree(d_cs.d_buffers[1]));
  if (d_os.d_buffers[0]) CubDebugExit(g_allocator.DeviceFree(d_os.d_buffers[0]));
  if (d_os.d_buffers[1]) CubDebugExit(g_allocator.DeviceFree(d_os.d_buffers[1]));

  if (d_G) CubDebugExit(g_allocator.DeviceFree(d_G));

  const float avgItemsPerNonEmptyCell = float(NO) / nonEmpty;
  const float avgCellsPerItem = float(NO) / nobjs;

  if (clInfoStats) {
    float avedevItemsPerNonEmptyCell = 0.0f;
    float avedevCellsPerItem = 0.0f;

    float *d_squaresA = NULL;
    CubDebugExit(g_allocator.DeviceAllocate((void**)&d_squaresA, sizeof(float)*nonEmpty));

    setGrid(MAX_WG_SIZE, nonEmpty, grid, block);
    avedev_samples<<<grid, block>>>(d_squaresA, d_counts, avgItemsPerNonEmptyCell, nonEmpty);

    void *d_tmpA = NULL;
    size_t tmpA_bytes = 0;
    cub::DeviceReduce::Sum(d_tmpA, tmpA_bytes, d_squaresA, d_squaresA, nonEmpty);
    CubDebugExit(g_allocator.DeviceAllocate(&d_tmpA, tmpA_bytes));
    cub::DeviceReduce::Sum(d_tmpA, tmpA_bytes, d_squaresA, d_squaresA, nonEmpty);

    CubDebugExit(cudaMemcpy(&avedevItemsPerNonEmptyCell, d_squaresA, sizeof(float), cudaMemcpyDeviceToHost));
    avedevItemsPerNonEmptyCell = (1.0f/nonEmpty)*avedevItemsPerNonEmptyCell;

    if (d_tmpA) CubDebugExit(g_allocator.DeviceFree(d_tmpA));
    if (d_squaresA) CubDebugExit(g_allocator.DeviceFree(d_squaresA));

    setGrid(MAX_WG_SIZE, nobjs, grid, block);
    intersected_cells<<<grid, block>>>(d_V, d_boxes, d_Px, d_Py, d_Pz, nobjs);

    float *d_squaresB = NULL;
    CubDebugExit(g_allocator.DeviceAllocate((void**)&d_squaresB, sizeof(float)*nobjs));

    setGrid(MAX_WG_SIZE, nobjs, grid, block);
    avedev_samples<<<grid, block>>>(d_squaresB, d_V, avgCellsPerItem, nobjs);

    void *d_tmpB = NULL;
    size_t tmpB_bytes = 0;
    cub::DeviceReduce::Sum(d_tmpB, tmpB_bytes, d_squaresB, d_squaresB, nobjs);
    CubDebugExit(g_allocator.DeviceAllocate(&d_tmpB, tmpB_bytes));
    cub::DeviceReduce::Sum(d_tmpB, tmpB_bytes, d_squaresB, d_squaresB, nobjs);

    CubDebugExit(cudaMemcpy(&avedevCellsPerItem, d_squaresB, sizeof(float), cudaMemcpyDeviceToHost));
    avedevCellsPerItem = (1.0f/nobjs)*avedevCellsPerItem;

    if (d_tmpB) CubDebugExit(g_allocator.DeviceFree(d_tmpB));
    if (d_squaresB) CubDebugExit(g_allocator.DeviceFree(d_squaresB));

    int *d_out = NULL;
    CubDebugExit(g_allocator.DeviceAllocate((void**)&d_out, sizeof(int)));

    void *d_tmpC = NULL;
    size_t tmpC_bytes = 0;
    cub::DeviceReduce::Max(d_tmpC, tmpC_bytes, d_V, d_out, nobjs);
    CubDebugExit(g_allocator.DeviceAllocate(&d_tmpC, tmpC_bytes));
    cub::DeviceReduce::Max(d_tmpC, tmpC_bytes, d_V, d_out, nobjs);

    int maxCellsPerItem = 0;
    CubDebugExit(cudaMemcpy(&maxCellsPerItem, d_out, sizeof(int), cudaMemcpyDeviceToHost));

    if (d_tmpC) CubDebugExit(g_allocator.DeviceFree(d_tmpC));
    if (d_out) CubDebugExit(g_allocator.DeviceFree(d_out));

    ////////////////////////////////////////////////////////////////////////////////
    const float emptyCellsPct = 100.0f * float(N - nonEmpty) / N;

    size_t memGrid = 0;
    memGrid += sizeof(uint) * (N+1);	// C
    memGrid += sizeof(uint) * NO;	// L

    printf("---\n");

    printf("grid res: %dx%dx%d\n# cells: %u\n%% empty cells: %.2f%%\navg # items / n-empt cell: %.2f\navedev # items / n-empt cell: %.2f\nmax # cells / item: %d\navg # cells / item: %.2f\navedev # cells / item: %.2f\nNO: %d\n",
      M.x, M.y ,M.z,
      N,
      emptyCellsPct,
      avgItemsPerNonEmptyCell,
      avedevItemsPerNonEmptyCell,
      maxCellsPerItem,
      avgCellsPerItem,
      avedevCellsPerItem,
      NO);

    printf("mem grid: %lu\n", memGrid);
  }

  if (clBenchmark) {
    printf("---\n");
    std::cout << "count cells time: " << std::fixed << std::setprecision(2) << countCells.ElapsedMillis() << " ms" << std::endl;
    std::cout << "make pairs time: " << std::fixed << std::setprecision(2) << makePairs.ElapsedMillis() << " ms" << std::endl;
    std::cout << "sort time: " << std::fixed << std::setprecision(2) << sort.ElapsedMillis() << " ms" << std::endl;
    std::cout << "compute cells time: " << std::fixed << std::setprecision(2) << computeCells.ElapsedMillis() << " ms" << std::endl;
    std::cout << "total grid build time: " << std::fixed << std::setprecision(2) << delta*1000.0 << " ms" << std::endl;
    printf("---\n");
  }

  ////////////////////////////////////////////////////////////////////////////////
  if (d_V) CubDebugExit(g_allocator.DeviceFree(d_V));

  if (d_counts) CubDebugExit(g_allocator.DeviceFree(d_counts));

  finish();
}

//--------------------------------------------------------------------------------
__global__
static void make_pairs(uint *C, uint *O, int3 M, const uint *V, const T_AABB *boxes, int length)
{
  int i = blockDim.x * blockIdx.x + threadIdx.x;

  if (i >= length)
    return;

  const T_AABB& box = boxes[i];

  uint place = V[i];

  for (short x=box.lox; x<=box.hix; ++x) {
    for (short y=box.loy; y<=box.hiy; ++y) {
      for (short z=box.loz; z<=box.hiz; ++z) {
        const int3 id = (int3){x, y, z};
        C[place] = coords(M, id);
        O[place] = i;
        place++;
      }
    }
  }
}

__global__
static void extract_cells(uint *Left, uint *Right, const uint *C, const int N, const int NO)
{
  int i = blockDim.x * blockIdx.x + threadIdx.x;

  if (i > NO)
    return;

  if (i == 0) {
    Left[C[i]] = i;
  } else if (i == NO) {
    Right[C[i-1]] = i;
  } else if (C[i] != C[i-1]) {
    Left[C[i]] = i;
    Right[C[i-1]] = i;
  }
}

__global__
static void size_cells(uint *G, uint *Left, uint *Right, const int length)
{
  int i = blockDim.x * blockIdx.x + threadIdx.x;

  if (i >= length)
    return;

  G[i] = Right[i]-Left[i];
}

//--------------------------------------------------------------------------------
extern "C" void buildKS(unsigned nobjs, T_AABB *boxes, float *planes, float *Px, float *Py, float *Pz, int Mx, int My, int Mz) {
  begin(nobjs, boxes, planes, Px, Py, Pz, Mx, My, Mz);

  ////////////////////////////////////////////////////////////////////////////////
  double delta = bench.get_time();

  N = Mx*My*Mz;

  uint *d_V = NULL;
  CubDebugExit(g_allocator.DeviceAllocate((void**)&d_V, sizeof(uint)*(nobjs+1)));

  uint data = 0;
  CubDebugExit(cudaMemcpy(d_V + nobjs, &data, sizeof(uint), cudaMemcpyHostToDevice));

  ////////////////////////////////////////
  GpuTimer countCells;

  cudaDeviceSynchronize();

  countCells.Start();

  int grid, block;
  setGrid(MAX_WG_SIZE, nobjs, grid, block);
  count_cells<<<grid, block>>>(d_V, d_boxes, nobjs);

  void *d_tmp1 = NULL;
  size_t tmp1_bytes = 0;

  CubDebugExit(cub::DeviceScan::ExclusiveSum(d_tmp1, tmp1_bytes, d_V, d_V, nobjs+1));
  CubDebugExit(g_allocator.DeviceAllocate(&d_tmp1, tmp1_bytes));
  CubDebugExit(cub::DeviceScan::ExclusiveSum(d_tmp1, tmp1_bytes, d_V, d_V, nobjs+1));
  cudaDeviceSynchronize();

  countCells.Stop();

  CubDebugExit(cudaMemcpy(&NO, d_V + nobjs, sizeof(uint), cudaMemcpyDeviceToHost));

  DoubleBuffer<uint> d_cs;
  DoubleBuffer<uint> d_os;
  CubDebugExit(g_allocator.DeviceAllocate((void**)&d_cs.d_buffers[0], sizeof(uint)*NO));
  CubDebugExit(g_allocator.DeviceAllocate((void**)&d_os.d_buffers[0], sizeof(uint)*NO));

  uint *d_C, *d_O;
  d_C = d_cs.Current();
  d_O = d_os.Current();

  GpuTimer makePairs;
  cudaDeviceSynchronize();
  makePairs.Start();

  setGrid(MAX_WG_SIZE, nobjs, grid, block);
  make_pairs<<<grid, block>>>(d_C, d_O, M, d_V, d_boxes, nobjs);
  makePairs.Stop();

  CubDebugExit(g_allocator.DeviceAllocate((void**)&d_cs.d_buffers[1], sizeof(uint)*NO));
  CubDebugExit(g_allocator.DeviceAllocate((void**)&d_os.d_buffers[1], sizeof(uint)*NO));

  GpuTimer sort;
  cudaDeviceSynchronize();
  sort.Start();

  void *d_tmp3 = NULL;
  size_t tmp3_bytes = 0;

  CubDebugExit(DeviceRadixSort::SortPairs(d_tmp3, tmp3_bytes, d_cs, d_os, NO));
  CubDebugExit(g_allocator.DeviceAllocate(&d_tmp3, tmp3_bytes));
  CubDebugExit(DeviceRadixSort::SortPairs(d_tmp3, tmp3_bytes, d_cs, d_os, NO));

  d_C = d_cs.Current();
  d_O = d_os.Current();

  sort.Stop();

  uint *d_G = NULL;
  CubDebugExit(g_allocator.DeviceAllocate((void**)&d_G, sizeof(uint)*(N+1)));
  uint *d_Left = NULL;
  CubDebugExit(g_allocator.DeviceAllocate((void**)&d_Left, sizeof(uint)*N));
  uint *d_Right = NULL;
  CubDebugExit(g_allocator.DeviceAllocate((void**)&d_Right, sizeof(uint)*N));

  GpuTimer computeCells;
  cudaDeviceSynchronize();
  computeCells.Start();

  CubDebugExit(cudaMemset(d_Left, 0, sizeof(uint)*N));
  CubDebugExit(cudaMemset(d_Right, 0, sizeof(uint)*N));

  // extract cell ranges.
  setGrid(MAX_WG_SIZE, NO, grid, block);
  extract_cells<<<grid, block>>>(d_Left, d_Right, d_C, N, NO);

  // compact grid.
  setGrid(MAX_WG_SIZE, N, grid, block);
  size_cells<<<grid, block>>>(d_G, d_Left, d_Right, N);

  uint zero = 0u;
  CubDebugExit(cudaMemcpy(d_G + N, &zero, sizeof(uint), cudaMemcpyHostToDevice));

  void *d_tmp5 = NULL;
  size_t tmp5_bytes = 0;

  CubDebugExit(cub::DeviceScan::ExclusiveSum(d_tmp5, tmp5_bytes, d_G, d_G, N+1));
  CubDebugExit(g_allocator.DeviceAllocate(&d_tmp5, tmp5_bytes));
  CubDebugExit(cub::DeviceScan::ExclusiveSum(d_tmp5, tmp5_bytes, d_G, d_G, N+1));

  computeCells.Stop();

  cudaDeviceSynchronize();

  delta = bench.get_time() - delta;
  buildTime += delta;

  result(d_G, d_O);


  ////////////////////////////////////////////////////////////////////////////////
  if (d_C) CubDebugExit(g_allocator.DeviceFree(d_C));
  if (d_O) CubDebugExit(g_allocator.DeviceFree(d_O));

  if (d_G) CubDebugExit(g_allocator.DeviceFree(d_G));
  if (d_Left) CubDebugExit(g_allocator.DeviceFree(d_Left));
  if (d_Right) CubDebugExit(g_allocator.DeviceFree(d_Right));

  if (clInfoStats) {
    size_t memGrid = 0;
    memGrid += sizeof(uint) * (N+1);	// C
    memGrid += sizeof(uint) * NO;	// L

    printf("---\n");

    printf("grid res: %dx%dx%d\n# cells: %u\nNO: %d\n",
      M.x, M.y ,M.z,
      N,
      NO);

    printf("mem grid: %lu\n", memGrid);
  }

  if (clBenchmark) {
    printf("---\n");
    std::cout << "count cells time: " << std::fixed << std::setprecision(2) << countCells.ElapsedMillis() << " ms" << std::endl;
    std::cout << "make pairs time: " << std::fixed << std::setprecision(2) << makePairs.ElapsedMillis() << " ms" << std::endl;
    std::cout << "sort time: " << std::fixed << std::setprecision(2) << sort.ElapsedMillis() << " ms" << std::endl;
    std::cout << "compute cells time: " << std::fixed << std::setprecision(2) << computeCells.ElapsedMillis() << " ms" << std::endl;
    std::cout << "total grid build time: " << std::fixed << std::setprecision(2) << delta*1000.0 << " ms" << std::endl;
    printf("---\n");
  }

  ////////////////////////////////////////////////////////////////////////////////
  if (d_V) CubDebugExit(g_allocator.DeviceFree(d_V));

  finish();
}

//--------------------------------------------------------------------------------
__global__
static void build_cells(uint *G, const int3 M, const T_AABB *boxes, const float *Px, const float *Py, const float *Pz, int length)
{
  int j = blockDim.x * blockIdx.x + threadIdx.x;

  if (j >= length)
    return;

  const T_AABB& box = boxes[j];
  float4 P = tex1Dfetch(t_planes, j);

  for (short x=box.lox; x<=box.hix; ++x) {
    for (short y=box.loy; y<=box.hiy; ++y) {
      for (short z=box.loz; z<=box.hiz; ++z) {
      	float3 lo = (float3){Px[x], Py[y], Pz[z]};
        float3 hi = (float3){Px[x+1], Py[y+1], Pz[z+1]};
        if (collides(P, lo, hi)) {
          const int3 id = (int3){x, y, z};
          const uint i = coords(M, id);
          atomicAdd(&G[i], 1);
        }
      }
    }
  }
}

__global__
static void build_item_lists(uint *G, uint *O, const int3 M, const T_AABB *boxes, const float *Px, const float *Py, const float *Pz, int length)
{
  int j = blockDim.x * blockIdx.x + threadIdx.x;

  if (j >= length)
    return;

  const T_AABB& box = boxes[j];
  float4 P = tex1Dfetch(t_planes, j);

  for (short x=box.lox; x<=box.hix; ++x) {
    for (short y=box.loy; y<=box.hiy; ++y) {
      for (short z=box.loz; z<=box.hiz; ++z) {
      	float3 lo = (float3){Px[x], Py[y], Pz[z]};
	      float3 hi = (float3){Px[x+1], Py[y+1], Pz[z+1]};
	      if (collides(P, lo, hi)) {
	        const int3 id = (int3){x, y, z};
	        const uint i = coords(M, id);

	        uint place = atomicSub(&G[i], 1);
	        O[place-1] = j;
        }
      }
    }
  }
}

//--------------------------------------------------------------------------------
extern "C" void buildLD(unsigned nobjs, T_AABB *boxes, float *planes, float *Px, float *Py, float *Pz, int Mx, int My, int Mz) {
  begin(nobjs, boxes, planes, Px, Py, Pz, Mx, My, Mz);

  ////////////////////////////////////////////////////////////////////////////////
  double delta = bench.get_time();

  N = Mx*My*Mz;

  uint *d_G = NULL;
  CubDebugExit(g_allocator.DeviceAllocate((void**)&d_G, sizeof(uint)*(N+1)));

  CubDebugExit(cudaMemset(d_G, 0, sizeof(uint)*(N+1)));

  ////////////////////////////////////////
  GpuTimer buildCells;

  cudaDeviceSynchronize();
  buildCells.Start();

  int grid, block;
  setGrid(MAX_WG_SIZE, nobjs, grid, block);
  build_cells<<<grid, block>>>(d_G, M, d_boxes, d_Px, d_Py, d_Pz, nobjs);

  buildCells.Stop();

  ////////////////////////////////////////
  GpuTimer scanCells;

  cudaDeviceSynchronize();
  scanCells.Start();

  void *d_tmp = NULL;
  size_t tmp_bytes = 0;

  CubDebugExit(cub::DeviceScan::InclusiveSum(d_tmp, tmp_bytes, d_G, d_G, N+1));
  CubDebugExit(g_allocator.DeviceAllocate(&d_tmp, tmp_bytes));
  CubDebugExit(cub::DeviceScan::InclusiveSum(d_tmp, tmp_bytes, d_G, d_G, N+1));

  scanCells.Stop();

  CubDebugExit(cudaMemcpy(&NO, d_G + N, sizeof(uint), cudaMemcpyDeviceToHost));

  uint *d_O = NULL;
  CubDebugExit(g_allocator.DeviceAllocate((void**)&d_O, sizeof(uint)*NO));

  ////////////////////////////////////////
  GpuTimer buildItemLists;

  cudaDeviceSynchronize();
  buildItemLists.Start();

  setGrid(MAX_WG_SIZE, nobjs, grid, block);
  build_item_lists<<<grid, block>>>(d_G, d_O, M, d_boxes, d_Px, d_Py, d_Pz, nobjs);

  buildItemLists.Stop();

  cudaDeviceSynchronize();

  delta = bench.get_time() - delta;
  buildTime += delta;

  result(d_G, d_O);


  ////////////////////////////////////////////////////////////////////////////////
  if (clInfoStats) {
    size_t memGrid = 0;
    memGrid += sizeof(uint) * (N+1);	// C
    memGrid += sizeof(uint) * NO;	// L

    printf("---\n");

    printf("grid res: %dx%dx%d\n# cells: %u\nNO: %d\n",
      M.x, M.y ,M.z,
      N,
      NO);

    printf("mem grid: %lu\n", memGrid);
  }

  if (clBenchmark) {
    printf("---\n");
    std::cout << "intersect objects time: " << std::fixed << std::setprecision(2) << buildCells.ElapsedMillis() << " ms" << std::endl;
    std::cout << "compute prefix sum time: " << std::fixed << std::setprecision(2) << scanCells.ElapsedMillis() << " ms" << std::endl;
    std::cout << "insert indices time: " << std::fixed << std::setprecision(2) << buildItemLists.ElapsedMillis() << " ms" << std::endl;
    std::cout << "total grid build time: " << std::fixed << std::setprecision(2) << delta*1000.0 << " ms" << std::endl;
    printf("---\n");
  }

  ////////////////////////////////////////////////////////////////////////////////
  finish();
}
