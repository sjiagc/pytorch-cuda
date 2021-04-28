#include <torch/torch.h>
#include <torch/script.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <iostream>

// It's from Fashion MNIST test data indexed 100
static const float TEST_DATA[1][28][28] =
{ {{0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
0.0000, 0.1373, 0.2980, 0.2824, 0.0000, 0.0000, 0.0000, 0.0000,
0.3176, 0.2980, 0.0078, 0.0706, 0.0000, 0.0000, 0.0000, 0.0000,
0.0000, 0.0000, 0.0000, 0.0000},
{0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
0.3843, 0.8118, 0.9412, 0.7137, 0.3765, 0.5098, 0.5412, 0.4235,
0.5882, 0.7490, 0.7569, 0.6745, 0.3059, 0.0000, 0.0000, 0.0000,
0.0000, 0.0000, 0.0000, 0.0000},
{0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.2314,
0.6118, 0.5882, 0.8745, 0.7608, 0.8078, 0.5294, 0.5098, 0.2588,
0.0392, 0.3529, 0.6000, 0.7020, 0.8941, 0.1804, 0.0000, 0.0000,
0.0000, 0.0000, 0.0000, 0.0000},
{0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.4471,
0.6314, 0.6118, 0.8314, 0.6980, 0.7843, 0.7255, 0.2941, 0.5098,
0.7529, 0.2471, 0.4549, 0.4314, 0.6392, 0.2275, 0.0000, 0.0000,
0.0000, 0.0000, 0.0000, 0.0000},
{0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
0.4824, 1.0000, 0.8000, 0.6157, 0.3725, 0.5529, 0.1725, 0.2510,
0.3529, 0.1922, 0.4784, 0.0588, 0.0000, 0.0000, 0.0000, 0.0000,
0.0000, 0.0000, 0.0000, 0.0000},
{0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
0.0000, 0.0824, 0.8588, 0.7373, 0.6157, 0.6353, 0.2549, 0.4667,
0.2471, 0.4706, 0.2431, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
0.0000, 0.0000, 0.0000, 0.0000},
{0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
0.0000, 0.0000, 0.6314, 0.6549, 0.8078, 0.7961, 0.3020, 0.2471,
0.2275, 0.4314, 0.2078, 0.0000, 0.0039, 0.0000, 0.0000, 0.0000,
0.0000, 0.0000, 0.0000, 0.0000},
{0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
0.0000, 0.0000, 0.4314, 0.3176, 0.7098, 0.7569, 0.6353, 0.2784,
0.1412, 0.2745, 0.2196, 0.0000, 0.0039, 0.0000, 0.0000, 0.0000,
0.0000, 0.0000, 0.0000, 0.0000},
{0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0039,
0.0000, 0.0000, 0.4078, 0.3373, 0.8784, 0.8196, 0.7255, 0.4118,
0.1294, 0.5569, 0.1451, 0.0000, 0.0078, 0.0000, 0.0000, 0.0000,
0.0000, 0.0000, 0.0000, 0.0000},
{0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0039,
0.0000, 0.0000, 0.4275, 0.3098, 0.8549, 0.0706, 0.3725, 0.7098,
0.1333, 0.7255, 0.2196, 0.0000, 0.0039, 0.0000, 0.0000, 0.0000,
0.0000, 0.0000, 0.0000, 0.0000},
{0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
0.0000, 0.0314, 0.4471, 0.4196, 0.3843, 0.2510, 0.2431, 0.1725,
0.5686, 0.8000, 0.3529, 0.0000, 0.0078, 0.0000, 0.0000, 0.0000,
0.0000, 0.0000, 0.0000, 0.0000},
{0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
0.0000, 0.0392, 0.4275, 0.4510, 0.4314, 0.4510, 0.1333, 0.4314,
0.8039, 0.8039, 0.3373, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
0.0000, 0.0000, 0.0000, 0.0000},
{0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
0.0000, 0.2745, 0.7255, 0.5686, 0.5451, 0.5647, 0.1843, 0.9412,
0.7843, 0.7176, 0.5569, 0.0078, 0.0000, 0.0078, 0.0000, 0.0000,
0.0000, 0.0000, 0.0000, 0.0000},
{0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
0.0000, 0.4039, 0.6157, 0.4275, 0.2118, 0.5843, 1.0000, 0.7608,
0.2157, 0.5882, 0.7804, 0.1647, 0.0000, 0.0196, 0.0000, 0.0000,
0.0000, 0.0000, 0.0000, 0.0000},
{0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
0.0000, 0.2980, 0.2863, 0.5922, 0.7686, 0.9294, 0.8706, 0.2353,
0.4667, 0.4235, 0.4471, 0.1137, 0.0000, 0.0118, 0.0000, 0.0000,
0.0000, 0.0000, 0.0000, 0.0000},
{0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
0.0000, 0.3216, 0.2510, 0.4706, 0.6118, 0.4863, 0.9843, 0.5333,
0.1412, 0.3216, 0.6706, 0.0431, 0.0000, 0.0039, 0.0000, 0.0000,
0.0000, 0.0000, 0.0000, 0.0000},
{0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
0.0000, 0.0549, 0.2980, 0.1490, 0.0902, 0.1529, 0.8824, 0.8039,
0.5765, 0.6667, 0.9765, 0.0078, 0.0000, 0.0078, 0.0000, 0.0000,
0.0000, 0.0000, 0.0000, 0.0000},
{0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0039,
0.0000, 0.3255, 0.6196, 0.1176, 0.1608, 0.2627, 0.9333, 0.8706,
0.8392, 0.8588, 0.7059, 0.0000, 0.0000, 0.0039, 0.0000, 0.0000,
0.0000, 0.0000, 0.0000, 0.0000},
{0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0118,
0.0000, 0.4627, 0.8275, 0.2745, 0.6902, 0.2471, 0.7490, 0.0314,
0.5529, 0.8431, 0.4941, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
0.0000, 0.0000, 0.0000, 0.0000},
{0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0118,
0.0000, 0.2235, 0.9176, 0.6784, 0.5686, 0.4824, 0.3882, 0.5020,
0.3569, 0.8275, 0.4784, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
0.0000, 0.0000, 0.0000, 0.0000},
{0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0078,
0.0000, 0.1059, 0.8471, 0.6510, 0.3255, 0.2392, 0.6588, 0.5725,
0.4706, 0.5804, 0.4549, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
0.0000, 0.0000, 0.0000, 0.0000},
{0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0078,
0.0000, 0.0118, 0.8627, 0.7020, 0.3569, 0.3569, 0.6471, 0.6588,
0.6314, 0.6235, 0.3020, 0.0000, 0.0078, 0.0000, 0.0000, 0.0000,
0.0000, 0.0000, 0.0000, 0.0000},
{0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
0.0000, 0.0000, 0.6784, 0.5882, 0.8039, 0.8863, 0.6627, 0.7843,
0.7412, 0.6941, 0.1882, 0.0000, 0.0078, 0.0000, 0.0000, 0.0000,
0.0000, 0.0000, 0.0000, 0.0000},
{0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
0.0000, 0.0000, 0.6706, 0.7882, 0.9294, 0.7176, 0.5333, 0.7725,
0.7373, 0.4353, 0.1373, 0.0000, 0.0078, 0.0000, 0.0000, 0.0000,
0.0000, 0.0000, 0.0000, 0.0000},
{0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0039,
0.0000, 0.0000, 0.6000, 0.8196, 0.9020, 0.7843, 0.7529, 0.4902,
0.3686, 0.1216, 0.0667, 0.0000, 0.0078, 0.0000, 0.0000, 0.0000,
0.0000, 0.0000, 0.0000, 0.0000},
{0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
0.0000, 0.0000, 0.7137, 0.9216, 0.8667, 0.7843, 0.8549, 0.5765,
0.4627, 0.7098, 0.0000, 0.0000, 0.0039, 0.0000, 0.0000, 0.0000,
0.0000, 0.0000, 0.0000, 0.0000},
{0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
0.0000, 0.0000, 0.5961, 0.8392, 0.9373, 0.9020, 0.8549, 0.7647,
0.5843, 0.6980, 0.0118, 0.0000, 0.0039, 0.0000, 0.0000, 0.0000,
0.0000, 0.0000, 0.0000, 0.0000},
{0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0039,
0.0000, 0.0000, 0.0000, 0.1294, 0.5255, 0.5765, 0.4824, 0.3765,
0.3686, 0.1255, 0.0000, 0.0039, 0.0000, 0.0000, 0.0000, 0.0000,
0.0000, 0.0000, 0.0000, 0.0000 }} };

static const int BATCH_SIZE = 128;
static const size_t TEST_DATA_SIZE = sizeof(TEST_DATA);
static const size_t TEST_DATA_ELEMENT_COUNT = TEST_DATA_SIZE / sizeof(float);

#define CUDA_DRVAPI_CALL(cudaAPI)                                                                               \
    do {                                                                                                        \
        CUresult errorCode = cudaAPI;                                                                           \
        if (errorCode != CUDA_SUCCESS) {                                                                        \
            const char *errName = nullptr;                                                                      \
            cuGetErrorName(errorCode, &errName);                                                                \
            std::cerr << "General error " << #cudaAPI << " returned error " << errName                          \
                << " in " << __FUNCTION__ << "(" << __FILE__ << ":" << __LINE__ << ")"                          \
                << std::endl;                                                                                   \
            throw std::exception();                                                                             \
        }                                                                                                       \
    } while (0)

double ms_now() {
	double ret;
	auto timePoint = std::chrono::high_resolution_clock::now().time_since_epoch();
	ret = std::chrono::duration<double, std::milli>(timePoint).count();
	return ret;
}

//#define DRIVER_API

#ifdef DRIVER_API
void createCudaContext(CUcontext* outContext, int inGpu, CUctx_flags inFlags) {
	CUdevice cuDevice = 0;
	CUDA_DRVAPI_CALL(cuDeviceGet(&cuDevice, inGpu));
	char szDeviceName[80];
	CUDA_DRVAPI_CALL(cuDeviceGetName(szDeviceName, sizeof(szDeviceName), cuDevice));
	std::cout << "GPU in use: " << szDeviceName << std::endl;
	CUDA_DRVAPI_CALL(cuCtxCreate(outContext, inFlags, cuDevice));
}
#endif

int main() {
	static const int TEST_ROUND = 1000;

	std::cout << "BATCH_SIZE = " << BATCH_SIZE << ", TEST_ROUND = " << TEST_ROUND << std::endl;

	int deviceCount = 0;
	cudaGetDeviceCount(&deviceCount);
	std::cout << "Device count " << deviceCount << std::endl;

	std::cout << "Preparing data" << std::endl;

#ifdef DRIVER_API
	CUDA_DRVAPI_CALL(cuInit(0));
	CUcontext roContext = nullptr;
	CUdevice cuDevice = 0;
	CUDA_DRVAPI_CALL(cuDeviceGet(&cuDevice, 0));
	char szDeviceName[80];
	CUDA_DRVAPI_CALL(cuDeviceGetName(szDeviceName, sizeof(szDeviceName), cuDevice));
	std::cout << "GPU in use: " << szDeviceName << std::endl;
	CUDA_DRVAPI_CALL(cuDevicePrimaryCtxRetain(&roContext, cuDevice));
	std::cout << "Context primary " << roContext << std::endl;
	CUDA_DRVAPI_CALL(cuCtxPushCurrent(roContext));

	CUdeviceptr testData_d;
	CUDA_DRVAPI_CALL(cuMemAlloc(&testData_d, TEST_DATA_SIZE * BATCH_SIZE));
	for (int i = 0; i < BATCH_SIZE; ++i)
		CUDA_DRVAPI_CALL(cuMemcpyHtoD(testData_d + TEST_DATA_ELEMENT_COUNT * i, TEST_DATA, TEST_DATA_SIZE));
#else
	float *testData_d;
	cudaMalloc(&testData_d, TEST_DATA_SIZE * BATCH_SIZE);
	for (int i = 0; i < BATCH_SIZE; ++i)
		cudaMemcpy(testData_d + TEST_DATA_ELEMENT_COUNT * i, TEST_DATA, TEST_DATA_SIZE, cudaMemcpyHostToDevice);
#endif
	float* testData = new float[TEST_DATA_ELEMENT_COUNT * BATCH_SIZE];
	for (int i = 0; i < BATCH_SIZE; ++i)
		memcpy_s(testData + TEST_DATA_ELEMENT_COUNT * i, TEST_DATA_SIZE, TEST_DATA, TEST_DATA_SIZE);

	torch::Device cpu(torch::kCPU);
	torch::Device device(torch::kCPU);
	if (torch::cuda::is_available()) {
		std::cout << "CUDA is available! Inferencing on GPU." << std::endl;
		device = torch::Device(torch::kCUDA);
	}

	static const char const *MODEL_FILE_PATH = "model.pt";

	std::ifstream is(MODEL_FILE_PATH, std::ios::binary);
	if (!is) {
		std::cerr << "Cannot open model file: " << MODEL_FILE_PATH << std::endl;
		return -1;
	}

	torch::jit::script::Module module;
	try {
		// Deserialize the ScriptModule from a file using torch::jit::load().
		module = torch::jit::load(is, device);
	}
	catch (const c10::Error& e) {
		std::cerr << "error loading the model, " << e.what() << "\n";
		return -1;
	}
	torch::Tensor t_d = torch::from_blob((void*)testData_d, { BATCH_SIZE, 1, 28, 28 }, device);
	torch::Tensor t = torch::from_blob((void*)testData, { BATCH_SIZE, 1, 28, 28 });
	// Warm up
	std::cout << "Warming up" << std::endl;
	for (int i = 0; i < 2; ++i) {
		std::vector<torch::jit::IValue> inputs_d{ t_d };
		auto outTensor = module.forward(inputs_d);
	}
	for (int i = 0; i < 2; ++i) {
		std::vector<torch::jit::IValue> inputs{ t.to(device) };
		auto outTensor = module.forward(inputs);
	}
	double ms = 0;
	// GPU
	std::cout << "Running with GPU input" << std::endl;
	ms = ms_now();
	for (int i = 0; i < TEST_ROUND; ++i) {
		std::vector<torch::jit::IValue> inputs_d{ t_d };
		auto outTensor = module.forward(inputs_d);
	}
	ms = ms_now() - ms;
	std::cout << "Time GPU: " << ms << std::endl;
	// CPU
	std::cout << "Running with CPU input" << std::endl;
	ms = ms_now();
	for (int i = 0; i < TEST_ROUND; ++i) {
		std::vector<torch::jit::IValue> inputs{ t.to(device) };
		auto outTensor = module.forward(inputs);
	}
	ms = ms_now() - ms;
	std::cout << "Time CPU: " << ms << std::endl;
	// CPU
	std::cout << "Running with CPU input, round 2" << std::endl;
	ms = ms_now();
	for (int i = 0; i < TEST_ROUND; ++i) {
		std::vector<torch::jit::IValue> inputs{ t.to(device) };
		auto outTensor = module.forward(inputs);
	}
	ms = ms_now() - ms;
	std::cout << "Time CPU: " << ms << std::endl;
	// GPU
	std::cout << "Running with GPU input, round 2" << std::endl;
	ms = ms_now();
	for (int i = 0; i < TEST_ROUND; ++i) {
		std::vector<torch::jit::IValue> inputs_d{ t_d };
		auto outTensor = module.forward(inputs_d);
	}
	ms = ms_now() - ms;
	std::cout << "Time GPU: " << ms << std::endl;

#ifdef DRIVER_API
	CUDA_DRVAPI_CALL(cuMemFree(testData_d));
#else
	cudaFree(testData_d);
#endif
	delete[] testData;
}

