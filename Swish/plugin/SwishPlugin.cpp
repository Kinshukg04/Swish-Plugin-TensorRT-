#include <algorithm>
#include <cassert>
#include <iostream>
#include <numeric>
#include <vector>
#include <cmath>

#include <cublas_v2.h>

#include "NvInferPlugin.h"

// Macro for calling GPU functions
#define CHECK(status)                             \
    do                                            \
    {                                             \
        auto ret = (status);                      \
        if (ret != 0)                             \
        {                                         \
            std::cout << "Cuda failure: " << ret; \
            abort();                              \
        }                                         \
    } while (0)

using namespace nvinfer1;

namespace
{
	const char* SWISH_PLUGIN_VERSION{ "1" };
	const char* SWISH_PLUGIN_NAME{ "Swish_TRT" };
}

class SwishPlugin : public IPluginV2
{
public:
	// Ordinary ctor, plugin not yet configured for particular inputs/output
	SwishPlugin() {}

	// Ctor for clone()
	SwishPlugin(int totalElements)
	{
		mTotalElements = totalElements;
	}

	// Ctor for loading from serialized byte array
	SwishPlugin(const void* data, size_t length)
	{
		const char* d = reinterpret_cast<const char*>(data);
		const char* a = d;

		mTotalElements = read<int>(d);

		assert(d == a + length);
	}

	int getNbOutputs() const override
	{
		return 1;
	}

	Dims getOutputDimensions(int index, const Dims* inputs, int nbInputDims) override
	{
		assert(nbInputDims >= 1);
		assert(index == 0);

		//Output dimensions are same as input dims
		//Using dimensions of any element
		return DimsCHW(inputs[0].d[0], inputs[0].d[1], inputs[0].d[2]);
	}

	int initialize() override
	{
		CHECK(cublasCreate(&mCublas));
		return 0;
	}

	void terminate() override
	{
		CHECK(cublasDestroy(mCublas));
	}

	size_t getWorkspaceSize(int maxBatchSize) const override
	{
		return 0;
	}

	int enqueue(int batchSize, const void* const* inputs, void** outputs, void*, cudaStream_t stream) override
	{
		size_t inputOffset = 0;
		float* output = reinterpret_cast<float*>(outputs[0]);

		//Activation layer applied to one input only thats why we index 0
		const float* input = reinterpret_cast<const float*>(inputs[0]);
		cublasSetStream(mCublas, stream);

		for (size_t i = 0; i < mTotalElements; ++i) {
			float x = *(input+i);
			float exp_val = exp((double)-x);
			float output_val = 1 / (1 + exp_val);
			*output = output_val;
			output = output + 1;
		}

		return 0;
	}
	size_t getSerializationSize() const override
	{
		size_t size = sizeof(mTotalElements);
		return size;
	}

	void serialize(void* buffer) const override
	{
		char* d = reinterpret_cast<char*>(buffer);
		char* a = d;

		size_t totalElements = mTotalElements;

		write(d, totalElements);

		assert(d == a + getSerializationSize());
	}

	void configureWithFormat(const Dims* inputs, int nbInputs, const Dims* outputDims, int nbOutputs, nvinfer1::DataType type, nvinfer1::PluginFormat format, int maxBatchSize) override
	{
		assert(nbOutputs == 1);

		mTotalElements = 0;

		for (int i = 0; i < nbInputs; ++i)
		{
			//Number of elements to change
			mTotalElements += inputs[i].d[0] * inputs[i].d[1] * inputs[i].d[2];
		}
	}

	bool supportsFormat(DataType type, PluginFormat format) const override
	{
		return (type == DataType::kFLOAT && format == PluginFormat::kNCHW);
	}

	const char* getPluginType() const override { return SWISH_PLUGIN_NAME; }

	const char* getPluginVersion() const override { return SWISH_PLUGIN_VERSION; }

	void destroy() override {}

	IPluginV2* clone() const override
	{
		return new SwishPlugin(mTotalElements);
	}

	void setPluginNamespace(const char* pluginNamespace) override
	{
		mPluginNamespace = pluginNamespace;
	}

	const char* getPluginNamespace() const override
	{
		return mPluginNamespace.c_str();
	}


private:
	template <typename T>
	void write(char*& buffer, const T & val) const
	{
		*reinterpret_cast<T*>(buffer) = val;
		buffer += sizeof(T);
	}

	template <typename T>
	T read(const char*& buffer)
	{
		T val = *reinterpret_cast<const T*>(buffer);
		buffer += sizeof(T);
		return val;
	}

	int mTotalElements;
	cublasHandle_t mCublas;
	std::string mPluginNamespace = "";
};

// PluginCreator boilerplate code for FlattenConcat plugin
class SwishPluginCreator : public IPluginCreator
{
public:
	SwishPluginCreator()
	{
		mFC.nbFields = 0;
		mFC.fields = 0;
	}

	~SwishPluginCreator() {}

	const char* getPluginName() const override { return SWISH_PLUGIN_NAME; }

	const char* getPluginVersion() const override { return SWISH_PLUGIN_VERSION; }

	const PluginFieldCollection* getFieldNames() override { return &mFC; }

	IPluginV2* createPlugin(const char* name, const PluginFieldCollection* fc) override
	{
		return new SwishPlugin();
	}

	IPluginV2* deserializePlugin(const char* name, const void* serialData, size_t serialLength) override
	{

		return new SwishPlugin(serialData, serialLength);
	}

	void setPluginNamespace(const char* pluginNamespace) override
	{
		mPluginNamespace = pluginNamespace;
	}

	const char* getPluginNamespace() const override
	{
		return mPluginNamespace.c_str();
	}

private:
	static PluginFieldCollection mFC;
	static std::vector<PluginField> mPluginAttributes;
	std::string mPluginNamespace = "";
};

PluginFieldCollection SwishPluginCreator::mFC{};
std::vector<PluginField> SwishPluginCreator::mPluginAttributes;

REGISTER_TENSORRT_PLUGIN(SwishPluginCreator);