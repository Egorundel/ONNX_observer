#include "onnx_observer.h"

int main() {
    // Create a TensorRT logger
    sample::Logger gLogger_;

    nvinfer1::IBuilder *builder = nvinfer1::createInferBuilder(gLogger_);
    initLibNvInferPlugins(&gLogger_, "");

    // Create a TensorRT network definition
    // For expicit batch:
    uint32_t flag = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    nvinfer1::INetworkDefinition *network = builder->createNetworkV2(flag);
    
    // For implicit batch:
    // nvinfer1::INetworkDefinition *network = builder->createNetworkV2(0U);

    // Create an ONNX parser
    nvonnxparser::IParser* parser = nvonnxparser::createParser(*network, gLogger_);

    // Parse the ONNX model from a file
    if (!parser->parseFromFile("./model.onnx", static_cast<int32_t>(nvinfer1::ILogger::Severity::kWARNING))) {
        std::cerr << "Failed to parse ONNX model." << std::endl;
        return -1;
    }

    // Get the number of input tensors
    int nbInputs = network->getNbInputs();
    std::cout << "Number of input tensors: " << nbInputs << std::endl;

    // Iterate over the input tensors and print their names and dimensions
    for (int i = 0; i < nbInputs; ++i) {
        nvinfer1::ITensor* inputTensor = network->getInput(i);
        std::cout << "Input tensor " << i << ": " << inputTensor->getName() << std::endl;
        std::cout << "Dimensions: ";
        for (int j = 0; j < inputTensor->getDimensions().nbDims; ++j) {
            std::cout << inputTensor->getDimensions().d[j] << " ";
        }
        std::cout << std::endl << std::endl << std::endl;
    }

    // Get the number of output tensors
    int nbOutputs = network->getNbOutputs();
    std::cout << "Number of output tensors: " << nbOutputs << std::endl;

    // Iterate over the output tensors and print their names
    for (int i = 0; i < nbOutputs; ++i) {
        nvinfer1::ITensor* outputTensor = network->getOutput(i);
        std::cout << "Output tensor " << i << ": " << outputTensor->getName() << std::endl;
        std::cout << "Dimensions: ";
        for (int j = 0; j < outputTensor->getDimensions().nbDims; ++j) {
            std::cout << outputTensor->getDimensions().d[j] << " ";
        }
        std::cout << std::endl << std::endl;
    }

    // Clean up
    delete parser;
    delete network;

    return 0;
}