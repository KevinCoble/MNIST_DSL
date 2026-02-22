//
//  MNISTGraph.swift
//  MNIST_DSL
//
//  Created by Kevin Coble on 1/22/26.
//

import Foundation
import MetalPerformanceShaders
import MetalPerformanceShadersGraph
import MPSGraphDSL

struct MNISTGraph {
    let graph: Graph
    let batchSize: Int = 16
    
    init() {
        graph = Graph(batchSize: batchSize) {
            PlaceHolder(shape: [28, 28], type: .float32, name: "input")
            ConvolutionLayer(input: "input", kernelHeight: 5, kernelWidth: 5, numFilters: 32, activationFunction: .relu, heightStride: 1, widthStride: 1, name: "conv0")
                .leaveFilterDimensionLast()         //  Filter in channel dimension for pooling
                .learnWithRespectTo("loss")
            PoolingLayer(function: .max, kernelHeight: 2, kernelWidth: 2, heightStride: 2, widthStride: 2, name: "pool0")
            ConvolutionLayer(kernelHeight: 5, kernelWidth: 5, numFilters: 64, activationFunction: .relu, heightStride: 1, widthStride: 1, name: "conv1")
                .leaveFilterDimensionLast()         //  Filter in channel dimension for pooling
                .learnWithRespectTo("loss")
            PoolingLayer(function: .max, kernelHeight: 2, kernelWidth: 2, heightStride: 2, widthStride: 2, name: "pool1")
            FullyConnectedLayer(outputShape: TensorShape([1024]), activationFunction: .relu, name: "fc0")
                .learnWithRespectTo("loss")
            FullyConnectedLayer(outputShape: TensorShape([10]), activationFunction: .none, name: "fc1")
                .learnWithRespectTo("loss")
                .targetForModes(["learn"])
            SoftMax(name: "result")
                .targetForModes(["infer"])
            PlaceHolder(shape: [10], type: .float32, modes: ["learn"], name: "labels")
            SoftMaxCrossEntropy(input: "fc1", labels: "labels", reductionType: .sum, name: "loss")
                .targetForModes(["learn"])
            Learning(learningRate: 0.01, learningModes: ["learn"])
        }
    }
    
    func performTesting() async throws {
        let results = try await graph.runClassifierTest(mode: "infer", testDataSet: dataSets.testing, inputTensorName: "input", resultTensorName: "result",)
        let numEpochs = appState.numTrainingEpochs
        let numTestingSamples = await dataSets.testing.numSamples
        let numIncorrect = numTestingSamples - results.totalCorrect
        let testResult = TestResult(numTrainEpochs: numEpochs, numIncorrect: numIncorrect)
        appState.testResults.append(testResult)
    }
    
    func trainOneEpoch() async throws {
        let loss = try await graph.runTraining(mode: "learn", trainingDataSet: dataSets.training, inputTensorName: "input", expectedValueTensorName: "labels", lossTensorName: "loss", epochSize: 300)
        appState.numTrainingEpochs += 1
        print("Total training loss: \(loss!)")
        try await performTesting()
    }
    
    func infer(pixels: [UInt8]) throws -> [Double] {
        //  Create an input tensor from the pixels
        var batchRepeatPixels: [Float32] = []
        let floatPixels = pixels.map{ Float32($0) / 255.0 }
        for _ in 0..<batchSize { batchRepeatPixels += floatPixels }
        let inputTensor = try TensorFloat32(shape: TensorShape([batchSize, 28, 28]), initialValues: batchRepeatPixels)
        
        //  Run the one batch tensor through
        let results = try graph.runOne(mode: "infer", inputTensors: ["input" : inputTensor])
        
        //  Get the result values
        let resultBatchTensor = results["result"]!
        let resultTensor = try resultBatchTensor.getTensorForBatch(0)
        let probabilities: [Double] = resultTensor.getElements()
        return probabilities
    }
}
