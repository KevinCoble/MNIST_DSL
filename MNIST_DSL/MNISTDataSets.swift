//
//  MNISTDataSets.swift
//  MNIST_DSL
//
//  Created by Kevin Coble on 1/21/26.
//

import Foundation
import MPSGraphDSL

actor MNISTDataSets {
    enum DataSetState {
        case notLoaded
        case loading
        case loaded
        case failedToLoad
    }
    let training : DataSet
    let testing : DataSet
    var state: DataSetState = .notLoaded
    
    init() {
        training = DataSet(inputShape: TensorShape([28, 28]), inputType: .float32, outputShape: TensorShape([10]), outputType: .float32)
        testing = DataSet(inputShape: TensorShape([28, 28]), inputType: .float32, outputShape: TensorShape([10]), outputType: .float32)
    }
    
    func load() async {
        state = .loading
        
        do {
            //  Set URLs as needed
            let imageTrainURL = Bundle.main.url(forResource:"train-images-idx3-ubyte", withExtension: "bdata")
            let labelTrainURL = Bundle.main.url(forResource:"train-labels-idx1-ubyte", withExtension: "bdata")
            let imageTestURL = Bundle.main.url(forResource:"t10k-images-idx3-ubyte", withExtension: "bdata")
            let labelTestURL = Bundle.main.url(forResource:"t10k-labels-idx1-ubyte", withExtension: "bdata")

            //  Load the data into memory
            let trainingInputData = try Data(contentsOf: imageTrainURL!)
            let trainingOutputData = try Data(contentsOf: labelTrainURL!)
            let testingInputData = try Data(contentsOf: imageTestURL!)
            let testingOutputData = try Data(contentsOf: labelTestURL!)

            //  Create the parsers
            let MNISTInputParser = DataParser {
                UnusedData(length: 16, format: .fUInt8)
                RepeatSampleTillDone {
                    RepeatDimension(count: 28, dimension: .Dimension0, affects: .input) {
                        InputData(length: 28, format : .fUInt8, postProcessing : .Scale_0_1)
                        SetDimension(dimension: .Dimension1, toValue: 0, affects: .input)
                    }
                }
            }
            let MNISTOutputParser = DataParser {
                UnusedData(length: 8, format: .fUInt8)
                RepeatSampleTillDone {
                    LabelIndex(count: 1, format: .fUInt8)
                }
            }

//            let twoSample = false
//            if (twoSample) {
//                let tempData = DataSet(inputShape: TensorShape([28, 28]), inputType: .float32, outputShape: TensorShape([10]), outputType: .float32)
//                //  Parse the data concurrently from data
//                try await MNISTInputParser.parseBinaryData(testingInputData, intoDataSet: tempData)
//                try await MNISTOutputParser.parseBinaryData(testingOutputData, intoDataSet: tempData)
//                
//                for i in 0..<10 {
//                    let sample = try await tempData.getSample(sampleIndex: i)
//                    try await training.appendSample(sample)
//                    try await testing.appendSample(sample)
//                }
//            }
//            else {
                //  Parse the data concurrently from data
                try await MNISTInputParser.parseBinaryData(trainingInputData, intoDataSet: training)
                try await MNISTOutputParser.parseBinaryData(trainingOutputData, intoDataSet: training)
                try await MNISTInputParser.parseBinaryData(testingInputData, intoDataSet: testing)
                try await MNISTOutputParser.parseBinaryData(testingOutputData, intoDataSet: testing)
//            }

            state = .loaded            
        }
        catch {
            state = .failedToLoad
        }
    }
}
