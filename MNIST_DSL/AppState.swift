//
//  AppState.swift
//  MNIST_DSL
//
//  Created by Kevin Coble on 1/22/26.
//

import Foundation
import Combine

struct TestResult {
    let numTrainEpochs: Int
    let numIncorrect: Int
}

struct InferResult: Identifiable {
    let digit: Int
    var probability: Double
    let id: UUID
}

@MainActor
class AppState: ObservableObject {
    @Published var readyForTraining = false
    @Published var currentStatus: String = "Loading Data Sets..."

    @Published var numTrainingEpochs: Int = 0
    @Published var testResults: [TestResult] = []
    
    @Published var probabilities: [InferResult] = []

    init() {
        probabilities = []
        for i in 0..<10 {
            let p = InferResult(digit: i, probability: 0.0, id: UUID())
            probabilities.append(p)
        }
    }
}
