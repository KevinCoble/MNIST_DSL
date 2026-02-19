//
//  ContentView.swift
//  MNIST_DSL
//
//  Created by Kevin Coble on 1/21/26.
//

import SwiftUI
import Charts
import MPSGraphDSL

let dataSets = MNISTDataSets()
let MNIST_Graph = MNISTGraph()
@MainActor
let appState = AppState()

struct ContentView: View {
    @ObservedObject var appStateObject = appState
    var body: some View {
        TabView {
            VStack {
                Image(systemName: "brain.head.profile")
                    .imageScale(.large)
                    .foregroundStyle(.tint)
                HStack {
                    Text("Training Status:")
                    Text(appStateObject.currentStatus)
                }
                HStack {
                    Text("Train:")
                    Button("1 Epoch") {
                        appState.readyForTraining = false
                        appStateObject.currentStatus = "Training 1 Epoch..."
                        Task {
                            try await MNIST_Graph.trainOneEpoch()
                            Task { @MainActor in
                                appState.readyForTraining = true
                                appStateObject.currentStatus = "Ready for training!"
                            }
                        }
                    }
                    Button("10 Epochs") {
                        appState.readyForTraining = false
                        appStateObject.currentStatus = "Training 10 Epochs..."
                        Task {
                            for _ in 0..<10 {
                                try await MNIST_Graph.trainOneEpoch()
                                Task { @MainActor in
                                    appState.readyForTraining = true
                                    appStateObject.currentStatus = "Ready for training!"
                                }
                            }
                         }
                    }
                    Button("100 Epochs") {
                        appState.readyForTraining = false
                        appStateObject.currentStatus = "Training 100 Epochs..."
                        Task {
                            for _ in 0..<100 {
                                try await MNIST_Graph.trainOneEpoch()
                                Task { @MainActor in
                                    appState.readyForTraining = true
                                    appStateObject.currentStatus = "Ready for training!"
                                }
                            }
                         }
                    }
                }
                .disabled(!appStateObject.readyForTraining)
                Text ("Trained \(appStateObject.numTrainingEpochs) Epochs")
                Chart(appStateObject.testResults, id: \.numTrainEpochs) {
                    LineMark(
                        x: .value("Num Epochs", $0.numTrainEpochs),
                        y: .value("Num Incorrect", $0.numIncorrect)
                    )
                    .lineStyle(StrokeStyle(lineWidth: 2))
                    .foregroundStyle(.blue.gradient)
                    .interpolationMethod(.linear)
                }
                .padding(5)
            }
            .tabItem {
                Label("Training", systemImage: "brain.head.profile")
            }
            VStack {
                InferenceView()
            }
            .tabItem {
                Label("Inference", systemImage: "pencil.circle")
            }
        }
        .onAppear {
            Task {
                //  See if the datasets need to be loaded
                if (await dataSets.state == .notLoaded) {
                    await dataSets.load()
                    if await dataSets.state == .loaded {
                        Task { @MainActor in
                            appStateObject.currentStatus = "Performing initial test..."
                        }
                        do {
                            try await MNIST_Graph.performTesting()
                            Task { @MainActor in
                                appState.readyForTraining = true
                                appStateObject.currentStatus = "Ready for training!"
                            }
                        }
                        catch {
                            Task { @MainActor in
                                appStateObject.currentStatus = "Failed initial test"
                            }
                        }
                    }
                    else {
                        Task { @MainActor in
                            appStateObject.currentStatus = "Failed to load datasets"
                        }
                    }
                }
            }
        }
    }
}

#Preview {
    ContentView()
}
