//
//  InferenceView.swift
//  MNIST_DSL
//
//  Created by Kevin Coble on 2/18/26.
//

import SwiftUI
import Charts

struct InferenceView: View {
    @State private var pixels: [UInt8] = Array(repeating: 0, count: 28*28)
    @ObservedObject var appStateObject = appState
    @State private var prediction: Int = 0

    var body: some View {
        VStack {
            HStack {
                Button("Reset") {
                    pixels = Array(repeating: 0, count: 28*28)
                    
                    //  Run the data through inference
                    infer()
                }
                .disabled(!appStateObject.readyForTraining)
                Canvas()  { context, size in
                    let rectSize = CGSize(width: 10, height: 10)
                    for row in 0..<28 {
                        let y = CGFloat(row) * rectSize.height
                        for col in 0..<28 {
                            let rect = CGRect(x: CGFloat(col) * rectSize.width, y: y, width: rectSize.width, height: rectSize.height)
                            let path = Rectangle().path(in: rect)
                            let pixelIndex = row * 28 + col
                            let pixelValue = Double(pixels[pixelIndex]) / 255.0
                            let color = Color(red: pixelValue, green: pixelValue, blue: pixelValue)
                            context.fill(path, with: .color(color))
                        }
                    }
                }
                .frame(width: 280, height: 280)
                .gesture(
                    DragGesture()
                        .onChanged { value in
                            if (appStateObject.readyForTraining  ) {              let pixelx = Int(value.location.x / 10)
                                let pixely = Int(value.location.y / 10)
                                if ((pixelx >= 0 && pixelx < 28) &&
                                    (pixely >= 0 && pixely < 28)) {
                                    pixels[pixelx + pixely * 28] = 255
                                }
                                
                                //  Run the data through inference
                                infer()
                            }
                        }
                        .onEnded { value in
                        }
                )
                VStack {
                    Text("Prediction:")
                    Text("\(prediction)")
                        .font(Font.largeTitle.bold())
                }
            }
            Chart(appStateObject.probabilities) { probability in
                BarMark(
                    x: .value("Digit", probability.digit),
                    y: .value("Probability", probability.probability)
                )
            }
            .chartXAxis {
                AxisMarks(values: .automatic(desiredCount: 10))
            }
            .chartXScale(domain: [0, 9])
            .chartYScale(domain: [0, 1.0])
        }
    }
    
    func infer() {
        do {
            let inferResults = try MNIST_Graph.infer(pixels: pixels)
            var predictedDigit = 0
            var maxProb: Double = -1.0
            for i in 0..<10 {
                appStateObject.probabilities[i].probability = inferResults[i]
                if (inferResults[i] > maxProb) {
                    predictedDigit = i
                    maxProb = inferResults[i]
                }
            }
            prediction = predictedDigit
        }
        catch {
            print("Error inferring digit probababilities")
        }
    }
}

#Preview {
    InferenceView()
}
