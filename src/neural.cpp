#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <iomanip>
#include <chrono>
#include <algorithm>
#include <numeric>
#include <fstream>
#include <queue>
#include <memory>
#include <bitset>

class NeuralNetwork {
private:
    // Struktur jaringan
    std::vector<int> layerSizes;
    std::vector<std::vector<std::vector<double>>> weights;
    std::vector<std::vector<double>> biases;
    
    // Momentum terms untuk persamaan (9): Δw(t) = -ε ∂E/∂w(t) + α Δw(t-1)
    std::vector<std::vector<std::vector<double>>> weightVelocities;  // Δw(t-1)
    std::vector<std::vector<double>> biasVelocities;                 // Δb(t-1)
    
    // Gradient accumulators (untuk akumulasi ∂E/∂w sebelum update)
    std::vector<std::vector<std::vector<double>>> weightGradientAccumulators;
    std::vector<std::vector<double>> biasGradientAccumulators;
    
    // Aktivasi dan output
    std::vector<std::vector<double>> activations;  // x_j
    std::vector<std::vector<double>> outputs;      // y_j
    
    // Training history
    std::vector<double> trainingErrors;
    std::vector<double> validationErrors;
    
    // Parameter momentum dan learning rate
    double epsilon;      // ε dalam persamaan (8) dan (9)
    double alpha;        // α dalam persamaan (9), exponential decay factor
    
    // Fungsi aktivasi dan turunannya
    double sigmoid(double x) {
        // Clip nilai untuk menghindari overflow
        if (x < -45.0) x = -45.0;
        if (x > 45.0) x = 45.0;
        return 1.0 / (1.0 + exp(-x));
    }
    
    double sigmoidDerivativeFromOutput(double y) {
        return y * (1.0 - y);
    }
    
    double sigmoidDerivative(double x) {
        double s = sigmoid(x);
        return s * (1.0 - s);
    }
    
    // Persamaan (4): ∂E/∂y_j = y_j - d_j
    double compute_dE_dy(double y, double d) {
        return y - d;
    }
    
    // Persamaan (5): ∂E/∂x_j = ∂E/∂y_j * y_j(1-y_j)
    double compute_dE_dx(double dE_dy, double y) {
        return dE_dy * y * (1.0 - y);
    }
    
    // Persamaan (6): ∂E/∂w_ji = ∂E/∂x_j * y_i
    double compute_dE_dw(double dE_dx, double y_i) {
        return dE_dx * y_i;
    }
    
    // Persamaan (7): ∂E/∂y_i = Σ_j ∂E/∂x_j * w_ji
    double compute_dE_dy_i(int layer, int neuron, const std::vector<double>& deltas_next) {
        double sum = 0.0;
        for (size_t j = 0; j < deltas_next.size(); j++) {
            sum += deltas_next[j] * weights[layer][j][neuron];
        }
        return sum;
    }
    
    // Persamaan (9): Δw(t) = -ε ∂E/∂w(t) + α Δw(t-1)
    double compute_weight_update(double gradient, double previous_velocity) {
        return -epsilon * gradient + alpha * previous_velocity;
    }
    
    // Inisialisasi gradient accumulators
    void initializeGradientAccumulators() {
        weightGradientAccumulators.resize(weights.size());
        biasGradientAccumulators.resize(biases.size());
        
        for (size_t layer = 0; layer < weights.size(); layer++) {
            weightGradientAccumulators[layer].resize(weights[layer].size());
            biasGradientAccumulators[layer].resize(biases[layer].size(), 0.0);
            
            for (size_t neuron = 0; neuron < weights[layer].size(); neuron++) {
                weightGradientAccumulators[layer][neuron].resize(
                    weights[layer][neuron].size(), 0.0);
            }
        }
    }
    
    // Reset gradient accumulators ke 0
    void resetGradientAccumulators() {
        for (size_t layer = 0; layer < weightGradientAccumulators.size(); layer++) {
            for (size_t neuron = 0; neuron < weightGradientAccumulators[layer].size(); neuron++) {
                std::fill(weightGradientAccumulators[layer][neuron].begin(),
                         weightGradientAccumulators[layer][neuron].end(), 0.0);
                biasGradientAccumulators[layer][neuron] = 0.0;
            }
        }
    }
    
    // Inisialisasi bobot dengan random kecil (seperti di paper: "To break symmetry we start with small random weights")
    void initializeWeights() {
        std::random_device rd;
        std::mt19937 gen(rd());
        
        weights.clear();
        biases.clear();
        weightVelocities.clear();
        biasVelocities.clear();
        
        for (size_t layer = 1; layer < layerSizes.size(); layer++) {
            int currentLayerSize = layerSizes[layer];
            int prevLayerSize = layerSizes[layer - 1];
            
            // Inisialisasi dengan random kecil (seperti dalam paper)
            // Menggunakan distribusi normal dengan std dev kecil
            double stddev = 0.1;  // Small random weights
            std::normal_distribution<double> dist(0.0, stddev);
            
            std::vector<std::vector<double>> layerWeights(currentLayerSize, 
                std::vector<double>(prevLayerSize));
            std::vector<double> layerBiases(currentLayerSize);
            
            std::vector<std::vector<double>> layerWeightVelocities(currentLayerSize,
                std::vector<double>(prevLayerSize, 0.0));
            std::vector<double> layerBiasVelocities(currentLayerSize, 0.0);
            
            for (int neuron = 0; neuron < currentLayerSize; neuron++) {
                for (int input = 0; input < prevLayerSize; input++) {
                    layerWeights[neuron][input] = dist(gen);
                }
                layerBiases[neuron] = 0.0;  // Bias biasanya diinisialisasi dengan 0
            }
            
            weights.push_back(layerWeights);
            biases.push_back(layerBiases);
            weightVelocities.push_back(layerWeightVelocities);
            biasVelocities.push_back(layerBiasVelocities);
        }
        
        // Inisialisasi gradient accumulators
        initializeGradientAccumulators();
    }

public:
    // Constructor dengan parameter opsional untuk epsilon dan alpha
    NeuralNetwork(const std::vector<int>& architecture, 
                  double epsilon = 0.01, 
                  double alpha = 0.9) 
        : layerSizes(architecture), epsilon(epsilon), alpha(alpha) {
        
        if (layerSizes.size() < 2) {
            throw std::invalid_argument("Network must have at least 2 layers (input and output)");
        }
        
        initializeWeights();
        
        // Inisialisasi storage untuk aktivasi dan output
        activations.resize(layerSizes.size());
        outputs.resize(layerSizes.size());
        
        for (size_t i = 0; i < layerSizes.size(); i++) {
            activations[i].resize(layerSizes[i], 0.0);
            outputs[i].resize(layerSizes[i], 0.0);
        }
    }
    
    // Setter untuk epsilon dan alpha
    void setLearningRate(double new_epsilon) { epsilon = new_epsilon; }
    void setMomentum(double new_alpha) { alpha = new_alpha; }
    
    // Forward propagation
    std::vector<double> forward(const std::vector<double>& input) {
        if (input.size() != static_cast<size_t>(layerSizes[0])) {
            throw std::invalid_argument("Input size does not match input layer size");
        }
        
        // Layer input
        outputs[0] = input;
        
        // Untuk setiap layer dari 1 hingga akhir
        for (size_t layer = 1; layer < layerSizes.size(); layer++) {
            int currentLayerIdx = layer;
            int prevLayerIdx = layer - 1;
            
            // Untuk setiap neuron di layer saat ini
            for (int neuron = 0; neuron < layerSizes[currentLayerIdx]; neuron++) {
                // Persamaan (1): x_j = Σ_i y_i * w_ji + bias
                double activation = biases[layer-1][neuron];
                
                for (int inputNeuron = 0; inputNeuron < layerSizes[prevLayerIdx]; inputNeuron++) {
                    activation += outputs[prevLayerIdx][inputNeuron] * 
                                  weights[layer-1][neuron][inputNeuron];
                }
                
                activations[currentLayerIdx][neuron] = activation;
                
                // Persamaan (2): y_j = 1 / (1 + e^(-x_j))
                outputs[currentLayerIdx][neuron] = sigmoid(activation);
            }
        }
        
        return outputs.back();
    }
    
    // Backward pass untuk satu contoh
    void backwardPassSingleExample(const std::vector<double>& target, 
                                   bool accumulate = false) {
        int numLayers = layerSizes.size();
        
        // Vector untuk menyimpan ∂E/∂x_j untuk setiap neuron (deltas)
        std::vector<std::vector<double>> deltas(numLayers);
        for (int i = 0; i < numLayers; i++) {
            deltas[i].resize(layerSizes[i], 0.0);
        }
        
        // 1. Hitung delta untuk output layer (layer terakhir)
        int outputLayerIdx = numLayers - 1;
        
        for (int j = 0; j < layerSizes[outputLayerIdx]; j++) {
            double y_j = outputs[outputLayerIdx][j];
            double d_j = target[j];
            
            // Persamaan (4): ∂E/∂y_j = y_j - d_j
            double dE_dy = compute_dE_dy(y_j, d_j);
            
            // Persamaan (5): ∂E/∂x_j = ∂E/∂y_j * y_j(1-y_j)
            double dE_dx = compute_dE_dx(dE_dy, y_j);
            
            deltas[outputLayerIdx][j] = dE_dx;
        }
        
        // 2. Backpropagate melalui hidden layers menggunakan persamaan (7)
        for (int layer = outputLayerIdx - 1; layer >= 1; layer--) {
            for (int i = 0; i < layerSizes[layer]; i++) {
                // Persamaan (7): ∂E/∂y_i = Σ_j ∂E/∂x_j * w_ji
                double dE_dy_i = compute_dE_dy_i(layer, i, deltas[layer + 1]);
                
                // y_i dari neuron saat ini
                double y_i = outputs[layer][i];
                
                // ∂E/∂x_i = ∂E/∂y_i * y_i(1-y_i)
                double dE_dx_i = dE_dy_i * sigmoidDerivativeFromOutput(y_i);
                
                deltas[layer][i] = dE_dx_i;
            }
        }
        
        // 3. Hitung gradien untuk weights dan biases
        for (int layer = 1; layer < numLayers; layer++) {
            for (int j = 0; j < layerSizes[layer]; j++) {
                // ∂E/∂b_j = ∂E/∂x_j
                double dE_db = deltas[layer][j];
                
                if (accumulate) {
                    biasGradientAccumulators[layer-1][j] += dE_db;
                } else {
                    // Untuk online learning, kita akan update nanti
                }
                
                // Untuk setiap input i ke neuron j
                for (int i = 0; i < layerSizes[layer-1]; i++) {
                    // Persamaan (6): ∂E/∂w_ji = ∂E/∂x_j * y_i
                    double dE_dw = compute_dE_dw(deltas[layer][j], outputs[layer-1][i]);
                    
                    if (accumulate) {
                        weightGradientAccumulators[layer-1][j][i] += dE_dw;
                    } else {
                        // Untuk online learning, kita akan update nanti
                    }
                }
            }
        }
    }
    
    // Update weights dengan momentum (persamaan 9)
    void updateWeightsWithMomentum() {
        // Persamaan (9): Δw(t) = -ε ∂E/∂w(t) + α Δw(t-1)
        for (size_t layer = 0; layer < weights.size(); layer++) {
            for (size_t neuron = 0; neuron < weights[layer].size(); neuron++) {
                // Update bias dengan momentum
                double biasGradient = biasGradientAccumulators[layer][neuron];
                biasVelocities[layer][neuron] = 
                    compute_weight_update(biasGradient, biasVelocities[layer][neuron]);
                biases[layer][neuron] += biasVelocities[layer][neuron];
                
                // Update weights dengan momentum
                for (size_t input = 0; input < weights[layer][neuron].size(); input++) {
                    double weightGradient = weightGradientAccumulators[layer][neuron][input];
                    weightVelocities[layer][neuron][input] = 
                        compute_weight_update(weightGradient, weightVelocities[layer][neuron][input]);
                    
                    weights[layer][neuron][input] += weightVelocities[layer][neuron][input];
                }
            }
        }
    }
    
    // Batch learning dengan momentum (seperti di paper)
    double trainBatchWithMomentum(const std::vector<std::vector<double>>& inputs,
                                 const std::vector<std::vector<double>>& targets) {
        if (inputs.size() != targets.size()) {
            throw std::invalid_argument("Inputs and targets must have same size");
        }
        
        // Reset gradient accumulators
        resetGradientAccumulators();
        
        double totalError = 0.0;
        
        // Akumulasi gradien untuk semua input-output cases
        for (size_t c = 0; c < inputs.size(); c++) {
            // Forward pass
            std::vector<double> prediction = forward(inputs[c]);
            
            // Hitung error untuk case ini
            for (size_t j = 0; j < prediction.size(); j++) {
                double diff = prediction[j] - targets[c][j];
                totalError += 0.5 * diff * diff;
            }
            
            // Backward pass dengan akumulasi
            backwardPassSingleExample(targets[c], true);
        }
        
        // Update weights dengan momentum (persamaan 9)
        updateWeightsWithMomentum();
        
        return totalError / inputs.size();
    }
    
    // Training dengan epochs menggunakan momentum
    void trainWithMomentum(const std::vector<std::vector<double>>& trainingInputs,
                          const std::vector<std::vector<double>>& trainingTargets,
                          const std::vector<std::vector<double>>& validationInputs,
                          const std::vector<std::vector<double>>& validationTargets,
                          int epochs = 100,
                          bool verbose = true) {
        
        std::cout << "\n=== Training with Momentum (Equation 9) ===\n";
        std::cout << "Δw(t) = -ε ∂E/∂w(t) + α Δw(t-1)\n";
        std::cout << "where ε = " << epsilon << ", α = " << alpha << "\n";
        
        trainingErrors.clear();
        validationErrors.clear();
        
        for (int epoch = 0; epoch < epochs; epoch++) {
            // Shuffle training data
            std::vector<size_t> indices(trainingInputs.size());
            std::iota(indices.begin(), indices.end(), 0);
            std::random_device rd;
            std::mt19937 g(rd());
            std::shuffle(indices.begin(), indices.end(), g);
            
            double epochTrainingError = 0.0;
            
            // Akumulasi gradien untuk batch penuh
            std::vector<std::vector<double>> shuffledInputs;
            std::vector<std::vector<double>> shuffledTargets;
            
            for (size_t idx : indices) {
                shuffledInputs.push_back(trainingInputs[idx]);
                shuffledTargets.push_back(trainingTargets[idx]);
            }
            
            epochTrainingError = trainBatchWithMomentum(shuffledInputs, shuffledTargets);
            trainingErrors.push_back(epochTrainingError);
            
            // Validation error
            double epochValidationError = 0.0;
            for (size_t i = 0; i < validationInputs.size(); i++) {
                std::vector<double> prediction = forward(validationInputs[i]);
                for (size_t j = 0; j < prediction.size(); j++) {
                    double diff = prediction[j] - validationTargets[i][j];
                    epochValidationError += 0.5 * diff * diff;
                }
            }
            epochValidationError /= validationInputs.size();
            validationErrors.push_back(epochValidationError);
            
            // Print progress
            if (verbose && (epoch % std::max(1, epochs / 20) == 0 || epoch == epochs - 1)) {
                std::cout << "Epoch " << std::setw(4) << epoch 
                          << " | Train Error: " << std::fixed << std::setprecision(6) 
                          << std::setw(10) << epochTrainingError
                          << " | Val Error: " << std::setw(10) << epochValidationError 
                          << " | ε: " << epsilon << "\n";
            }
            
            // Optional: learning rate decay
            // epsilon *= 0.995;  // Uncomment untuk learning rate decay
        }
        
        std::cout << "=== Training Complete ===\n";
    }
    
    // Print velocity information (untuk debugging momentum)
    void printVelocityInfo() const {
        std::cout << "\n=== Velocity Information (Δw(t-1)) ===\n";
        for (size_t layer = 0; layer < weightVelocities.size(); layer++) {
            std::cout << "Layer " << layer + 1 << ":\n";
            double maxVelocity = 0.0;
            double minVelocity = 0.0;
            double avgVelocity = 0.0;
            int count = 0;
            
            for (size_t neuron = 0; neuron < weightVelocities[layer].size(); neuron++) {
                for (size_t input = 0; input < weightVelocities[layer][neuron].size(); input++) {
                    double v = weightVelocities[layer][neuron][input];
                    maxVelocity = std::max(maxVelocity, v);
                    minVelocity = std::min(minVelocity, v);
                    avgVelocity += v;
                    count++;
                }
            }
            
            if (count > 0) {
                avgVelocity /= count;
                std::cout << "  Weight velocities: Max=" << maxVelocity 
                          << ", Min=" << minVelocity 
                          << ", Avg=" << avgVelocity << "\n";
            }
        }
    }
    
    // Print semua persamaan yang diimplementasi
    void printAllEquations() const {
        std::cout << "\n=== All Implemented Equations ===\n";
        std::cout << "Forward propagation:\n";
        std::cout << "  (1) x_j = Σ_i y_i * w_ji + bias\n";
        std::cout << "  (2) y_j = 1 / (1 + e^(-x_j))\n";
        std::cout << "\nError calculation:\n";
        std::cout << "  (3) E = 1/2 * Σ_c Σ_j (y_j,c - d_j,c)^2\n";
        std::cout << "\nBackward propagation:\n";
        std::cout << "  (4) ∂E/∂y_j = y_j - d_j\n";
        std::cout << "  (5) ∂E/∂x_j = ∂E/∂y_j * y_j(1-y_j)\n";
        std::cout << "  (6) ∂E/∂w_ji = ∂E/∂x_j * y_i\n";
        std::cout << "  (7) ∂E/∂y_i = Σ_j ∂E/∂x_j * w_ji\n";
        std::cout << "\nWeight update (simple gradient descent):\n";
        std::cout << "  (8) Δw = -ε ∂E/∂w\n";
        std::cout << "\nWeight update with momentum:\n";
        std::cout << "  (9) Δw(t) = -ε ∂E/∂w(t) + α Δw(t-1)\n";
        std::cout << "      where t is incremented for each sweep through all cases\n";
        std::cout << "      α is exponential decay factor (0 < α < 1)\n";
    }
    
    // ... [fungsi-fungsi lainnya seperti saveModel, loadModel, dll.] ...
    
    // Predict untuk multiple inputs
    std::vector<std::vector<double>> predict(const std::vector<std::vector<double>>& inputs) {
        std::vector<std::vector<double>> predictions;
        predictions.reserve(inputs.size());
        
        for (const auto& input : inputs) {
            predictions.push_back(forward(input));
        }
        
        return predictions;
    }
    
    // Save model
    void saveModel(const std::string& filename) {
        std::ofstream file(filename);
        if (!file.is_open()) {
            throw std::runtime_error("Cannot open file for writing: " + filename);
        }
        
        file << layerSizes.size() << "\n";
        for (int size : layerSizes) {
            file << size << " ";
        }
        file << "\n";
        
        file << epsilon << " " << alpha << "\n";
        
        for (size_t layer = 0; layer < weights.size(); layer++) {
            file << weights[layer].size() << " " 
                 << (weights[layer].empty() ? 0 : weights[layer][0].size()) << "\n";
            
            for (size_t neuron = 0; neuron < weights[layer].size(); neuron++) {
                for (double w : weights[layer][neuron]) {
                    file << w << " ";
                }
                file << biases[layer][neuron] << "\n";
            }
        }
        
        file.close();
    }
    
    // Load model
    void loadModel(const std::string& filename) {
        std::ifstream file(filename);
        if (!file.is_open()) {
            throw std::runtime_error("Cannot open file for reading: " + filename);
        }
        
        int numLayers;
        file >> numLayers;
        
        layerSizes.resize(numLayers);
        for (int i = 0; i < numLayers; i++) {
            file >> layerSizes[i];
        }
        
        file >> epsilon >> alpha;
        
        initializeWeights();
        
        for (size_t layer = 0; layer < weights.size(); layer++) {
            int numNeurons, numInputs;
            file >> numNeurons >> numInputs;
            
            for (int neuron = 0; neuron < numNeurons; neuron++) {
                for (int input = 0; input < numInputs; input++) {
                    file >> weights[layer][neuron][input];
                }
                file >> biases[layer][neuron];
            }
        }
        
        file.close();
    }
    
    // Print network information
    void printInfo() const {
        std::cout << "\n=== Neural Network Information ===\n";
        std::cout << "Architecture: ";
        for (size_t i = 0; i < layerSizes.size(); i++) {
            std::cout << layerSizes[i];
            if (i < layerSizes.size() - 1) std::cout << " -> ";
        }
        std::cout << "\n";
        std::cout << "Parameters: ε = " << epsilon << ", α = " << alpha << "\n";
        
        int totalWeights = 0;
        int totalBiases = 0;
        
        for (size_t layer = 1; layer < layerSizes.size(); layer++) {
            int weightsInLayer = layerSizes[layer] * layerSizes[layer - 1];
            int biasesInLayer = layerSizes[layer];
            
            std::cout << "Layer " << layer << " (" << layerSizes[layer] << " neurons): "
                      << weightsInLayer << " weights, " << biasesInLayer << " biases\n";
            
            totalWeights += weightsInLayer;
            totalBiases += biasesInLayer;
        }
        
        std::cout << "Total parameters: " << totalWeights + totalBiases 
                  << " (" << totalWeights << " weights, " << totalBiases << " biases)\n";
    }
    
    // Getter
    const std::vector<double>& getTrainingErrors() const { return trainingErrors; }
    const std::vector<double>& getValidationErrors() const { return validationErrors; }
    const std::vector<std::vector<std::vector<double>>>& getWeights() const { return weights; }
    const std::vector<std::vector<double>>& getBiases() const { return biases; }
    double getEpsilon() const { return epsilon; }
    double getAlpha() const { return alpha; }
};

// Dataset generator untuk symmetry detection (seperti di paper)
class SymmetryDataset {
public:
    // Generate semua kemungkinan binary patterns untuk array dengan panjang tertentu
    static std::vector<std::vector<double>> generateAllBinaryPatterns(int length) {
        int totalPatterns = 1 << length;  // 2^length
        std::vector<std::vector<double>> patterns;
        patterns.reserve(totalPatterns);
        
        for (int i = 0; i < totalPatterns; i++) {
            std::vector<double> pattern(length);
            for (int bit = 0; bit < length; bit++) {
                pattern[bit] = (i >> bit) & 1;  // Extract each bit
            }
            patterns.push_back(pattern);
        }
        
        return patterns;
    }
    
    // Generate symmetry dataset seperti di paper
    static std::pair<std::vector<std::vector<double>>, std::vector<std::vector<double>>> 
    generateSymmetryDataset(int arrayLength, int numExamples = -1) {
        if (arrayLength <= 0 || arrayLength % 2 != 0) {
            throw std::invalid_argument("Array length must be positive and even for symmetry detection");
        }
        
        std::vector<std::vector<double>> inputs;
        std::vector<std::vector<double>> targets;
        
        if (numExamples <= 0) {
            // Gunakan semua kemungkinan patterns
            inputs = generateAllBinaryPatterns(arrayLength);
            targets.reserve(inputs.size());
            
            for (const auto& input : inputs) {
                // Check if the pattern is symmetric
                bool symmetric = true;
                for (int i = 0; i < arrayLength / 2; i++) {
                    if (input[i] != input[arrayLength - 1 - i]) {
                        symmetric = false;
                        break;
                    }
                }
                targets.push_back({symmetric ? 1.0 : 0.0});
            }
        } else {
            // Generate random examples
            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_int_distribution<> dist(0, 1);
            std::uniform_real_distribution<> probDist(0.0, 1.0);
            
            for (int i = 0; i < numExamples; i++) {
                std::vector<double> pattern(arrayLength);
                for (int j = 0; j < arrayLength; j++) {
                    pattern[j] = dist(gen);
                }
                
                inputs.push_back(pattern);
                
                // Determine if symmetric
                bool symmetric = true;
                for (int j = 0; j < arrayLength / 2; j++) {
                    if (pattern[j] != pattern[arrayLength - 1 - j]) {
                        symmetric = false;
                        break;
                    }
                }
                targets.push_back({symmetric ? 1.0 : 0.0});
            }
        }
        
        return {inputs, targets};
    }
    
    // Generate dataset dengan noise
    static std::pair<std::vector<std::vector<double>>, std::vector<std::vector<double>>> 
    generateNoisySymmetryDataset(int arrayLength, int numExamples, double noiseLevel = 0.1) {
        auto [cleanInputs, cleanTargets] = generateSymmetryDataset(arrayLength, numExamples);
        
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> noiseDist(-noiseLevel, noiseLevel);
        
        std::vector<std::vector<double>> noisyInputs = cleanInputs;
        for (auto& input : noisyInputs) {
            for (double& val : input) {
                val += noiseDist(gen);
                // Clip ke [0, 1]
                if (val < 0.0) val = 0.0;
                if (val > 1.0) val = 1.0;
            }
        }
        
        return {noisyInputs, cleanTargets};
    }
    
    // Print patterns dengan indikasi symmetry
    static void printPatternsWithSymmetry(const std::vector<std::vector<double>>& patterns,
                                         const std::vector<std::vector<double>>& targets,
                                         int maxToPrint = 20) {
        std::cout << "\n=== Sample Patterns and Symmetry ===\n";
        std::cout << std::setw(15) << "Pattern" << std::setw(15) << "Symmetric" 
                  << std::setw(15) << "Target\n";
        
        int count = std::min(maxToPrint, (int)patterns.size());
        for (int i = 0; i < count; i++) {
            std::cout << "[";
            for (size_t j = 0; j < patterns[i].size(); j++) {
                std::cout << (patterns[i][j] > 0.5 ? "1" : "0");
                if (j < patterns[i].size() - 1) std::cout << " ";
            }
            std::cout << "]";
            
            // Check actual symmetry
            bool isSymmetric = true;
            for (size_t j = 0; j < patterns[i].size() / 2; j++) {
                if (patterns[i][j] != patterns[i][patterns[i].size() - 1 - j]) {
                    isSymmetric = false;
                    break;
                }
            }
            
            std::cout << std::setw(10) << (isSymmetric ? "Yes" : "No")
                      << std::setw(15) << targets[i][0] << "\n";
        }
    }
};

// Class untuk menjalankan symmetry detection experiment seperti di paper
class SymmetryExperiment {
private:
    int arrayLength;
    
public:
    SymmetryExperiment(int length = 6) : arrayLength(length) {
        if (length <= 0) {
            throw std::invalid_argument("Array length must be positive");
        }
    }
    
    void runExperiment(bool useMomentum = true) {
        std::cout << "\n" << std::string(60, '=') << "\n";
        std::cout << "SYMMETRY DETECTION EXPERIMENT\n";
        std::cout << "As described in the paper by Rumelhart, Hinton, & Williams (1986)\n";
        std::cout << "Array length: " << arrayLength << "\n";
        std::cout << std::string(60, '=') << "\n\n";
        
        // Quote dari paper
        std::cout << "From the paper:\n";
        std::cout << "\"One simple task that cannot be done by just connecting the\n";
        std::cout << "input units to the output units is the detection of symmetry.\n";
        std::cout << "To detect whether the binary activity levels of a one-dimensional\n";
        std::cout << "array of input units are symmetrical about the centre point,\n";
        std::cout << "it is essential to use an intermediate layer...\"\n\n";
        
        // Generate dataset
        std::cout << "Generating symmetry dataset...\n";
        auto [allInputs, allTargets] = SymmetryDataset::generateSymmetryDataset(arrayLength);
        
        // Print some examples
        SymmetryDataset::printPatternsWithSymmetry(allInputs, allTargets, 10);
        
        // Split dataset
        std::vector<int> indices(allInputs.size());
        std::iota(indices.begin(), indices.end(), 0);
        std::random_device rd;
        std::mt19937 g(rd());
        std::shuffle(indices.begin(), indices.end(), g);
        
        size_t trainSize = allInputs.size() * 0.7;
        size_t valSize = allInputs.size() * 0.15;
        
        std::vector<std::vector<double>> trainInputs, valInputs, testInputs;
        std::vector<std::vector<double>> trainTargets, valTargets, testTargets;
        
        for (size_t i = 0; i < indices.size(); i++) {
            if (i < trainSize) {
                trainInputs.push_back(allInputs[indices[i]]);
                trainTargets.push_back(allTargets[indices[i]]);
            } else if (i < trainSize + valSize) {
                valInputs.push_back(allInputs[indices[i]]);
                valTargets.push_back(allTargets[indices[i]]);
            } else {
                testInputs.push_back(allInputs[indices[i]]);
                testTargets.push_back(allTargets[indices[i]]);
            }
        }
        
        std::cout << "\nDataset sizes:\n";
        std::cout << "  Training: " << trainInputs.size() << " examples\n";
        std::cout << "  Validation: " << valInputs.size() << " examples\n";
        std::cout << "  Testing: " << testInputs.size() << " examples\n";
        
        // Experiment 1: Network tanpa hidden layer (hanya input-output)
        std::cout << "\n" << std::string(60, '-') << "\n";
        std::cout << "EXPERIMENT 1: Network without hidden layer\n";
        std::cout << "Architecture: " << arrayLength << " -> 1\n";
        std::cout << "This should fail at symmetry detection (as stated in the paper)\n";
        
        NeuralNetwork netNoHidden({arrayLength, 1}, 0.1, 0.9);
        netNoHidden.printInfo();
        netNoHidden.printAllEquations();
        
        if (useMomentum) {
            netNoHidden.trainWithMomentum(trainInputs, trainTargets, valInputs, valTargets, 200, true);
        }
        
        // Test performance
        int correctNoHidden = 0;
        for (size_t i = 0; i < testInputs.size(); i++) {
            double prediction = netNoHidden.forward(testInputs[i])[0];
            double target = testTargets[i][0];
            if ((prediction > 0.5 && target > 0.5) || (prediction <= 0.5 && target <= 0.5)) {
                correctNoHidden++;
            }
        }
        double accuracyNoHidden = static_cast<double>(correctNoHidden) / testInputs.size();
        
        std::cout << "\nTest Accuracy (no hidden layer): " 
                  << std::fixed << std::setprecision(2) << accuracyNoHidden * 100 << "%\n";
        
        // Experiment 2: Network dengan 2 hidden units (seperti di paper)
        std::cout << "\n" << std::string(60, '-') << "\n";
        std::cout << "EXPERIMENT 2: Network with 2 hidden units (as in paper)\n";
        std::cout << "Architecture: " << arrayLength << " -> 2 -> 1\n";
        std::cout << "The paper states: \"The learning procedure discovered an elegant\n";
        std::cout << "solution using just two intermediate units, as shown in Fig. 1.\"\n";
        
        NeuralNetwork net2Hidden({arrayLength, 2, 1}, 0.1, 0.9);
        net2Hidden.printInfo();
        
        if (useMomentum) {
            net2Hidden.trainWithMomentum(trainInputs, trainTargets, valInputs, valTargets, 200, true);
        }
        
        // Test performance
        int correct2Hidden = 0;
        std::vector<std::vector<double>> hiddenActivations;
        
        for (size_t i = 0; i < testInputs.size(); i++) {
            double prediction = net2Hidden.forward(testInputs[i])[0];
            double target = testTargets[i][0];
            if ((prediction > 0.5 && target > 0.5) || (prediction <= 0.5 && target <= 0.5)) {
                correct2Hidden++;
            }
        }
        double accuracy2Hidden = static_cast<double>(correct2Hidden) / testInputs.size();
        
        std::cout << "\nTest Accuracy (2 hidden units): " 
                  << std::fixed << std::setprecision(2) << accuracy2Hidden * 100 << "%\n";
        
        // Experiment 3: Network dengan lebih banyak hidden units
        std::cout << "\n" << std::string(60, '-') << "\n";
        std::cout << "EXPERIMENT 3: Network with more hidden units\n";
        std::cout << "Architecture: " << arrayLength << " -> 4 -> 2 -> 1\n";
        
        NeuralNetwork netMoreHidden({arrayLength, 4, 2, 1}, 0.1, 0.9);
        netMoreHidden.printInfo();
        
        if (useMomentum) {
            netMoreHidden.trainWithMomentum(trainInputs, trainTargets, valInputs, valTargets, 200, true);
        }
        
        // Test performance
        int correctMoreHidden = 0;
        for (size_t i = 0; i < testInputs.size(); i++) {
            double prediction = netMoreHidden.forward(testInputs[i])[0];
            double target = testTargets[i][0];
            if ((prediction > 0.5 && target > 0.5) || (prediction <= 0.5 && target <= 0.5)) {
                correctMoreHidden++;
            }
        }
        double accuracyMoreHidden = static_cast<double>(correctMoreHidden) / testInputs.size();
        
        std::cout << "\nTest Accuracy (4->2 hidden units): " 
                  << std::fixed << std::setprecision(2) << accuracyMoreHidden * 100 << "%\n";
        
        // Summary
        std::cout << "\n" << std::string(60, '=') << "\n";
        std::cout << "EXPERIMENT SUMMARY\n";
        std::cout << std::string(60, '=') << "\n";
        std::cout << "Network Architecture          Test Accuracy\n";
        std::cout << std::string(40, '-') << "\n";
        std::cout << arrayLength << " -> 1" 
                  << std::setw(20) << std::fixed << std::setprecision(2) 
                  << accuracyNoHidden * 100 << "%\n";
        std::cout << arrayLength << " -> 2 -> 1" 
                  << std::setw(17) << accuracy2Hidden * 100 << "%\n";
        std::cout << arrayLength << " -> 4 -> 2 -> 1" 
                  << std::setw(13) << accuracyMoreHidden * 100 << "%\n";
        
        std::cout << "\nConclusion:\n";
        std::cout << "As stated in the paper, symmetry detection requires intermediate\n";
        std::cout << "(hidden) layers. The network with just input-output connections\n";
        std::cout << "cannot learn this task effectively.\n";
        
        // Print weight velocities untuk melihat efek momentum
        if (useMomentum) {
            std::cout << "\nMomentum information for the 2-hidden-unit network:\n";
            net2Hidden.printVelocityInfo();
        }
    }
};

// Main function
int main() {
    try {
        std::cout << "==================================================\n";
        std::cout << "COMPLETE NEURAL NETWORK IMPLEMENTATION\n";
        std::cout << "Based on: \"Learning representations by back-propagating errors\"\n";
        std::cout << "Rumelhart, Hinton, & Williams (1986)\n";
        std::cout << "==================================================\n\n";
        
        // Bagian 1: Demonstrasi momentum
        std::cout << "\n" << std::string(60, '=') << "\n";
        std::cout << "PART 1: MOMENTUM DEMONSTRATION\n";
        std::cout << std::string(60, '=') << "\n\n";
        
        // Quote dari paper tentang momentum
        std::cout << "From the paper:\n";
        std::cout << "\"This method does not converge as rapidly as methods which make\n";
        std::cout << "use of the second derivatives, but it is much simpler and can\n";
        std::cout << "easily be implemented by local computations in parallel hardware.\n";
        std::cout << "It can be significantly improved, without sacrificing the simplicity\n";
        std::cout << "and locality, by using an acceleration method...\"\n\n";
        
        std::cout << "Equation (9): Δw(t) = -ε ∂E/∂w(t) + α Δw(t-1)\n";
        std::cout << "where t is incremented by 1 for each sweep through the whole set\n";
        std::cout << "of input-output cases, and α is an exponential decay factor\n";
        std::cout << "between 0 and 1.\n\n";
        
        // Contoh sederhana dengan XOR untuk menunjukkan momentum
        std::cout << "Simple XOR example with momentum:\n";
        
        std::vector<std::vector<double>> xorInputs = {
            {0, 0}, {0, 1}, {1, 0}, {1, 1}
        };
        std::vector<std::vector<double>> xorTargets = {
            {0}, {1}, {1}, {0}
        };
        
        NeuralNetwork xorNetWithMomentum({2, 4, 1}, 0.5, 0.9);
        xorNetWithMomentum.printInfo();
        
        // Training dengan momentum
        xorNetWithMomentum.trainWithMomentum(xorInputs, xorTargets, 
                                            xorInputs, xorTargets,  // Menggunakan data yang sama untuk validation
                                            500, false);
        
        // Test
        std::cout << "\nXOR Results with Momentum:\n";
        for (int i = 0; i < 4; i++) {
            double prediction = xorNetWithMomentum.forward(xorInputs[i])[0];
            std::cout << xorInputs[i][0] << " XOR " << xorInputs[i][1] << " = " 
                      << (prediction > 0.5 ? 1 : 0) 
                      << " (prediction: " << std::fixed << std::setprecision(4) 
                      << prediction << ")\n";
        }
        
        // Bagian 2: Symmetry detection experiment
        std::cout << "\n" << std::string(60, '=') << "\n";
        std::cout << "PART 2: SYMMETRY DETECTION EXPERIMENT\n";
        std::cout << std::string(60, '=') << "\n\n";
        
        // Quote tentang symmetry detection
        std::cout << "From the paper about symmetry detection:\n";
        std::cout << "\"One simple task that cannot be done by just connecting the\n";
        std::cout << "input units to the output units is the detection of symmetry.\n";
        std::cout << "To detect whether the binary activity levels of a one-dimensional\n";
        std::cout << "array of input units are symmetrical about the centre point,\n";
        std::cout << "it is essential to use an intermediate layer because the activity\n";
        std::cout << "in an individual input unit, considered alone, provides no evidence\n";
        std::cout << "about the symmetry or non-symmetry of the whole input vector...\"\n\n";
        
        // Jalankan experiment symmetry detection
        SymmetryExperiment experiment(6);  // Array length 6 (seperti di paper)
        experiment.runExperiment(true);    // Gunakan momentum
        
        // Bagian 3: Variasi architecture untuk symmetry detection
        std::cout << "\n" << std::string(60, '=') << "\n";
        std::cout << "PART 3: ARCHITECTURE VARIATIONS FOR SYMMETRY\n";
        std::cout << std::string(60, '=') << "\n\n";
        
        // Coba berbagai panjang array
        std::vector<int> arrayLengths = {4, 6, 8};
        for (int length : arrayLengths) {
            std::cout << "\nTesting with array length = " << length << "\n";
            SymmetryExperiment exp(length);
            exp.runExperiment(true);
        }
        
        std::cout << "\n==================================================\n";
        std::cout << "PROGRAM COMPLETE\n";
        std::cout << "All equations from the paper have been implemented:\n";
        std::cout << "- Forward propagation (equations 1-2)\n";
        std::cout << "- Error calculation (equation 3)\n";
        std::cout << "- Backpropagation (equations 4-7)\n";
        std::cout << "- Simple gradient descent (equation 8)\n";
        std::cout << "- Momentum (equation 9)\n";
        std::cout << "- Symmetry detection example\n";
        std::cout << "==================================================\n";
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
