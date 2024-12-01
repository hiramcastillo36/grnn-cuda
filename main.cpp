#include "grnn.h"
#include <vector>
#include <fstream>
#include <sstream>
#include <iostream>
#include <string>
#include <algorithm>

struct TimeSeriesData {
    std::vector<float> time;
    std::vector<float> values;
};

TimeSeriesData loadCSV(const std::string& filename) {
    TimeSeriesData data;
    std::ifstream file(filename);
    std::string line;

    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string value;


        std::getline(ss, value, ',');
        data.time.push_back(std::stof(value));

        std::getline(ss, value, ',');
        data.values.push_back(std::stof(value));
        std::getline(ss, value, ',');
        data.values.push_back(std::stof(value));
    }

    return data;
}

void normalizeData(std::vector<float>& data, float& min_val, float& max_val) {
    min_val = *std::min_element(data.begin(), data.end());
    max_val = *std::max_element(data.begin(), data.end());

    for (auto& value : data) {
        value = (value - min_val) / (max_val - min_val);
    }
}

void denormalizeData(std::vector<float>& predictions, float min_val, float max_val) {
    for (auto& value : predictions) {
        value = value * (max_val - min_val) + min_val;
    }
}

int main() {
    GRNNParams params;
    params.n_features = 2;
    params.sigma = 0.1f;

    std::cout << "Cargando datos de entrenamiento..." << std::endl;
    auto train_data = loadCSV("DS-5-1-GAP-1-1-N-1_v2.csv");
    params.n_train = train_data.time.size();

    std::cout << "Cargando datos de prueba..." << std::endl;
    auto test_data = loadCSV("DS-5-1-GAP-5-1-N-3_v2.csv");
    int n_test = test_data.time.size();

    std::vector<float> predictions(n_test * 2);


    std::cout << "Realizando predicciones..." << std::endl;
    predictGRNN(
        train_data.values.data(),
        &train_data.values[1],
        test_data.values.data(),
        predictions.data(),
        params,
        n_test
    );


    std::ofstream output("resultados.csv");
    output << "tiempo,curva1,curva2,prediccion,error\n";

    float mse = 0.0f;
    for (int i = 0; i < n_test; i++) {
        float error = std::abs(predictions[i] - test_data.values[i*2 + 1]);
        mse += error * error;

        output << test_data.time[i] << ","
               << test_data.values[i*2] << ","
               << test_data.values[i*2 + 1] << ","
               << predictions[i] << ","
               << error << "\n";
    }
    mse /= n_test;

    std::cout << "\nResultados:" << std::endl;
    std::cout << "------------------------" << std::endl;
    std::cout << "Muestras de entrenamiento: " << params.n_train << std::endl;
    std::cout << "Muestras de prueba: " << n_test << std::endl;
    std::cout << "Error cuadrático medio: " << mse << std::endl;
    std::cout << "------------------------" << std::endl;
    std::cout << "Resultados guardados en 'resultados.csv'" << std::endl;

    return 0;
}
