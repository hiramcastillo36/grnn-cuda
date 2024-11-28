#include "grnn.h"
#include <vector>
#include <random>
#include <iostream>

void printPredictions(const float* predictions, int n) {
    for(int i = 0; i < n; i++) {
        std::cout << "Entrada: [" << (i & 2 ? 1 : 0) << ", " << (i & 1 ? 1 : 0)
                  << "] -> Predicción: " << predictions[i]
                  << " (Esperado: " << ((i & 2 && i & 1) ? 1 : 0) << ")" << std::endl;
    }
}

int main() {
    // Parámetros de la red
    GRNNParams params;
    params.sigma = 0.1f;        // Reducimos sigma para mayor sensibilidad
    params.n_features = 2;      // Solo necesitamos 2 features para OR
    params.n_train = 4;         // Las 4 combinaciones posibles de OR

    // Datos de entrenamiento para OR
    std::vector<float> X_train = {
        0, 0,  // [0,0]
        0, 1,  // [0,1]
        1, 0,  // [1,0]
        1, 1   // [1,1]
    };

    // Resultados esperados para OR
    std::vector<float> y_train = {
        0,  // 0 OR 0 = 0
        1,  // 0 OR 1 = 1
        1,  // 1 OR 0 = 1
        0   // 1 OR 1 = 1
    };

    // Datos de prueba (probamos todas las combinaciones)
    int n_test = 4;
    std::vector<float> X_test = {
        0, 0,
        0, 1,
        1, 0,
        1, 1
    };

    // Vector para las predicciones
    std::vector<float> predictions(n_test);

    // Realizar predicciones
    predictGRNN(
        X_train.data(),
        y_train.data(),
        X_test.data(),
        predictions.data(),
        params,
        n_test
    );

    // Imprimir resultados
    std::cout << "Resultados del OR lógico:" << std::endl;
    std::cout << "------------------------" << std::endl;
    printPredictions(predictions.data(), n_test);

    return 0;
}
