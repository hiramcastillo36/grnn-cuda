#pragma once

struct GRNNParams {
    float sigma;
    int n_train;
    int n_features;
};

void predictGRNN(
    const float* X_train,
    const float* y_train,
    const float* X_test,
    float* output,
    const GRNNParams params,
    const int n_test
);
