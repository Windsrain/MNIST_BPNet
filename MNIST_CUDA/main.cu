#include <vector>
#include <algorithm>
#include "mnist_reader.cuh"
#include "matrix.cuh"

int trainLen = 60000;
int testLen = 10000;
int cnt;
const int batch = 200;
double epsilon = 0.001;
double regLambda = 0.00;
int numExamples, nbsPerEpoch, inputDim, nnHdim[4] = {0, 256, 512, 10};

Matrix *Xb, *Yb, *Xt, *Yt, *Y_pred, *W[3], *B[3], *a[3], *delta[3], *dW[3], *dB[3];
Matrix *softMaxSum;
Matrix X_train, Y_train, X_test, Y_test;

void generate();
void predict();
void forwardPropagation();
void backPropagation();
void trainModel(int numPasses, bool printLoss);

int main() {
    cout.precision(16);
    unordered_map<string, Matrix> dataMap = readData();
    X_train = dataMap["trainImages"];
    Y_train = dataMap["trainLabels"];
    X_test = dataMap["testImages"];
    Y_test = dataMap["testLabels"];
    printf("(%d, %d) (%d, %d)\n", X_train.height, X_train.width, Y_train.height, Y_train.width);
    printf("(%d, %d) (%d, %d)\n", X_test.height, X_test.width, Y_test.height, Y_test.width);
    numExamples = X_train.height;
    inputDim = X_train.width;
    nbsPerEpoch = (int)(numExamples / batch);
    generate();
    trainModel(50000, true);

    return 0;
}

void generate() {
    printf("input dim: %d\n", inputDim);
    nnHdim[0] = inputDim;
    cudaMallocManaged((void **)&(Xb), sizeof(Matrix));
    Xb->height = batch; Xb->width = inputDim;
    cudaMallocManaged((void **)&(Yb), sizeof(Matrix));
    Yb->height = batch; Yb->width = 10;
    cudaMallocManaged((void **)&(Xt), sizeof(Matrix));
    Xt->height = batch; Xt->width = inputDim;
    cudaMallocManaged((void **)&(Yt), sizeof(Matrix));
    Yt->height = batch; Yt->width = 10;
    cudaMallocManaged((void **)&(Y_pred), sizeof(Matrix));
    Y_pred->height = batch; Y_pred->width = 10;
    cudaMallocManaged((void **)&(Xb->elements), batch * inputDim * sizeof(double));
    cudaMallocManaged((void **)&(Yb->elements), batch * 10 * sizeof(double));
    cudaMallocManaged((void **)&(Xt->elements), batch * inputDim * sizeof(double));
    cudaMallocManaged((void **)&(Yt->elements), batch * 10 * sizeof(double));
    cudaMallocManaged((void **)&(Y_pred->elements), batch * 10 * sizeof(double));
    for (int i = 0; i < 3; i++) {
        int row = nnHdim[i], col = nnHdim[i + 1];
        double std = sqrt(col);
        cudaMallocManaged((void **)&(W[i]), sizeof(Matrix));
        W[i]->width = col; W[i]->height = row;
        cudaMallocManaged((void **)&(B[i]), sizeof(Matrix));
        B[i]->width = col; B[i]->height = 1;
        cudaMallocManaged((void **)&(a[i]), sizeof(Matrix));
        a[i]->width = col; a[i]->height = batch;
        cudaMallocManaged((void **)&(delta[i]), sizeof(Matrix));
        delta[i]->width = col; delta[i]->height = batch;
        cudaMallocManaged((void **)&(dW[i]), sizeof(Matrix));
        dW[i]->width = col; dW[i]->height = row;
        cudaMallocManaged((void **)&(dB[i]), sizeof(Matrix));
        dB[i]->width = col; dB[i]->height = 1;
        cudaMallocManaged((void **)&(W[i]->elements), row * col * sizeof(double));
        cudaMallocManaged((void **)&(B[i]->elements), 1 * col * sizeof(double));
        cudaMallocManaged((void **)&(a[i]->elements), batch * col * sizeof(double));
        cudaMallocManaged((void **)&(delta[i]->elements), batch * col * sizeof(double));
        cudaMallocManaged((void **)&(dW[i]->elements), row * col * sizeof(double));
        cudaMallocManaged((void **)&(dB[i]->elements), row * col * sizeof(double));
        printf("fc: %d -> %d\n", row, col);
        initialize(W[i], std);
        initialize(B[i], 0);
    }
    cudaMallocManaged((void **)&(softMaxSum), sizeof(Matrix));
    softMaxSum->width = 1; softMaxSum->height = batch;
    cudaMallocManaged((void **)&(softMaxSum->elements), batch * 1 * sizeof(double));
    cudaDeviceSynchronize();
}

void predict() {
    dim3 blockSize(32, 32);
	dim3 gridSize(32, 32);   
    matDotKernel <<<gridSize, blockSize>>> (Xt, W[0], a[0]);
    matPlusKernel <<<gridSize, blockSize>>> (a[0], B[0], a[0]);
    matTanhKernel <<<gridSize, blockSize>>> (a[0]);
    matDotKernel <<<gridSize, blockSize>>> (a[0], W[1], a[1]);
    matPlusKernel <<<gridSize, blockSize>>> (a[1], B[1], a[1]);
    matTanhKernel <<<gridSize, blockSize>>> (a[1]);
    matDotKernel <<<gridSize, blockSize>>> (a[1], W[2], a[2]);
    matPlusKernel <<<gridSize, blockSize>>> (a[2], B[2], a[2]);
    cudaDeviceSynchronize();
    for (int i = 0; i < a[2]->height; i++) {
        int maxIndex = 0; 
        double maxValue = a[2]->elements[i * a[2]->width];
        for (int j = 0; j < a[2]->width; j++)
            if (a[2]->elements[i * a[2]->width + j] > maxValue) {
                maxIndex = j;
                maxValue = a[2]->elements[i * a[2]->width + j];
            }
        if (Yt->elements[i * a[2]->width + maxIndex])
            cnt++;
    }
    cudaDeviceSynchronize();
}

void forwardPropagation() {
    dim3 blockSize(32, 32);
	dim3 gridSize(32, 32);        
    matDotKernel <<<gridSize, blockSize>>> (Xb, W[0], a[0]);
    matPlusKernel <<<gridSize, blockSize>>> (a[0], B[0], a[0]);
    matTanhKernel <<<gridSize, blockSize>>> (a[0]);
    matDotKernel <<<gridSize, blockSize>>> (a[0], W[1], a[1]);
    matPlusKernel <<<gridSize, blockSize>>> (a[1], B[1], a[1]);
    matTanhKernel <<<gridSize, blockSize>>> (a[1]);
    matDotKernel <<<gridSize, blockSize>>> (a[1], W[2], a[2]);
    matPlusKernel <<<gridSize, blockSize>>> (a[2], B[2], a[2]);
    matExpKernel <<<gridSize, blockSize>>> (a[2]);
    matSumKernel <<<gridSize, blockSize>>> (a[2], softMaxSum, 1);
    matDivKernel <<<gridSize, blockSize>>> (a[2], softMaxSum);
    cudaDeviceSynchronize();
}

void backPropagation() {
    dim3 blockSize(32, 32);
    dim3 gridSize(32, 32);
    matSubKernel <<<gridSize, blockSize>>> (a[2], Yb, delta[2]);
    cudaDeviceSynchronize();

    matDotKernel <<<gridSize, blockSize>>> (a[1], delta[2], dW[2], true);   
    matSumKernel <<<gridSize, blockSize>>> (delta[2], dB[2], 0);
    cudaDeviceSynchronize();

    matPowKernel <<<gridSize, blockSize>>> (a[1], 2);
    matSubKernel <<<gridSize, blockSize>>> (1, a[1], a[1]);
    matDotKernel <<<gridSize, blockSize>>> (delta[2], W[2], delta[1], false, true);      
    matMulKernel <<<gridSize, blockSize>>> (delta[1], a[1], delta[1]);
    matDotKernel <<<gridSize, blockSize>>> (a[0], delta[1], dW[1], true);
    matSumKernel <<<gridSize, blockSize>>> (delta[1], dB[1], 0);
    cudaDeviceSynchronize();

    matPowKernel <<<gridSize, blockSize>>> (a[0], 2);
    matSubKernel <<<gridSize, blockSize>>> (1, a[0], a[0]);
    matDotKernel <<<gridSize, blockSize>>> (delta[1], W[1], delta[0], false, true);
    matMulKernel <<<gridSize, blockSize>>> (delta[0], a[0], delta[0]);
    matDotKernel <<<gridSize, blockSize>>> (Xb, delta[0], dW[0], true);
    matSumKernel <<<gridSize, blockSize>>> (delta[0], dB[0], 0);
    cudaDeviceSynchronize();

    for (int i = 0; i < 3; i++) {
        matMulKernel <<<gridSize, blockSize>>> (dW[i], epsilon);
        matMulKernel <<<gridSize, blockSize>>> (dB[i], epsilon);
        matSubKernel <<<gridSize, blockSize>>> (W[i], dW[i], W[i]);
        matSubKernel <<<gridSize, blockSize>>> (B[i], dB[i], B[i]);
        cudaDeviceSynchronize();
    }
}

void trainModel(int numPasses, bool printLoss) {
    int i;
    for (i = 0; i <= numPasses; i++) {
        int j = i % nbsPerEpoch;
        if (j == 0) {
            vector<int> ridx(numExamples);
            int k;
            for (k = 0; k < numExamples; k++)
                ridx[k] = k;
            random_shuffle(ridx.begin(), ridx.end());
            X_train.shuffle(ridx);
            Y_train.shuffle(ridx);
        }
        dataCopy(Xb, X_train, j * batch, (j + 1) * batch);
        dataCopy(Yb, Y_train, j * batch, (j + 1) * batch, true);
        forwardPropagation();
        backPropagation();
        if (printLoss && (i % 100 == 0)) {
            epsilon *= 0.99;
            cnt = 0;
            for (int k = 0; k < (int)(X_test.height / batch); k++) {
                dataCopy(Xt, X_test, k * batch, (k + 1) * batch);
                dataCopy(Yt, Y_test, k * batch, (k + 1) * batch, true);   
                predict();     
                cudaDeviceSynchronize();
            }
            double accuracy = (cnt * 1.0 / X_test.height);
            struct tm *p;
            time_t t = time(0);
            p = localtime(&t);
            printf("%02d:%02d:%02d testing accuracy after iteration %d: %.2lf%%\n", p->tm_hour, p->tm_min, p->tm_sec, i, accuracy * 100);
        }
    }
}