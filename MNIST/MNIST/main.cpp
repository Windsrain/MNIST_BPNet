#define _CRT_SECURE_NO_DEPRECATE
#include <iostream>
#include <string>
#include <utility>
#include <algorithm>
#include <vector>
#include <string>
#include <map>
#include <cstdlib>
#include <ctime>
#include "mnist_reader_less.hpp"
#include "matrix.hpp"

using namespace std;

string path;

Matrix predict(map<string, Matrix> &model, Matrix X);
map<string, Matrix> build_model(Matrix &X, Matrix &y, Matrix &X_t, Matrix &y_t, vector<int> nn_hdim, double epsilon, double reg_lambda, int num_passes, bool print_loss);

int main(int argc, char* argv[]) {
	// MNIST_DATA_LOCATION set by MNIST cmake config
	cout.precision(16);
	cout << "Please input MNIST data directory: ";
	cin >> path;

	// Load MNIST data
	auto dataset = mnist::read_dataset(path);
	Matrix X_train = Matrix(dataset.training_images);
	Matrix Y_train = Matrix(dataset.training_labels);
	Matrix X_test = Matrix(dataset.test_images);
	Matrix Y_test = Matrix(dataset.test_labels);
	X_train.binarize_each();
	X_test.binarize_each();

	printf("(%d, %d) (%d, %d)\n", X_train.shape().first, X_train.shape().second, Y_train.shape().first, Y_train.shape().second);
	printf("(%d, %d) (%d, %d)\n", X_test.shape().first, X_test.shape().second, Y_test.shape().first, Y_test.shape().second);

	int input_dim = X_train.shape().second;
	vector<int> nn_hdim = { input_dim, 256, 512, 10 };
	double epsilon = 0.001, reg_lambda = 0.00;

	build_model(X_train, Y_train, X_test, Y_test, nn_hdim, epsilon, reg_lambda, 50000, true);

	return 0;
}

Matrix predict(map<string, Matrix> &model, Matrix X) {
	auto& W1 = model["W1"]; auto& W2 = model["W2"]; auto& W3 = model["W3"];
	auto& b1 = model["b1"]; auto& b2 = model["b2"]; auto& b3 = model["b3"];
	Matrix z1 = X.dot(W1) + b1;
	Matrix a1 = z1.Tanh();
	Matrix z2 = a1.dot(W2) + b2;
	Matrix a2 = z2.Tanh();
	Matrix z3 = a2.dot(W3) + b3;
	vector<int> probs;

	int i;
	for (i = 0; i < z3.shape().first; i++) {
		probs.push_back(max_element((z3.getMatrix()[i]).begin(), (z3.getMatrix()[i]).end()) - (z3.getMatrix()[i]).begin());
	}

	return Matrix(probs);
}

map<string, Matrix> build_model(Matrix &X, Matrix &y, Matrix &X_t, Matrix &y_t, vector<int> nn_hdim, double epsilon, double reg_lambda, int num_passes, bool print_loss) {

	// Initialize the parameters to random values.
	map<string, Matrix> model;
	int num_examples = X.shape().first;
	int nn_input_dim = nn_hdim[0];
	cout << "input dim: " << nn_input_dim << endl;

	int hdim1 = nn_hdim[1];
	double std1 = sqrt(2.0 / hdim1);
	Matrix W1 = Matrix(nn_input_dim, hdim1);
	W1.initialize_each(std1);
	Matrix b1 = Matrix(1, hdim1);
	printf("fc: %d -> %d\n", nn_input_dim, hdim1);

	int hdim2 = nn_hdim[2];
	double std2 = sqrt(2.0 / hdim2);
	Matrix W2 = Matrix(hdim1, hdim2);
	W2.initialize_each(std2);
	Matrix b2 = Matrix(1, hdim2);
	printf("fc: %d -> %d\n", hdim1, hdim2);

	int hdim3 = nn_hdim[3];
	double std3 = sqrt(2.0 / hdim3);
	Matrix W3 = Matrix(hdim2, hdim3);
	W3.initialize_each(std3);
	Matrix b3 = Matrix(1, hdim3);
	printf("fc: %d -> %d\n", hdim2, hdim3);

	int bs = 128;
	int nbs_per_epoch = int(num_examples / bs);

	// Gradient descent. For each batch..
	int i;
	for (i = 0; i <= num_passes; i++) {
		int j = i % nbs_per_epoch;
		if (j == 0) {
			vector<int> ridx(num_examples);
			int k;
			for (k = 0; k < num_examples; k++)
				ridx[k] = k;
			random_shuffle(ridx.begin(), ridx.end());
			X.shuffle(ridx);
			y.shuffle(ridx);
		}
		Matrix Xb = X.section(j * bs, (j + 1) * bs);
		Matrix yb = y.section(j * bs, (j + 1) * bs);

		// Forward propagation
		Matrix z1 = Xb.dot(W1) + b1;
		Matrix a1 = z1.Tanh();
		Matrix z2 = a1.dot(W2) + b2;
		Matrix a2 = z2.Tanh();
		Matrix z3 = a2.dot(W3) + b3;
		Matrix exp_scores = z3.Exp();
		Matrix probs = exp_scores / exp_scores.sum(1);

		// Back propagation
		Matrix delta_loss = probs - yb.expand_01();

		Matrix dW3 = (a2.T()).dot(delta_loss);
		Matrix db3 = delta_loss.sum(0);

		Matrix all1Matrix1 = Matrix(a2.shape().first, a2.shape().second, 1);
		Matrix delta3 = delta_loss.dot((W3.T())) * (all1Matrix1 - a2.power(2));
		Matrix dW2 = (a1.T()).dot(delta3);
		Matrix db2 = delta3.sum(0);

		Matrix all1Matrix2 = Matrix(a1.shape().first, a1.shape().second, 1);
		Matrix delta2 = delta3.dot((W2.T())) * (all1Matrix2 - a1.power(2));
		Matrix dW1 = (Xb.T()).dot(delta2);
		Matrix db1 = delta2.sum(0);

		// Gradient descent parameter update
		W1 = W1 - (dW1 * epsilon);
		W2 = W2 - (dW2 * epsilon);
		W3 = W3 - (dW3 * epsilon);
		b1 = b1 - (db1 * epsilon);
		b2 = b2 - (db2 * epsilon);
		b3 = b3 - (db3 * epsilon);

		// Optionally print the loss.
		if (print_loss && (i % 100 == 0)) {
			epsilon *= 0.99;
			// Assign new parameters to the model
			model["W1"] = W1; model["W2"] = W2; model["W3"] = W3;
			model["b1"] = b1; model["b2"] = b2; model["b3"] = b3;
			Matrix y_pred = predict(model, X_t);
			double accuracy = (y_pred - y_t).count(0) * 1.0 / (1.0 * y_t.shape().first);
			struct tm *p;
			time_t t = time(0);
			p = localtime(&t);
			printf("%02d:%02d:%02d testing accuracy after iteration %d: %.2lf%%\n", p->tm_hour, p->tm_min, p->tm_sec, i, accuracy * 100);
		}
	}

	return model;
}