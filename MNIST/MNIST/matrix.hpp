#ifndef MATRIX_HPP
#define MATRIX_HPP
#define Pixel uint8_t
#define Label uint8_t

#include <iostream>
#include <vector>
#include <random>
#include <cstdint>
#include <utility>
#include <numeric>
#include <cmath>
#include <cstdlib>

class Matrix {
private:
	std::vector<std::vector<double>> matrix;
public:
	Matrix() {};

	Matrix(int row, int line) {
		std::vector<double> temp(line, 0);
		matrix.resize(row, temp);
	}

	Matrix(int row, int line, double value) {
		std::vector<double> temp(line, value);
		matrix.resize(row, temp);
	}

	Matrix(std::vector<std::vector<Pixel>> images) {
		int i, j;
		matrix.resize(images.size());
		for (i = 0; i < images.size(); i++)
			for (j = 0; j < images[i].size(); j++)
				matrix[i].push_back(images[i][j]);
	}

	Matrix(std::vector<std::vector<double>> images) {
		int i, j;
		matrix.resize(images.size());
		for (i = 0; i < images.size(); i++)
			for (j = 0; j < images[i].size(); j++)
				matrix[i].push_back(images[i][j]);
	}

	Matrix(std::vector<Label> labels) {
		int i;
		matrix.resize(labels.size());
		for (i = 0; i < labels.size(); i++)
			matrix[i].push_back(labels[i]);
	}

	Matrix(std::vector<int> labels) {
		int i;
		matrix.resize(labels.size());
		for (i = 0; i < labels.size(); i++)
			matrix[i].push_back(labels[i]);
	}

	// matrix print
	void Print(int row, int line) {
		int i, j;
		for (i = 0; i < row; i++) {
			for (j = 0; j < line; j++)
				std::cout << matrix[i][j] << " ";
			if (j == line)
				std::cout << std::endl;
		}
	}

	// matrix
	std::vector<std::vector<double>>& getMatrix() {
		return matrix;
	}

	// matrix shape
	std::pair<int, int> shape() {
		return std::make_pair(matrix.size(), matrix[0].size());
	}

	// matrix centralization
	void centralize_each(double m, double s) {
		for (auto& vec : matrix)
			for (auto& v : vec) {
				v -= m;
				v /= s;
			}
	}

	// matrix initialization
	void initialize_each(double s) {
		int seed = 6;
		std::default_random_engine gen(seed);
		std::normal_distribution<double> dis(0, s);
		int i, j;
		for (i = 0; i < matrix.size(); i++)
			for (j = 0; j < matrix[0].size(); j++)
				matrix[i][j] = dis(gen);
	}

	// matrix binarization
	void binarize_each(double threshold = 0) {
		for (auto& vec : matrix)
			for (auto& v : vec)
				v = (v > threshold) ? 1.0 : 0.0;
	}

	// matrix transpose
	Matrix T() {
		Matrix res(matrix[0].size(), matrix.size());
		int i, j;
		for (i = 0; i < matrix.size(); i++)
			for (j = 0; j < matrix[0].size(); j++)
				res.matrix[j][i] = matrix[i][j];

		return res;
	}

	// matrix max pos
	int max_pos(int row) {
		int i, maxp = 0;
		double max = matrix[row][0];
		for (i = 0; i < matrix[0].size(); i++)
			if (matrix[row][i] > max) {
				max = matrix[row][i];
				maxp = i;
			}

		return maxp;
	}

	// sigmoid
	double sigmoid(double x) {
		return 1 / (1 + exp(-x));
	}

	// matrix tanh
	Matrix Tanh() {
		Matrix res(matrix.size(), matrix[0].size());
		int i, j;
		for (i = 0; i < matrix.size(); i++)
			for (j = 0; j < matrix[0].size(); j++)
				res.matrix[i][j] = 2 * sigmoid(2 * matrix[i][j]) - 1;

		return res;
	}

	// matrix exp
	Matrix Exp() {
		Matrix res(matrix.size(), matrix[0].size());
		int i, j;
		for (i = 0; i < matrix.size(); i++)
			for (j = 0; j < matrix[0].size(); j++)
				res.matrix[i][j] = exp(matrix[i][j]);

		return res;
	}

	// matrix power
	Matrix power(double k) {
		Matrix res(matrix.size(), matrix[0].size());
		int i, j;
		for (i = 0; i < matrix.size(); i++)
			for (j = 0; j < matrix[0].size(); j++)
				res.matrix[i][j] = pow(matrix[i][j], k);

		return res;
	}

	// matrix shuffle
	void shuffle(std::vector<int> ridx) {
		Matrix temp = Matrix(matrix.size(), matrix[0].size());
		int i;
		for (i = 0; i < matrix.size(); i++)
			temp.matrix[i] = matrix[ridx[i]];
		for (i = 0; i < matrix.size(); i++)
			matrix[i] = temp.matrix[i];
	}

	// matrix section
	Matrix section(int up, int down) {
		Matrix res(down - up, matrix[0].size());
		int i;
		for (i = up; i < down; i++)
			res.matrix[i - up] = matrix[i % matrix.size()];

		return res;
	}

	// matrix sum
	Matrix sum(int axis) {
		Matrix res;
		if (axis == 0) {
			Matrix res(1, matrix[0].size());
			int i, j;
			for (j = 0; j < matrix[0].size(); j++)
				for (i = 0; i < matrix.size(); i++)
					res.matrix[0][j] += matrix[i][j];

			return res;
		}
		if (axis == 1) {
			Matrix res(1, matrix.size());
			int i;
			for (i = 0; i < matrix.size(); i++)
				res.matrix[0][i] = accumulate(matrix[i].begin(), matrix[i].end(), (double)0.0);

			return res;
		}

		return res;
	}

	// matrix expand for 01, matrix[i][value] = 1, other is 0
	Matrix expand_01() {
		Matrix res(matrix.size(), 10);
		int i;
		for (i = 0; i < matrix.size(); i++) {
			int value = (int)matrix[i][0];
			res.matrix[i][value] = 1;
		}

		return res;
	}

	int count(double value) {
		int res = 0;
		int i, j;
		for (i = 0; i < matrix.size(); i++)
			for (j = 0; j < matrix[0].size(); j++)
				if (matrix[i][j] == value)
					res++;

		return res;
	}

	// matrix dot multiplication
	Matrix dot(const Matrix &t) {
		Matrix res(matrix.size(), t.matrix[0].size());
		int i, j, k;
		for (i = 0; i < matrix.size(); i++)
			for (j = 0; j < t.matrix[0].size(); j++)
				for (k = 0; k < matrix[0].size(); k++)
					res.matrix[i][j] += matrix[i][k] * t.matrix[k][j];

		return res;
	}

	// matrix addition
	Matrix operator+(const Matrix & b) {
		Matrix res(matrix.size(), matrix[0].size());
		int i, j;
		for (i = 0; i < matrix.size(); i++)
			for (j = 0; j < matrix[0].size(); j++)
				res.matrix[i][j] = matrix[i][j] + b.matrix[0][j];

		return res;
	}

	// matrix subtraction
	Matrix operator-(const Matrix & b) {
		Matrix res(matrix.size(), matrix[0].size());
		int i, j;
		for (i = 0; i < matrix.size(); i++)
			for (j = 0; j < matrix[0].size(); j++)
				res.matrix[i][j] = matrix[i][j] - b.matrix[i][j];

		return res;
	}

	// matrix multiplication
	Matrix operator*(const Matrix & t) {
		Matrix res(matrix.size(), t.matrix[0].size());
		int i, j;
		for (i = 0; i < matrix.size(); i++)
			for (j = 0; j < t.matrix[0].size(); j++)
					res.matrix[i][j] = matrix[i][j] * t.matrix[i][j];

		return res;
	}

	Matrix operator*(const double &t) {
		Matrix res(matrix.size(), matrix[0].size());
		int i, j;
		for (i = 0; i < matrix.size(); i++)
			for (j = 0; j < matrix[0].size(); j++)
				res.matrix[i][j] = matrix[i][j] * t;

		return res;
	}

	// matrix division
	Matrix operator/(const Matrix & t) {
		Matrix res(matrix.size(), matrix[0].size());
		int i, j;
		for (i = 0; i < matrix.size(); i++) {
			for (j = 0; j < matrix[0].size(); j++)
				res.matrix[i][j] = matrix[i][j] / t.matrix[0][i];
		}

		return res;
	}
};

#endif