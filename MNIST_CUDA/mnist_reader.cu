#include "mnist_reader.cuh"

inline int reverseInt(int value) {
    return (value << 24) | ((value << 8) & 0x00FF0000) | ((value >> 8) & 0X0000FF00) | (value >> 24);
}

void readLabels(string fileName, Matrix &labels, int len) {
    ifstream file(fileName, ios::binary);
    if (!file.is_open())
        cout << "Error Opening File!\n" << endl;
    int magic, count;
    file.read((char *)&magic, sizeof(magic));
    file.read((char *)&count, sizeof(count));
    magic = reverseInt(magic);
    count = reverseInt(count);
    for (int i = 0; i < len; i++) {
        unsigned char label;
        file.read((char *)&label, sizeof(label));
        labels.elements[i] = (double)label;
    }
    file.close();
}

void readImages(string fileName, Matrix &images, int len) {
    ifstream file(fileName, ios::binary);
    if (!file.is_open())
        cout << "Error Opening File!\n" << endl;
    int magic, count, row, col;
    file.read((char *)&magic, sizeof(magic));
    file.read((char *)&count, sizeof(count));
    file.read((char *)&row, sizeof(row));
    file.read((char *)&col, sizeof(col));
    magic = reverseInt(magic);
    count = reverseInt(count);
    row = reverseInt(row);
    col = reverseInt(col);
    for (int i = 0; i < len; i++)
        for (int j = 0; j < row; j++)
            for (int k = 0; k < col; k++) {
                unsigned char image = 0;
                file.read((char *)&image, sizeof(image));
                // Binarilization
                if (image > 0)
                    images.elements[i * row * col + j * col + k] = 1;
                else
                    images.elements[i * row * col + j * col + k] = 0;
            }
    file.close();
}

unordered_map<string, Matrix> readData() {
    extern int trainLen;
    extern int testLen;

    Matrix trainImages = Matrix(trainLen, 28 * 28);
    Matrix trainLabels = Matrix(trainLen, 1);
    Matrix testImages = Matrix(testLen, 28 * 28);
    Matrix testLabels = Matrix(testLen, 1);

    readImages("data\\train-images.idx3-ubyte", trainImages, trainLen);
    readImages("data\\t10k-images.idx3-ubyte", testImages, testLen);
    readLabels("data\\train-labels.idx1-ubyte", trainLabels,trainLen);
    readLabels("data\\t10k-labels.idx1-ubyte", testLabels,testLen);

    unordered_map<string, Matrix> dataMap;

    dataMap.insert({"trainImages", trainImages});
    dataMap.insert({"trainLabels", trainLabels});
    dataMap.insert({"testImages", testImages});
    dataMap.insert({"testLabels", testLabels});

    return dataMap;
}