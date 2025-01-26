#include <iostream>
#include <fstream>
#include <vector>
#include <stdexcept>
#include <algorithm>

std::vector<std::vector<uint8_t>> readMNISTImages(const std::string& filepath, int& numberOfImages, int& rows, int& cols) {
    std::ifstream file(filepath, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Unable to open file: " + filepath);
    }

    int magicNumber = 0;
    file.read(reinterpret_cast<char*>(&magicNumber), 4);
    magicNumber = __builtin_bswap32(magicNumber); // Convert from big-endian to little-endian

    if (magicNumber != 2051) {
        throw std::runtime_error("Invalid MNIST image file!");
    }

    file.read(reinterpret_cast<char*>(&numberOfImages), 4);
    file.read(reinterpret_cast<char*>(&rows), 4);
    file.read(reinterpret_cast<char*>(&cols), 4);

    numberOfImages = __builtin_bswap32(numberOfImages);
    rows = __builtin_bswap32(rows);
    cols = __builtin_bswap32(cols);

    std::vector<std::vector<uint8_t>> images(numberOfImages, std::vector<uint8_t>(rows * cols));
    for (int i = 0; i < numberOfImages; ++i) {
        file.read(reinterpret_cast<char*>(images[i].data()), rows * cols);
    }

    file.close();
    return images;
}

std::vector<uint8_t> readMNISTLabels(const std::string& filepath, int& numberOfLabels) {
    std::ifstream file(filepath, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Unable to open file: " + filepath);
    }

    int magicNumber = 0;
    file.read(reinterpret_cast<char*>(&magicNumber), 4);
    magicNumber = __builtin_bswap32(magicNumber);

    if (magicNumber != 2049) {
        throw std::runtime_error("Invalid MNIST label file!");
    }

    file.read(reinterpret_cast<char*>(&numberOfLabels), 4);
    numberOfLabels = __builtin_bswap32(numberOfLabels);

    std::vector<uint8_t> labels(numberOfLabels);
    file.read(reinterpret_cast<char*>(labels.data()), numberOfLabels);

    file.close();
    return labels;
}

// This function will read the .raw file and load the data into a vector
std::vector<uint8_t> convertImageToTestVector(const std::string& imagePath, int rows, int cols) {
    std::ifstream file(imagePath, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Unable to open image file: " + imagePath);
    }

    // Assuming the raw image is exactly 28x28 pixels (784 bytes)
    std::vector<uint8_t> imageVector(rows * cols);
    file.read(reinterpret_cast<char*>(imageVector.data()), rows * cols);

    file.close();
    return imageVector;
}

std::pair<int, std::pair<int, int>> K_Nearest_Neighbor(int K, int datasize, int rows, int cols, std::vector<std::vector<uint8_t>> &images, std::vector<uint8_t> &labels, std::vector<uint8_t> imageTest) {
    std::vector<std::pair<int, int>> Dists;
    int Pixels = rows * cols;
    for (int n_Image = 0; n_Image < datasize; ++n_Image) {
        int Dist = 0;
        for (int Pixel = 0; Pixel < Pixels; Pixel++) {
            int Euclidian_Dist = static_cast<int>(images[n_Image][Pixel]) - static_cast<int>(imageTest[Pixel]);
            Dist += Euclidian_Dist * Euclidian_Dist;
        }
        Dists.emplace_back(Dist, static_cast<int>(labels[n_Image]));
    }
    std::nth_element(Dists.begin(), Dists.begin() + K, Dists.end());
    Dists.resize(K);
    int possible_Answer[10] = {};
    for (int k = 0; k < K; ++k) {
        possible_Answer[Dists[k].second]++;
    }
    int guessed = -1, common = 0;
    for (int num = 0; num < 10; ++num) {
        if (possible_Answer[num] > common) {
            common = possible_Answer[num];
            guessed = num;
        }
    }
    int guessed2 = -1, common2 = 0;
    for (int num = 0; num < 10; ++num) {
        if (num == guessed) {
            continue;
        }
        if (possible_Answer[num] > common2) {
            common2 = possible_Answer[num];
            guessed2 = num;
        }
    }
    int guessed3 = -1, common3 = 0;
    for (int num = 0; num < 10; ++num) {
        if (num == guessed || num == guessed2) {
            continue;
        }
        if (possible_Answer[num] > common3) {
            common3 = possible_Answer[num];
            guessed3 = num;
        }
    }
    return {guessed, {guessed2, guessed3}};
}

int main() {
    int numberOfImages, numberOfTests, rows, cols;
    
    // Read MNIST images and labels
    auto images = readMNISTImages("train-images.idx3-ubyte", numberOfImages, rows, cols);
    auto labels = readMNISTLabels("train-labels.idx1-ubyte", numberOfImages);

    // Load the test image from a .raw file
    std::vector<uint8_t> imageTest = convertImageToTestVector("converted_28x28.raw", rows, cols);
    for (auto &e : imageTest) {
        e = 255 - e;
    }

    std::cout << "Loaded " << numberOfImages << " images of size " << rows << "x" << cols << "\n";

    // Display the test image (optional visualization)
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            std::cout << int(imageTest[r * cols + c]) << " ";
        }
        std::cout << "\n";
    }

    // Perform K-Nearest Neighbor on the test image
    for (int i = 1; i <= 6; i++) {
        std::pair<int, std::pair<int, int>> Guessed_Label = K_Nearest_Neighbor(i, 50000, rows, cols, images, labels, imageTest);
        std::cout << "GUESSED 1st : " << (Guessed_Label.first == -1 ? "None" : std::to_string(Guessed_Label.first)) << ", ";
        std::cout << "GUESSED 2nd : " << (Guessed_Label.second.first == -1 ? "None" : std::to_string(Guessed_Label.second.first)) << ", ";
        std::cout << "GUESSED 3rd : " << (Guessed_Label.second.second == -1 ? "None" : std::to_string(Guessed_Label.second.second)) << " ";
        std::cout << "\n";
    }
    return 0;
}
