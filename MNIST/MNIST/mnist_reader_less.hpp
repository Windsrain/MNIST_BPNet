#ifndef MNIST_READER_HPP
#define MNIST_READER_HPP

#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <cstdint>
#include <memory>

#include "mnist_reader_common.hpp"

namespace mnist {

	/*!
	 * \brief Represents a complete mnist dataset
	 * \tparam Pixel The type of a pixel
	 * \tparam Label The type of a label
	 */
	template <typename Pixel = uint8_t, typename Label = uint8_t>
	struct MNIST_dataset {
		std::vector<std::vector<Pixel>> training_images; ///< The training images
		std::vector<std::vector<Pixel>> test_images;     ///< The test images
		std::vector<Label> training_labels;              ///< The training labels
		std::vector<Label> test_labels;                  ///< The test labels
	};

	/*!
	 * \brief Read a MNIST image file and return a container filled with the images
	 * \param path The path to the image file
	 * \return A std::vector filled with the read images
	 */
	template <typename Pixel = uint8_t, typename Label = uint8_t>
	std::vector<std::vector<Pixel>> read_mnist_image_file(const std::string& path) {
		auto buffer = read_mnist_file(path, 0x803);

		if (buffer) {
			auto count = read_header(buffer, 1);
			auto rows = read_header(buffer, 2);
			auto columns = read_header(buffer, 3);

			//Skip the header
			//Cast to unsigned char is necessary cause signedness of char is
			//platform-specific
			auto image_buffer = reinterpret_cast<unsigned char*>(buffer.get() + 16);

			std::vector<std::vector<Pixel>> images;
			images.reserve(count);

			for (size_t i = 0; i < count; ++i) {
				images.emplace_back(rows * columns);

				for (size_t j = 0; j < rows * columns; ++j) {
					auto pixel = *image_buffer++;
					images[i][j] = static_cast<Pixel>(pixel);
				}
			}

			return images;
		}

		return {};
	}

	/*!
	 * \brief Read a MNIST label file and return a container filled with the labels
	 * \param path The path to the image file
	 * \return A std::vector filled with the read labels
	 */
	template <typename Label = uint8_t>
	std::vector<Label> read_mnist_label_file(const std::string& path) {
		auto buffer = read_mnist_file(path, 0x801);

		if (buffer) {
			auto count = read_header(buffer, 1);

			//Skip the header
			//Cast to unsigned char is necessary cause signedness of char is
			//platform-specific
			auto label_buffer = reinterpret_cast<unsigned char*>(buffer.get() + 8);

			std::vector<Label> labels(count);

			for (size_t i = 0; i < count; ++i) {
				auto label = *label_buffer++;
				labels[i] = static_cast<Label>(label);
			}

			return labels;
		}

		return {};
	}

	/*!
	 * \brief Read all training images and return a container filled with the images.
	 *
	 * The dataset is assumed to be in a mnist subfolder
	 *
	 * \return Container filled with the images
	 */
	template <typename Pixel = uint8_t>
	std::vector<std::vector<Pixel>> read_training_images(const std::string& path) {
		return read_mnist_image_file<Pixel>(path + "\\train-images.idx3-ubyte");
	}

	/*!
	 * \brief Read all test images and return a container filled with the images.
	 *
	 * The dataset is assumed to be in a mnist subfolder
	 *
	 * \return Container filled with the images
	 */
	template <typename Pixel = uint8_t>
	std::vector<std::vector<Pixel>> read_test_images(const std::string& path) {
		return read_mnist_image_file<Pixel>(path + "\\t10k-images.idx3-ubyte");
	}

	/*!
	 * \brief Read all training labels and return a container filled with the labels.
	 *
	 * The dataset is assumed to be in a mnist subfolder
	 *
	 * \return Container filled with the labels
	 */
	template <typename Label = uint8_t>
	std::vector<Label> read_training_labels(const std::string& path) {
		return read_mnist_label_file<Label>(path + "\\train-labels.idx1-ubyte");
	}

	/*!
	 * \brief Read all test labels and return a container filled with the labels.
	 *
	 * The dataset is assumed to be in a mnist subfolder
	 *
	 * \return Container filled with the labels
	 */
	template <typename Label = uint8_t>
	std::vector<Label> read_test_labels(const std::string& path) {
		return read_mnist_label_file<Label>(path + "\\t10k-labels.idx1-ubyte");
	}

	/*!
	 * \brief Read dataset.
	 *
	 * The dataset is assumed to be in a mnist subfolder
	 *
	 * \return The dataset
	 */
	template <typename Pixel = uint8_t, typename Label = uint8_t>
	MNIST_dataset<Pixel, Label> read_dataset(const std::string& path) {
		MNIST_dataset<Pixel, Label> dataset;

		dataset.training_images = read_training_images<Pixel>(path);
		dataset.training_labels = read_training_labels<Label>(path);

		dataset.test_images = read_test_images<Pixel>(path);
		dataset.test_labels = read_test_labels<Label>(path);

		return dataset;
	}

} //end of namespace mnist

#endif
