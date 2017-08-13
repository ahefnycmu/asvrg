// Convert SVM file format to Binary file format for fast reading.
// The format is as follows:
// Number of examples (64-bit integer)
// Number of features (32-bit integer)
// Examples where each examples is:
// - Label (1 byte)
// - Number of non-zero features (32-bit integer)
// - Non-zero features where each feature is:
//   * Feature index (32-bit integer)
//   * Feature value (32-bit float)

#include<unistd.h>
#include<fcntl.h>

#include "Platform.h"
#include "DataReader.h"

int fd;
const int BUFFER_SIZE = 16 * 1024;
char buffer[BUFFER_SIZE];
char * const buffer_end = buffer + BUFFER_SIZE;
char *ptr = buffer;

inline void writeBuffer() {
	int flag = write(fd, buffer, ptr-buffer);
        ASSERT(flag > 0, "");
	ptr = buffer;
}

inline void checkBuffer(int size) {
	if (ptr + size > buffer_end) {
		writeBuffer();
	}
}

int main(int argc, char **argv) {
	ASSERT(argc == 2, "Invalid number of parameters");
	const char *input = argv[1];
	
	Platform::init();
	
	BinaryDataReader reader(input);
	bool open_input = reader.init();
	ASSERT(open_input, "Could not open input file");

	SparseExample example;

	BinExampleCount num_examples = 0;
	SparseExample::Index max_feature_id = 0;

	auto start_time = Platform::getCurrentTime();

	while(reader.read(&example)) {
		// Output label
		std::cout << example.label << " ";

		// Ouptut non-zero features
		VectorIterator<SparseVec> iterator(example.feats);

		for(; iterator; iterator.next()) {
		  std::cout << iterator.index() << ":" << iterator.value() << " ";
		}	  

		std::cout << "\n";
		
		++num_examples;
		if (num_examples % 1000 == 0) { LOG(num_examples); }
	}

	// Write number of examples and number of features
	++max_feature_id;

	auto end_time = Platform::getCurrentTime();

	LOG("Time: " << Platform::getDurationms(start_time, end_time) << "ms");
	LOG("Number of examples: " << num_examples);
	LOG("Number of features: " << max_feature_id);
}
