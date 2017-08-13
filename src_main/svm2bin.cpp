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
	ASSERT(argc == 3, "Invalid number of parameters");
	const char *input = argv[1];
	const char *output = argv[2];
	
	Platform::init();
	
	SVMDataReader reader(input);
	//BinaryDataReader reader(input);
	bool open_input = reader.init();
	ASSERT(open_input, "Could not open input file");
	fd = open(output, O_CREAT | O_WRONLY | O_TRUNC | O_LARGEFILE, 0644);
	ASSERT(fd >= 0, "Could not open output file");

	SparseExample example;

	BinExampleCount num_examples = 0;
	SparseExample::Index max_feature_id = 0;
	ptr = buffer + sizeof(BinExampleCount) + sizeof(SparseExample::Index);

	auto start_time = Platform::getCurrentTime();

	while(reader.read(&example)) {
		// Output label
		checkBuffer(sizeof(BinLabel));
		*reinterpret_cast<BinLabel *>(ptr) = example.label;
		ptr += sizeof(BinLabel);

		// Output number of non-zero features
		checkBuffer(sizeof(BinNZFeatCount));
		*reinterpret_cast<BinNZFeatCount *>(ptr) = 
                    static_cast<BinNZFeatCount>(example.feats.size());
		ptr += sizeof(BinNZFeatCount);

		// Ouptut non-zero features
		VectorIterator<SparseVec> iterator(example.feats);

		for(; iterator; iterator.next()) {
                  checkBuffer(sizeof(SparseExample::Index) + sizeof(float));
                  *reinterpret_cast<SparseExample::Index *>(ptr) =
                      iterator.index();
                  ptr += sizeof(SparseExample::Index);
                  *reinterpret_cast<float *>(ptr) = iterator.value();
                  ptr += sizeof(float);

                  if (iterator.index() > max_feature_id) {
                    max_feature_id = iterator.index();
                  }
		}	  
		
		++num_examples;
		if (num_examples % 1000 == 0) { LOG(num_examples); }
	}

	writeBuffer();

	// Write number of examples and number of features
	++max_feature_id;
	lseek(fd, 0, SEEK_SET);
        int flag = write(
            fd, reinterpret_cast<char *>(&num_examples), sizeof(num_examples));
	flag |= write(
            fd, reinterpret_cast<char *>(&max_feature_id), sizeof(max_feature_id));
        ASSERT(flag > 0, "");
	close(fd);

	auto end_time = Platform::getCurrentTime();

	LOG("Time: " << Platform::getDurationms(start_time, end_time) << "ms");
	LOG("Number of examples: " << num_examples);
	LOG("Number of features: " << max_feature_id);
}
