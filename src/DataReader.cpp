#include "DataReader.h"
#include <cstring>
#include <cmath>

bool SVMDataReader::read(SparseExample* example) {
	if (ptr_ == buffer_end_) { 
		if (end_of_file_) { return false; }
		else {
			// Reached end of buffer. Fill buffer from file.
			bufferRead();
			ptr_ = buffer_;
			if (end_of_file_) { return false; }
		}
	}

	example->feats.clear();

	// find end of line
	char *p = (char*) memchr(ptr_, '\n', buffer_end_ - ptr_);
	
	if (p == 0) {		
		// Buffer ended before finding the end of the line. 
		// Move remaining data into stash and read from the file
		// into the buffer.
		readMoreBytes();

		p = (char*) memchr(buffer_, '\n', buffer_end_ - buffer_);
		ASSERT(p != 0, "Insufficient buffer");
	}

	//processLine
	processLine(ptr_, p - ptr_, example);
	ptr_ = p+1;
	++num_examples_;
	return true;
}

void SVMDataReader::processLine(char *line, size_t size,
                                SparseExample* example) {

  line[size] = 0;

  // Trim trailing white space
  while (line[size-1] == ' ') {
    line[--size] = 0;
  }
  
  char *end = line + size;
     
  char *label_end = (char*) memchr(line, ' ', size);

  if (label_end != 0) { // label_end can be 0 if all features are zero
    *label_end = 0;
  }
    
  example->label = atof(line);

  if (label_end != 0) {
    SparseExample::Index index; double value;

    char *last_p = label_end + 1;
    char *p = last_p;
    
    while((p = (char*) memchr(p+1, ' ', end-p-1)) != 0) {
      // Process Token		
      processToken(last_p, p-last_p, &index, &value);
      example->feats.addElement(index, value);
      last_p = p+1;
    }

    // Process last token in file
    processToken(last_p, end-last_p, &index, &value);
    example->feats.addElement(index, value);
  }
}

void SVMDataReader::processToken(char *token, size_t size, SparseExample::Index *index,
								 double *value) {
	token[size] = 0;
	int idx = 0;	
	while(*token != ':') idx = (idx)*10 + (*(token++) - '0');

	*index = idx;
	*value = atof(token+1);
}

bool BinaryDataReader::doInit() {
	Super::doInit();

	num_examples_ = *reinterpret_cast<BinExampleCount *>(ptr_);
	ptr_ += sizeof(num_examples_);
	num_features_ = *reinterpret_cast<SparseExample::Index *>(ptr_);
	ptr_ += sizeof(num_features_);

	LOG(num_examples_ << " " << num_features_);
	return true;
}

struct Entry {
	SparseExample::Index index;
	float value;
};
static_assert(sizeof(Entry) == sizeof(SparseExample::Index) + sizeof(float), "Size mismatch");

bool BinaryDataReader::read(SparseExample* example) {
  if (num_examples_ == 0) { return false; }
  --num_examples_;

  int read_size = sizeof(BinLabel) + sizeof(BinNZFeatCount);
	
  if (read_size > buffer_end_ - ptr_) { readMoreBytes(); }
	
  BinLabel *label = reinterpret_cast<BinLabel *>(ptr_);
  example->label = *label;
  ptr_ += sizeof(BinLabel);
  BinNZFeatCount num_nonzero = *reinterpret_cast<BinNZFeatCount *>(ptr_);
  ptr_ += sizeof(BinNZFeatCount);
 
  read_size = num_nonzero * sizeof(Entry);

  if (read_size > buffer_end_ - ptr_) { readMoreBytes(); }
  ASSERT(read_size <= buffer_end_ - ptr_, "Insufficient buffer");

  Entry *entries = reinterpret_cast<Entry *>(ptr_);

  example->feats.clear();
  example->feats.reserve(num_nonzero);

  for(BinNZFeatCount i = 0; i < num_nonzero; ++i) {
    example->feats.addElement(entries[i].index, entries[i].value);
  }

  ptr_ += read_size;
  return true;
}

void BinaryDataReader::readTrainingFile(
    const char *file_name, bool normalize_examples,
    std::vector<SparseVec> &data, std::vector<double> &labels,
    int &numFeatures) {    
  BinaryDataReader reader(file_name);
  numFeatures = -1;
  bool init_succeed = reader.init();
  ASSERT(init_succeed, "Could not read file" << file_name);

  numFeatures = reader.num_features();
  const int num_examples = reader.num_examples();

  data.clear();
  data.reserve(num_examples);
  labels.clear();
  labels.reserve(num_examples);
  SparseExample example;

  for(int i = 0; i < num_examples; ++i) {	   
    reader.read(&example);

    if (normalize_examples) {
      double norm = 0.0;
      for(const auto &x : example.feats) {
        norm += x.second * x.second;
      }
      
      norm = sqrt(norm);
      if (norm == 0.0) {norm = 1.0;}

      for(auto &x : example.feats) {
	x.second /= norm;
      }
    }
    
    data.push_back(example.feats);

    double label = example.label > 0.0 ?1.0 :0.0;
    labels.push_back(label);

    if(i % 10000 == 0) LOG("Read " << i << " examples");
  }
  
  reader.close();   
}

