#ifndef _SVRG_DATAREADER_H_
#define _SVRG_DATAREADER_H_

#include <string>
#include <cstring>

#include <unistd.h>
#include <fcntl.h>

#include "Vector.h"

template<class T>
class DataReader {
 public:
  virtual bool init() {
    if(!closed_) {close();}
    bool init_successful = doInit();
    if(init_successful) {closed_ = false;}
    return init_successful;
  }

  void close() {
    ASSERT(!closed_, "DataReader already closed");
    doClose();
    closed_ = true;
  }

  virtual bool read(T* example) = 0;

 protected:
  virtual bool doInit() = 0;
  virtual void doClose() = 0;
  bool closed() {return closed_;}

 private:
  bool closed_ = true;
};

template<class T> 
class DataReaderFromFile : public DataReader<T> {
 public:
  DataReaderFromFile(const std::string &file_name) 
      : file_name_(file_name) {}

 protected:
  // Reads BUFFER_SIZE bytes from file into buffer. 
  // Sets buffer_end_ to the end of read bytes.
  size_t bufferRead();

  // Move remaining buffer data into stash and read from the file
  // into the buffer.
  void readMoreBytes();

  bool doInit() override;
  void doClose() override;
  
  std::string file_name_;

  static constexpr int BUFFER_SIZE = 256 * 1024;
  static constexpr int STASH_SIZE = BUFFER_SIZE;

  char storage_[BUFFER_SIZE + STASH_SIZE];

  // The buffer stores contents that are from the file to be processed.
  char * const buffer_ = storage_ + STASH_SIZE;

  // When only a part of an example is stored in the buffer. This part is
  // moved into the stage before rewriting the buffer with new contents.
  char * const stash_ = storage_;

  char *last_stash_ = buffer_;
  char *ptr_; // Points to the beginning of the next line in the buffer
  char *buffer_end_;

  int file_descriptor_ = -1; 
  bool end_of_file_ = true;
};

struct SparseExample {
	typedef int Index;

	int label;
	SparseVec feats;
};

class SVMDataReader : public DataReaderFromFile<SparseExample> {
	typedef DataReaderFromFile<SparseExample> Super;
 public:
	using Super::Super;
	virtual bool read(SparseExample* example) override;

 protected:	
	virtual bool doInit() override {
	  num_examples_ = 0;
	  return Super::doInit();
	}

 private:
	void processLine(char *line, size_t size, SparseExample* example);
	void processToken(char *token, size_t size, SparseExample::Index *index,
					  double *value);

	unsigned long long num_examples_;
};

typedef char BinLabel;
typedef long long BinExampleCount;
typedef int BinNZFeatCount; //Non-zero Feature Count

class BinaryDataReader : public DataReaderFromFile<SparseExample> {
  typedef DataReaderFromFile<SparseExample> Super;
 public:
  using Super::Super;
  virtual bool read(SparseExample* example) override;

  BinExampleCount num_examples() const { return num_examples_;}
  SparseExample::Index num_features() const { return num_features_; }

  static void readTrainingFile(
      const char *file_name, bool normalize_examples,
      std::vector<SparseVec> &data,
      std::vector<double> &labels,
      int &numFeatures);

 protected:
    bool doInit() override;

 private:
    BinExampleCount num_examples_;
    SparseExample::Index num_features_;
  };

// =================================================================
// Implementation
// =================================================================

template<class T>
bool DataReaderFromFile<T>::doInit() {
  file_descriptor_ = ::open(file_name_.c_str(), O_RDONLY);

  if (file_descriptor_ < 0) { return false; }

  end_of_file_ = false;
  posix_fadvise(file_descriptor_, 0, 0, 1);
  bufferRead();
  ptr_ = buffer_;
	
  return true;
}

template<class T>
void DataReaderFromFile<T>::doClose() {
  ::close(file_descriptor_);
}

template<class T>
size_t DataReaderFromFile<T>::bufferRead() {
  ASSERT(!end_of_file_, "Attempting to read after EOF");

  size_t b = ::read(file_descriptor_, buffer_, BUFFER_SIZE);
  buffer_end_ = buffer_ + b;
  end_of_file_ = (b == 0);
  return b;
}

template<class T>
void DataReaderFromFile<T>::readMoreBytes() {
  const size_t stash_size = buffer_end_ - ptr_;
  ASSERT(stash_size <= STASH_SIZE, "Insufficient stash size");
  char *stash_ptr = buffer_ - stash_size;	
  memcpy(stash_ptr, ptr_, stash_size);
  ptr_ = stash_ptr;

  bufferRead();
}

#endif
