#ifndef _SVRG_VECTOR_H
#define _SVRG_VECTOR_H

#include <unordered_map>
#include <vector>

// Dense vector
class Vector : public std::vector<double> {
 public:
  typedef size_t Index;
  
  Vector() {}
  Vector(Index size)
      : vector<double>(size) {}
  
  void fill(double value) {
    std::fill(begin(), end(), 0.0);
  }

  double dot(const Vector &other) const {
    double output = 0.0;
    for(Index i = 0; i < size(); ++i) {output += (*this)[i] * other[i];}
    return output;
  }
};

// Read iterator for sparse vectors.
// Can be specified for each class that represents
// a sparse vector (See VectorIterator<SparseVec> below).
template<class IterableVector>
class VectorIterator {
  // A specification must provide the following methods:

  // Initializes the iterator to point to the first non-zero
  // entry in the vector.  
  VectorIterator(const IterableVector &);

  // determines whether the iterator did not pass the end of the vector
  operator bool() const;

  // advances the iterator to then next non-zero entry in the vector.
  void next();

  // returns the index at the current iterator position.
  typename IterableVector::Index index() const;
  
  double value() const; // returns the value at the current iterator position.
};

// Read/write iterator for sparse vectors.
template<class IterableVector>
class ModifyingVectorIterator : public VectorIterator<IterableVector> {
  // A specification must provide the following methods:
  
  // Initializes the iterator to point to the first non-zero
  // entry in the vector.  
  ModifyingVectorIterator(const IterableVector &);

  // returns a reference to the value at the current iterator position.
  double &valueRef() const;
};

// Sparse Vector
class SparseVec {
 public:
  typedef size_t Index;
  typedef std::vector<std::pair<Index, double>> InnerStorage;
  
  SparseVec() {}
  SparseVec(size_t size) {
    map_.reserve(size);
  }
  
  void addElement(Index index, double value) {
    ASSERT(map_.size() > 0 || map_.back().first < index,
           "Out of order insertion");
    map_.push_back(std::pair<Index, double>(index, value));
  }

  void clear() {map_.clear();}
  void reserve(size_t size) {map_.reserve(size);}
  size_t size() const {return map_.size();}

  InnerStorage::iterator begin() {return map_.begin();}
  InnerStorage::iterator end() {return map_.end();}
  InnerStorage::const_iterator begin() const {return map_.begin();}
  InnerStorage::const_iterator end() const {return map_.end();}
  
 private:
  InnerStorage map_;

  friend class VectorIterator<SparseVec>;
  friend class ModifyingVectorIterator<SparseVec>;
};

struct ScaledSparseVec {
  typedef size_t Index;
  double scale;
  const SparseVec *vector;
};

template<>
class VectorIterator<SparseVec> {
 public:
  VectorIterator(const SparseVec &vector)
      : iterator_(vector.map_.begin()), end_iterator_(vector.map_.end()) {}
  
  int index() const {return iterator_->first;}
  double value() const {return iterator_->second;}
  
  operator bool() const {return iterator_ != end_iterator_;}
  void next() {++iterator_;}

private:
  SparseVec::InnerStorage::const_iterator iterator_;
  SparseVec::InnerStorage::const_iterator end_iterator_;
};

template<>
class ModifyingVectorIterator<SparseVec> {
 public:
  ModifyingVectorIterator(SparseVec &vector)
      : iterator_(vector.map_.begin()), end_iterator_(vector.map_.end()) {}
  
  int index() const {return iterator_->first;}
  double value() const {return iterator_->second;}
  double &valueRef() const {return iterator_->second;}

  operator bool() const {return iterator_ != end_iterator_;}
  void next() {++iterator_;}

private:
  SparseVec::InnerStorage::iterator iterator_;
  SparseVec::InnerStorage::iterator end_iterator_;
};

template<>
class VectorIterator<ScaledSparseVec> {
 public:
  VectorIterator(const ScaledSparseVec &vector)
      : iterator_(*vector.vector),
        scale_(vector.scale) {}
  
  int index() const {return iterator_.index();}
  double value() const {return scale_ * iterator_.value();}
  
  operator bool() const {return iterator_;}
  void next() {iterator_.next();}

private:
  VectorIterator<SparseVec> iterator_;  
  double scale_;
};


#endif
