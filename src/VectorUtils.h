#ifndef _SVRG_VECTORUTILS_H_
#define _SVRG_VECTORUTILS_H_

#include "Vector.h"
#include "Platform.h"

// This class provides utility functions for vector operations.
// It supports arbitrary representation for dense and sparse vectors.
// A dense vector has to implement a size() method and [] operator to
// access elements.
// A sparse vector has to support VectorIterator (and ModVectorIterator if
// write access is needed) defined in Vector.h
class VectorUtils {
public:  
  // Computes self := self * self_scale + other * other_scale
  template<class DenseVector>
  static void addVector(DenseVector &self, double self_scale,
                        DenseVector &other, double other_scale) {
    ASSERT(self.size() == other.size(), "Incompatible vectors");

    for(typename DenseVector::Index i = 0; i < self.size(); ++i) {
      self[i] = self[i] * self_scale + other[i] * other_scale;
    }
  }

  // Computes v := v + increment
  //  If atomicComponentUpdates is set true, double addition is
  //  performed as an atomic operation (See Platform::addDouble) 
  template<class DenseVector>
  static void addVector(DenseVector &v,
                        const DenseVector &increment,
                        bool atomicComponentUpdates) {
    if(atomicComponentUpdates) {
      double *raw = v.data();
      int n = v.size();

      for(int i = 0; i < n; i++) {
        Platform::atomicAdd(raw + i, increment[i]);
      }
    } else {
      int n = v.size();

      for(int i = 0; i < n; i++) {
        v[i] += increment[i];
      }      
    }
  }

  // Same as above but with for a sparse increment vector.
  template<class DenseVector, class IterableVector>
  static void addVector(DenseVector &v,
                 const IterableVector &increment,
                 double scale,
                 bool atomicComponentUpdates) {
    double *raw = v.data();
    VectorIterator<IterableVector> iterator(increment);

    if(atomicComponentUpdates) {
      for(; iterator; iterator.next()) {
        int idx = iterator.index();
        Platform::atomicAdd(raw + idx, iterator.value() * scale);
      }
    } else {
      for(; iterator; iterator.next()) {
        int idx = iterator.index();
        raw[idx] += iterator.value() * scale;
      }
    }
  }

  template<class IterableVector1, class IterableVector2>
  static void addVector(const IterableVector1 &v1,
                        const IterableVector2 &v2,
                        SparseVec &output) {
    output = v1;
    return;
    
    output.clear();
        
    VectorIterator<IterableVector1> it1(v1);
    VectorIterator<IterableVector2> it2(v2);
    
    while(it1 && it2) {
      if(it1.index() == it2.index()) {
        output.addElement(it1.index(), it1.value() + it2.value());
        it1.next();
        it2.next();
      } else if(it1.index() < it2.index()) {
        output.addElement(it1.index(), it1.value());
        it1.next();
      } else { // it1.index > it2.index
        output.addElement(it2.index(), it2.value());
        it2.next();
      }
    }

    while(it1) {
      output.addElement(it1.index(), it1.value());
      it1.next();
    }

    while(it2) {
      output.addElement(it2.index(), it2.value());
      it2.next();
    }
  }

  // Computes self := self * self_scale + other * other_scale
  // where self and other are two compatible sparse vectors.
  // Two sparse vectors are compatible if both have the same non-zero indices.
  template<class ModIterableVector, class IterableVector>
  static inline void addCompatibleVec(
      ModIterableVector &self, double self_scale, 
      const IterableVector &other, double other_scale) {
    ModifyingVectorIterator<ModIterableVector> self_iterator(self);
    VectorIterator<IterableVector> other_iterator(other);

    for(; self_iterator; self_iterator.next(), other_iterator.next()) {
      ASSERT(other_iterator, "Vectors are not compatible");
      ASSERT(self_iterator.index() == other_iterator.index(),
             "Vectors are not compatible");
		  
      self_iterator.valueRef() = self_iterator.value() * self_scale +
          other_iterator.value() * other_scale;
    }

    ASSERT(!other_iterator, "Vectors are not compatible");
  }

  // Adds to non-zero components in a sparse vector 'self'
  // the corresponding components in a dense vector 'other'.
  template<class ModIterableVector, class DenseVector>
  static inline void selectiveAddVec(
      ModIterableVector &self, double self_scale,
      const DenseVector &other, double other_scale) {
    ModifyingVectorIterator<ModIterableVector> self_iterator(self);

    for(; self_iterator; self_iterator.next()) {
      self_iterator.valueRef() = self_iterator.value() * self_scale +
          other[self_iterator.index()] * other_scale;
    }
  }
  
  // Computes the dot product between a sparse vector and aribtrary vector
  // representation that implements [] operator
  template<class IterableVector, class DenseVector>
  static double sparseDot(
      const IterableVector &sparse, const DenseVector &other) {
    VectorIterator<IterableVector> sparse_iterator(sparse);

    double output = 0.0;
    
    for(; sparse_iterator; sparse_iterator.next()) {
      output += sparse_iterator.value() * other[sparse_iterator.index()];
    }

    return output;
  }
};

#endif
