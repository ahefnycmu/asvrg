#ifndef _SVRG_BATCH_ORACLE_H_
#define _SVRG_BATCH_ORACLE_H_

#include <unordered_map>
#include "Oracle.h"
#include "Platform.h"
#include "Vector.h"

// A wrapper class to extract mini-batch gradients given another oracle.
// Each "instance" refers to a minibatch.
template<class ParamVector>
class BatchOracle : public Oracle<ParamVector, SparseVec> {
 public:
  // Constructs a new BatchOracle.
  // oracle: Oracle used to extract gradients for individual instances.
  // own_oracle: If true, BatchOracle destroys oracle in the destructor.
  // batch_size: NUmber of examples per mini-batch.
  BatchOracle(Oracle<ParamVector, SparseVec> *oracle, bool own_oracle,
              int batch_size)
      : oracle_(oracle), own_oracle_(own_oracle), batch_size_(batch_size) {
    num_individual_instances_ = oracle->getNumInstances();
    num_batches_ = (num_individual_instances_ + batch_size - 1) / batch_size;
    dimension_ = oracle->getDimension();
    
    int num_threads = Platform::getNumLocalThreads();
        storage_ = new ThreadStorage[num_threads];

        for (int i = 0; i < num_threads; ++i) {
          storage_[i].vec_a.reserve(dimension_);
          storage_[i].vec_b.reserve(dimension_);
          storage_[i].vec_c.reserve(dimension_);
        }
  }

  ~BatchOracle() {
    if (own_oracle_) {delete oracle_;}
    delete[] storage_;
  }

  void computeGradient(
      const ParamVector &params, int instance,
      SparseVec &output) const override {

    int batch_start = instance * batch_size_;
    int batch_end = batch_start + batch_size_;
    if(batch_end > num_individual_instances_) {
      batch_end = num_individual_instances_;
    }

    int thread_id = Platform::getThreadId();
    SparseVec *instance_gradient = &storage_[thread_id].vec_a;
    SparseVec *gradient_sum = &storage_[thread_id].vec_b;
    SparseVec *gradient_sum_new = &storage_[thread_id].vec_c;

    output.clear();
    gradient_sum->clear();
    gradient_sum_new->clear();
    
    for(int i = batch_start; i < batch_end; ++i) {
      // Compute instance gradient
      oracle_->computeGradient(params, i, *instance_gradient);

      // Add it to gradient sum
      VectorUtils::addVector(
          *instance_gradient, *gradient_sum, *gradient_sum_new);

      // Exchange gradient sum pointers so that the sum up to i is stored
      // in *gradient_sum
      SparseVec *tmp = gradient_sum;
      gradient_sum = gradient_sum_new;
      gradient_sum_new = tmp;
    }
    
    output = *gradient_sum;

    for(auto &x : output) {
      x.second /= (batch_end - batch_start);
    }    
  }
  
  double computeObjective(const ParamVector &params,
                          int instance) const override {
        int batch_start = instance * batch_size_;
    int batch_end = batch_start + batch_size_;
    if(batch_end > num_individual_instances_) {
      batch_end = num_individual_instances_;
    }

    double output = 0.0;

    for(int i = batch_start; i < batch_end; ++i) {
      output += oracle_->computeObjective(params, i);
    }
        
    output /= (batch_end - batch_start);
    return output;    
  }
  
  double computeObjAndGradient(const ParamVector &params, int instance,
                               SparseVec &out_gradient) const override {
    int batch_start = instance * batch_size_;
    int batch_end = batch_start + batch_size_;
    if(batch_end > num_individual_instances_) {
      batch_end = num_individual_instances_;
    }

    int thread_id = Platform::getThreadId();
    SparseVec *instance_gradient = &storage_[thread_id].vec_a;
    SparseVec *gradient_sum = &storage_[thread_id].vec_b;
    SparseVec *gradient_sum_new = &storage_[thread_id].vec_c;

    double output = 0.0;
    out_gradient.clear();
    gradient_sum->clear();
    gradient_sum_new->clear();
   
    for(int i = batch_start; i < batch_end; ++i) {
      // Compute instance gradient and objective
      oracle_->computeObjAndGradient(params, i, *instance_gradient);

      // Add instacne gradient to gradient sum
      VectorUtils::addVector(
          *instance_gradient, *gradient_sum, *gradient_sum_new);

      // Exchange gradient sum pointers so that the sum up to i is stored
      // in *gradient_sum
      SparseVec *tmp = gradient_sum;
      gradient_sum = gradient_sum_new;
      gradient_sum_new = tmp;
    }

    out_gradient = *gradient_sum;

    for(auto &x : out_gradient) {
      x.second /= (batch_end - batch_start);
    }

    output /= (batch_end - batch_start);

    return output;    
  }

  void evalParams(
      const ParamVector &x,
      std::unordered_map<std::string, double> &output) const override {
    oracle_->evalParams(x, output);
  }
  
  const SparseVec *getInstance(int instance) const override {
    ASSERT(false, "Not supported");
  }
  
  int getNumInstances() const override {return num_batches_; }
  int getDimension() const override {return dimension_;}
  
 private:
  struct ThreadStorage {
    // Three sparse vectors to store instance gradient and summation of
    // gradients across the minibatch.
    SparseVec vec_a;
    SparseVec vec_b;
    SparseVec vec_c;
  };
 
  ThreadStorage *storage_;
  
  Oracle<ParamVector, SparseVec> *oracle_;
  bool own_oracle_;
  int batch_size_;
  int num_individual_instances_;
  int num_batches_;
  int dimension_;
};

#endif
