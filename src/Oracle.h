#ifndef _SVRG_ORACLE_H_
#define _SVRG_ORACLE_H_

#include <string>
#include <unordered_map>
#include <vector>

#include "DataReader.h"
#include "VectorUtils.h"

template<class ParamVector, class Gradient>
class Oracle {
 public:	
  virtual ~Oracle() {}

  virtual void computeGradient(const ParamVector &params, int instance, Gradient &output) const = 0;
  virtual double computeObjective(const ParamVector &params, int instance) const = 0;
  virtual double computeObjAndGradient(const ParamVector &params, int instance, Gradient &out_gradient) const {
    computeGradient(params, instance, out_gradient);
    return computeObjective(params, instance);
  }
  virtual int getNumInstances() const = 0;
  virtual int getDimension() const = 0;

  virtual void evalParams(
      const ParamVector &x,
      std::unordered_map<std::string, double> &output) const = 0;

  //TODO: This is a temp fix for SAGA
  virtual const Gradient *getInstance(int instance) const = 0;  
};

template<class ParamVector, class Label = double>
class SparseExampleOracle : public Oracle<ParamVector, SparseVec> {
 public:
  typedef SparseVec Gradient;
  SparseExampleOracle(const std::vector<SparseVec> *examples,
                      const std::vector<Label> *labels, int num_features,
                      double l2_reg)
      : examples_(examples), labels_(labels), num_features_(num_features),
        l2_reg_(l2_reg), feature_counts_(num_features_) {
    for(const auto &example : *examples) {
      VectorIterator<SparseVec> iterator(example);

      for(; iterator; iterator.next()) {
        ++feature_counts_[iterator.index()];
      }
    }
  }

  const Gradient *getInstance(int instance) const final {
    return &(*examples_)[instance];
  }

  void computeGradient(const ParamVector &params, int instance_id, Gradient &output) const final {
    const SparseVec &instance = (*examples_)[instance_id];
    doComputeGradient(params, (*examples_)[instance_id], (*labels_)[instance_id], output);

    // Add regularization
    VectorIterator<SparseVec> instance_iterator(instance);
    ModifyingVectorIterator<SparseVec> grad_iterator(output);

    for(; instance_iterator; instance_iterator.next(), grad_iterator.next()) {
      ASSERT(grad_iterator, "");
      ASSERT(instance_iterator.index() == grad_iterator.index(), "");

      int idx = instance_iterator.index();
      double x = params[idx];

      grad_iterator.valueRef() += 2 * l2_reg_ * x / feature_counts_[idx];
    }

    ASSERT(!grad_iterator, "");
  }

  double computeObjective(const ParamVector &params, int instance_id) const final {
    const SparseVec &instance = (*examples_)[instance_id];
    double obj = doComputeObjective(params, instance, (*labels_)[instance_id]);

    // Add regularization
    VectorIterator<SparseVec> instance_iterator(instance);

    for(; instance_iterator; instance_iterator.next()) {
      int idx = instance_iterator.index();
      double x = params[idx]; 

      obj += l2_reg_ * x * x / feature_counts_[idx];
    }

    return obj;
  }

  double computeObjAndGradient(const ParamVector &params, int instance_id, Gradient &out_gradient) const final {
    const SparseVec &instance = (*examples_)[instance_id];
    double obj = doComputeObjAndGradient(params, instance,
                                         (*labels_)[instance_id], out_gradient);

    // Add regularization
    VectorIterator<SparseVec> instance_iterator(instance);
    ModifyingVectorIterator<SparseVec> grad_iterator(out_gradient);

    for(; instance_iterator; instance_iterator.next(), grad_iterator.next()) {
      ASSERT(grad_iterator, "");
      ASSERT(instance_iterator.index() == grad_iterator.index(), "");

      int idx = instance_iterator.index();
      double x = params[idx];
      
      grad_iterator.valueRef() += 2 * l2_reg_ * x / feature_counts_[idx];
      obj += l2_reg_ * x * x / feature_counts_[idx];
    }

    ASSERT(!grad_iterator, "");

    return obj;
  }

  int getNumInstances() const override {return examples_->size();}
  int getDimension() const override {return num_features_;}

 protected:
  virtual void doComputeGradient(const ParamVector &params,
                                 const SparseVec &instance, const Label& label,
                                 Gradient &output) const = 0;
  virtual double doComputeObjective(const ParamVector &params,
                                    const SparseVec &instance,
                                    const Label& label) const = 0;
  virtual double doComputeObjAndGradient(
      const ParamVector &params, const SparseVec &instance,
      const Label& label, Gradient &out_gradient) const {
    doComputeGradient(params, instance, label, out_gradient);
    return doComputeObjective(params, instance, label);
  }

 private:
  int num_features_;
  double l2_reg_;

  // For each feature, stores number of examples where the feature
  // is not zero
  std::vector<int> feature_counts_;
  const std::vector<SparseVec> *examples_;
  const std::vector<double> *labels_;
};

#endif

