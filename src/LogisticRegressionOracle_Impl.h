#include "LogisticRegressionOracle.h"

#include <cstdlib>
#include <fstream>
#include <string>
#include <sstream>

using namespace std;

template<class ParamVector>
double LogisticRegressionOracle<ParamVector>::computeP(
    const ParamVector &params, const SparseVec &instance) const {
  double dot = VectorUtils::sparseDot(instance, params);
  double p = 1.0 / (1.0 + exp(-dot));
  return p;
}

template<class ParamVector>
double LogisticRegressionOracle<ParamVector>::computeObjectiveGivenP(
    double p, const SparseVec &instance, const double& label) const {
  double output = -(label > 0.0 ?log(p) :log(1-p));
  return output;
}

template<class ParamVector>
void LogisticRegressionOracle<ParamVector>::computeGradientGivenP(
    double p, const SparseVec &instance, const double& label,
    SparseVec &output) const {
  output = instance;  
  for(auto &x : output) {x.second *= p - label;}  
}

template<class ParamVector>
void LogisticRegressionOracle<ParamVector>::evalParams(
    const ParamVector &param_spec,    
    std::unordered_map<std::string, double> &output) const {
  if(test_examples_ == 0) {return;}
  
  int n_test = test_examples_->size();
  int num_mistakes = 0;
  
#pragma omp parallel for reduction(+:num_mistakes)
  for(int i = 0; i < n_test; ++i) {
    double p = computeP(param_spec, (*test_examples_)[i]);
    int label = (*test_labels_)[i];
    num_mistakes += (p < 0.5) ?label :(1-label);
  }

  output["test_error"] = static_cast<double>(num_mistakes) / n_test;
}


