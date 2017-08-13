#ifndef SVRG_LOGISTIC_REGRESSION_ORACLE_
#define SVRG_LOGISTIC_REGRESSION_ORACLE_

#include "Oracle.h"

template<class ParamVector>
class LogisticRegressionOracle : public SparseExampleOracle<ParamVector, double> {
  typedef SparseExampleOracle<ParamVector, double> Super;
  typedef double Label;
 public:
  LogisticRegressionOracle(const std::vector<SparseVec> *examples,
                           const std::vector<Label> *labels,
                           int num_features,
                           double l2_reg,
                           const std::vector<SparseVec> *test_examples = 0,
                           const std::vector<Label> *test_labels = 0)
      : Super(examples, labels, num_features, l2_reg),
        test_examples_(test_examples), test_labels_(test_labels) {}

  void evalParams(
      const ParamVector &x,
      std::unordered_map<std::string, double> &output) const override;
  
  static void readTrainingFile(
      const char *fileName, bool normalize_examples,
      std::vector<SparseVec> &data, std::vector<double> &labels, int &numFeatures);

 protected:
  void doComputeGradient(const ParamVector &params, const SparseVec &instance,
                         const double& label, SparseVec &output) const override {
    double p = computeP(params, instance);
    computeGradientGivenP(p, instance, label, output);
  }
  
  double doComputeObjective(const ParamVector &params, const SparseVec &instance,
                            const double& label) const override {
    double p = computeP(params, instance);
    return computeObjectiveGivenP(p, instance, label);
  }

  double doComputeObjAndGradient(const ParamVector &params,
                                 const SparseVec &instance,
                                 const Label& label,
                                 SparseVec &out_gradient) const override {
    double p = computeP(params, instance);
    computeGradientGivenP(p, instance, label, out_gradient);
    return computeObjectiveGivenP(p, instance, label);
  }
  
  void computeGradientGivenP(
      double p, const SparseVec &instance, const double& label, 
      SparseVec &output) const;
  double computeObjectiveGivenP(
      double p, const SparseVec &instance, const double& label) const;

 private:
  double computeP(const ParamVector &params, const SparseVec &instance) const;
  const std::vector<SparseVec> *test_examples_;
  const std::vector<Label> *test_labels_;
};

#include "LogisticRegressionOracle_Impl.h"

#endif
