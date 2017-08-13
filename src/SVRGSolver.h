#ifndef _SVRG_SVRG_SOLVER_H_
#define _SVRG_SVRG_SOLVER_H_

#include "Solver.h"
#include "SGDSolver.h"

// For effeciency, the SVRG parameter vector is represented as
// x + avg_gradient_multiple * avg_gradient,
// where avg_gradient is the average gradient for teh last iterate in the
// previous epoch and x accumulates sparse updates in the current epoch.
struct SVRGParamVector {
  const Vector *x;
  const Vector *avg_gradient;
  double avg_gradient_multiple;

  inline double operator[](int index) const {
    return (*x)[index] + avg_gradient_multiple * (*avg_gradient)[index];
  }
};

// Implementation of Solver abstract class for SVRG with sparse gradients.
class SVRGSolver : public Solver<SVRGParamVector, SparseVec> {
  typedef Solver<SVRGParamVector, SparseVec> Super;
  
public:
  typedef typename Super::Solution Solution;
  typedef typename Super::TraceElement TraceElement;
  typedef SVRGParamVector ParamVector;
  
  typedef SGDSolver::Options Options;

  SVRGSolver(const Options &options = Options())
      : options_(options) {}
  
  void setOptions(const Options &options) {options_ = options;}  
  Solution solve(Oracle<SVRGParamVector, SparseVec> *oracle) override;

private:
  Options options_;
};

#endif
