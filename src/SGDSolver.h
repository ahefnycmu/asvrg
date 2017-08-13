#ifndef _SVRG_SGD_SOLVER_H_
#define _SVRG_SGD_SOLVER_H_

#include "Solver.h"

// Implementation of Solver abstract class for Stochastic Gradient Descent
// with sparse gradients.
class SGDSolver : public Solver<Vector, SparseVec> {
  typedef Solver<Vector, SparseVec> Super;
  
public:
  typedef typename Super::Solution Solution;
  typedef typename Super::TraceElement TraceElement;
  typedef Vector ParamVector;
  
  struct Options : public Super::Options {
    Options() {}    
    
    double target_objective = -std::numeric_limits<double>::infinity();
    int max_num_epochs = 1000;
    int num_nupdates_per_epoch = 2; //Number of updates per epoch per n
    double step = 1e-4;
    double alpha_step = -1; 
    ParallelMode parallel_mode = ParallelMode::FREE_FOR_ALL;

    void print(std::ostream& out) const override {
      const auto &options = *this;
      out << "Target: " << options.target_objective << std::endl;
      out << "MaxEpoch: " << options.max_num_epochs << std::endl;
      out << "NUpdatePerEpoch: " << options.num_nupdates_per_epoch << std::endl;
      out << "Step: " << options.step << std::endl;
      out << "Alpha: " << options.alpha_step << std::endl;
      out << "ParallelMode: " <<
          options.parallel_mode.toString() << std::endl;
    }
  };

  SGDSolver(const Options &options = Options())
      : options_(options) {}
  
  void setOptions(const Options &options) {options_ = options;}  
  Solution solve(Oracle<Vector, SparseVec> *oracle) override;

private:
  Options options_;
};

#endif
