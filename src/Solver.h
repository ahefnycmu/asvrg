#ifndef _SVRG_SOLVER_H_
#define _SVRG_SOLVER_H_

#include <iostream>
#include <random>
#include <limits>
#include <memory>
#include <vector>
#include "Platform.h"
#include "Oracle.h"

// A class representing possible parallel modes. Can be used as a scoped enum
// but supports toString and fromString methods.
class ParallelMode {
 public:
  enum Mode {
    FREE_FOR_ALL, // Lock-free with non-atomic updates.
    LOCK_FREE,  // Lock-free with atomic updates.
    LOCKED // Common-read exclusive write lock.
  };

  ParallelMode(Mode mode)
      : mode_(mode) {}

  operator Mode() const {return mode_;}
  
  std::string toString() const {
    switch(mode_) {
      case ParallelMode::FREE_FOR_ALL: return "FREE_FOR_ALL"; break;
      case ParallelMode::LOCK_FREE: return "LOCK_FREE"; break;
      case ParallelMode::LOCKED: return "LOCKED"; break;
      default: return ""; break;
    }
  }

  static ParallelMode fromString(const std::string &str) {
    if(str == "FREE_FOR_ALL") {return ParallelMode::FREE_FOR_ALL;}
    else if(str == "LOCK_FREE") {return ParallelMode::LOCK_FREE;}
    else if(str == "LOCKED") {return ParallelMode::LOCKED;}
    else {ASSERT(false, "Invalid parallel mode.");}
  }

 private:
  Mode mode_;
};

// Template abstract class for solvers.
// Template parameters specify paramater vector representation and gradient
// representation.
template<class ParamVector, class Gradient>
class Solver {
 public:
  struct Options {
    virtual void print(std::ostream& out) const = 0;
  };
  
  struct TraceElement {
    int timems;
    double objective;
    double grad_sq_norm;
    std::unordered_map<std::string, double> other_info;
  };
  
  struct Solution {
    Vector x;
    double objective;
    int timems;
    std::vector<TraceElement> trace;
  };

  virtual Solution solve(Oracle<ParamVector, Gradient> *oracle) = 0;

 protected:
  // Creates a vector of random engines initialized with different prime
  // seeds.
  static std::vector<std::default_random_engine> createRandomEngines(
      int num_engines) {
    int primes[] = {2389, 4561, 7177, 8803, 10009, 10753, 14057, 15391, 18127,
                    20341, 23887, 26437, 28703, 30971, 38177, 44201, 52009,
                    58601, 62903, 78079, 80167, 83407, 86179, 87221, 90473,
                    92831, 95747, 99989, 101419, 101837, 104243, 104729};

    std::vector<std::default_random_engine> rand_engines;
    rand_engines.reserve(num_engines);
    
    for (int i = 0; i < num_engines; ++i) {
      int prime = i;
      if (i < num_engines) {prime = primes[i];}
      rand_engines.push_back(std::default_random_engine(prime));
    }

    return rand_engines;
  }
};

#endif
