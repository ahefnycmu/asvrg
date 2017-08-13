#include "SVRGSolver.h"

#include <atomic>
#include <cmath>
#include "VectorUtils.h"
#include "SpinLock.h"

SVRGSolver::Solution SVRGSolver::solve(Oracle<SVRGParamVector, SparseVec> *oracle) {   
  Solution solution;
  SpinLock param_lock;
  std::atomic<unsigned long long> iteration_ctr(1);

  bool use_param_lock = (options_.parallel_mode == ParallelMode::LOCKED);
  bool use_atomic_add = (options_.parallel_mode == ParallelMode::LOCK_FREE);
  
  int n = oracle->getNumInstances();
  int d = oracle->getDimension();
  double avg_gradient_multiple = 0.0;

  int num_updates_per_epoch = n * options_.num_nupdates_per_epoch;
  if(num_updates_per_epoch < 0) {
    num_updates_per_epoch
        = static_cast<int>(n / -options_.num_nupdates_per_epoch + 0.5);
  }

  double objective = 0.0;
  Vector x(d);
  Vector x_last_epoch(d); 
  Vector avg_gradient(d);

  int epoch = 0;
  bool done = false;

  int num_threads = Platform::getNumLocalThreads();
  auto rand_engines = createRandomEngines(num_threads);

  long long timeus = 0;

  g_monitor_new = true;
  
  do {        
    avg_gradient_multiple = 0.0;

    Platform::Time epoch_start_time = Platform::getCurrentTime();
    Platform::Time epoch_end_time;
      
    #pragma omp parallel 
    {
      int thread_id = Platform::getThreadId();
      std::default_random_engine &r = rand_engines[thread_id];
      int data_start = 0;
      int data_end = n;
                  
      std::uniform_int_distribution<int> u(data_start, data_end-1);
      SVRGParamVector param_spec;
      param_spec.avg_gradient = &avg_gradient;

      SparseVec g(d); // Gradient at x
      SparseVec g2(d); // Gradient at x_last_epoch
      
      #pragma omp for schedule(static) 
      for(int i = 0; i < num_updates_per_epoch; ++i) {
        // Select instance j at random
        int j = u(r);
       
        // Compute gradients        
        param_spec.x = &x;
        param_spec.avg_gradient_multiple = avg_gradient_multiple;
        oracle->computeGradient(param_spec, j, g);

        if(epoch > 0) {
          // Compute gradient difference w.r.t last epoch
          param_spec.x = &x_last_epoch;
          param_spec.avg_gradient_multiple = 0.0;   
          oracle->computeGradient(param_spec, j, g2);
          VectorUtils::addCompatibleVec(g, 1.0, g2, -1.0);
        }
        
        // Compute step
        double step = options_.step;
        if(options_.alpha_step > 0.0) {
          double t = static_cast<double>(
              iteration_ctr.fetch_add(1,std::memory_order_relaxed));
          step *= sqrt(options_.alpha_step / (t + options_.alpha_step));
        }        

        // Apply update        
        if(use_param_lock) {param_lock.lock();}
        VectorUtils::addVector(x, g, -step, use_atomic_add);

        if(epoch > 0) {
          // Subract average gradient
          if(use_atomic_add) {
            Platform::atomicAdd(&avg_gradient_multiple, -step);
          } else {
            avg_gradient_multiple -= step;
          }
        }
        
        if(use_param_lock) {param_lock.unlock();}        
      }

      epoch_end_time = Platform::getCurrentTime();
      
      #pragma omp single
      {
        VectorUtils::addVector(x, 1.0, avg_gradient, avg_gradient_multiple);

        x_last_epoch = x;
        avg_gradient.fill(0.0);            
        objective = 0.0;
      }
            
      //Recompute average gradient and objective
      param_spec.x = &x;
      param_spec.avg_gradient_multiple = 0.0;

      #pragma omp for schedule(dynamic) reduction(+:objective) 
      for(int i = 0; i < n; ++i) {
        double objective_i = oracle->computeObjAndGradient(param_spec, i, g);
        VectorUtils::addVector(avg_gradient, g, 1.0/n, true);
        objective += objective_i;
      }      
    } //end parallel block

    objective /= n;

    // In SVRG, computing the true gradient is part of the algorithm and
    // its time should be measured
    epoch_end_time = Platform::getCurrentTime();

    timeus += Platform::getDurationus(epoch_start_time, epoch_end_time);
    
    solution.trace.push_back(TraceElement());
    auto &trace_element = solution.trace.back();
    trace_element.objective = objective;

    trace_element.timems = timeus / 1000;
    trace_element.other_info["epoch"] = epoch;
    double grad_sq_norm = avg_gradient.dot(avg_gradient);
    trace_element.grad_sq_norm = grad_sq_norm;

    SVRGParamVector eval_x;
    eval_x.x = &x;
    eval_x.avg_gradient = &avg_gradient;
    eval_x.avg_gradient_multiple = 0.0;
    
    oracle->evalParams(eval_x, trace_element.other_info);

    ASSERT(!std::isnan(objective), "Objective is NaN");
    ASSERT(!std::isinf(objective), "Objective is Inf");
    
    done = (++epoch >= options_.max_num_epochs
            && options_.max_num_epochs > 0)
        || objective <= options_.target_objective;

    double last_step = options_.step;
    if(options_.alpha_step > 0.0) {
      double last_t = 1.0 * epoch * num_updates_per_epoch;
      last_step *= sqrt(options_.alpha_step / (last_t + options_.alpha_step));
    }
    
    LOG(epoch << " " << (timeus / 1000)
        << ":" << " obj=" << objective
        << " last_step=" << last_step 
        << " grad_sq_norm=" << grad_sq_norm);    
  }while(!done);

  solution.timems = timeus / 1000;
  solution.x = x;
  solution.objective = objective;
  
  return solution;
}
