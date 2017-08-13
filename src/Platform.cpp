#include <iostream>
#include <cstdlib>
#include <cmath>
#include <sys/types.h>
#include <signal.h>
#include <unistd.h>
#include <stacktrace.h>
#include "Platform.h"

bool g_monitor_new = false;

#ifdef USE_OPENMP
#include <omp.h>
#endif

using namespace std;

// Uncomment to report invokations of the new operator.
// Used to make sure that no memory allocations (via new) occur in tight loops.
/*
void* operator new (std::size_t size) {
  if(g_monitor_new) {
    LOG("Memory Allocation: " << size);
    //print_stacktrace();
  }
  
  return malloc(size);
}

void* operator new (std::size_t size, const std::nothrow_t& nothrow_value) noexcept {
  if(g_monitor_new) {
    LOG("Memory Allocation: " << size);
    //print_stacktrace();
  }

  return malloc(size);
}
*/

void handleSIGSEV(int sig) {
  print_stacktrace();
  exit(1);
}

void Platform::init() {
  //Reigster segmentation fault handler
  signal(SIGSEGV, handleSIGSEV);
  signal(SIGABRT, handleSIGSEV);
}

int Platform::getNumLocalThreads() {
#ifdef USE_OPENMP
  return omp_get_max_threads();
#else
  return 1;
#endif
}

void Platform::setNumLocalThreads(int n) {
#ifdef USE_OPENMP
  omp_set_num_threads(n);
#endif
}

int Platform::getThreadId() {
#ifdef USE_OPENMP
  return omp_get_thread_num();
#else
  return 0;
#endif
}

void Platform::sleepCurrentThread(int microseconds) {
  usleep(microseconds);
}

void Platform::waitForDebugger() {
  bool dbg = false; //Set to true to indicate attachment
  cerr << "Process " << getpid() << " is ready for attach" << endl;

  while(!dbg) {sleep(1);}
}

void Platform::measureTime(
    void (*function)(), int num_trials, double &mean, double &std_dev) {

  double sum = 0.0;
  double sum2 = 0.0;
  
  for (int i = 0; i < num_trials; ++i) {
    auto t_start = Platform::getCurrentTime();
    function();
    auto t_end = Platform::getCurrentTime();

    double t = Platform::getDurationms(t_start, t_end);
    sum += t;
    sum2 += t * t;
  }

  mean = sum / num_trials;
  std_dev = sqrt(sum2 / num_trials - mean * mean);
}
