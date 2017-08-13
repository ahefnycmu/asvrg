#ifndef _RCD_PLATFORM_H_
#define _RCD_PLATFORM_H_

#include <chrono>
#include <cmath>
#include <iostream>
#include <atomic>

extern bool g_monitor_new;

// Encapsulates functions related to parallelism, timing, atomic operations ... etc.
class Platform {
 public:
  typedef std::chrono::time_point<std::chrono::system_clock> Time;

  static void init();
  static int getNumLocalThreads();
  static void setNumLocalThreads(int n);	
  static int getThreadId();
  static void sleepCurrentThread(int microseconds);

  // Used to attach a debugger to the running process.
  // It enters an ifninite loop waiting for the debugger to set
  // local variable 'dbg' to true.
  static void waitForDebugger();

  static Time getCurrentTime() {
    return std::chrono::system_clock::now();
  }

  static int getDurationms(const Time &start, const Time &end) {
    return std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
  }

  static long long getDurationus(const Time &start, const Time &end) {
    return std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
  }

  // Measures the executation time of a function by running it 'num_trials' times
  // and reporting mean and standard deviation.
  static void measureTime(
      void (*function)(), int num_trials, double &mean, double &std_dev);

  
  // Adds a value increment to the variable pointed to by var as an
  // atomic operation.
  inline static void atomicAdd(volatile double *var, double increment) {
    __asm__ __volatile__ (        
        "1: movsd %0,%%xmm0\n\t"
        "movq %%xmm0,%%rax\n\t"
        "addsd %1,%%xmm0\n\t"
        "movq %%xmm0,%%rdx\n\t"
        "lock cmpxchg %%rdx,%0\n\t"
        "jnz 1b\n\t"
        :"+m"(*var)
         ,"+x"(increment)
        :
        :"cc", "xmm0", "rax", "rdx");
  }  

  /*
  inline static bool CompareAndSwap128(
      volatile __int128 *p, volatile __int128 *val, __int128 swap) {
    volatile long long *ap = (volatile long long *) val;
    volatile long long *cp = (volatile long long *) &swap;
    short success_flag = 1;

    __asm__ __volatile__ (
        "lock cmpxchg16b %0\n\t"
        "mov $0,%%cx\n\t"
        "cmovew %%cx,%1"		 
        :"+m"(*p)
         ,"=r"(success_flag)
         ,"+d"(ap[1])
         ,"+a"(ap[0])
         ,"+c"(cp[1])
         ,"+b"(cp[0])
        :
        :"cc");
  
    return success_flag == 0;
  }
  */
};

#endif

