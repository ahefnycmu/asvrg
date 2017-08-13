#include "Platform.h"
#include <iostream>

void f() {
  double x = 5.0;
  double inc = 2.0;
  
  for (int j = 0; j < 1000000; ++j) {
    Platform::atomicAdd(&x, inc);
  }
};

int main() {
  double x = 5.0;
  double y = 3.0;
  double inc = 2.0;

  double m, s;
  Platform::measureTime(&f, 1000, m, s);
  std::cout << m << " " << s << " " << std::endl;
    
  return 0;
}
