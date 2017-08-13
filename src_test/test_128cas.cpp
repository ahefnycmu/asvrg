#include <cmath>
#include <iostream>
using namespace std;

bool CompareAndSwap128(volatile __int128 *p, volatile __int128 *val, __int128 swap) {
  long long *ap = (long long *) val;
  long long *cp = (long long *) &swap;
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

int main(void) {
 __int128 a, b, c;

 a = 15;
 b = 15;

 ((long long *) &a)[1] = 1;
 ((long long *) &b)[1] = 0;
 
 c = 24;
 
 bool flag = CompareAndSwap128(&b, &a, c);

 cout << flag << " " << (long long) a << " " << (long long) b << " " << (long long) c << "\n";
 cout << sizeof(a) << " " << sizeof(b) << " " << sizeof(c) << "\n";

 double dot = 45.0;
 double p = exp(-dot);
 cout << p << endl;
 p = 1.0 / (1.0 + p);
 cout << p << endl;
 cout << log(0.0) << endl;
}
