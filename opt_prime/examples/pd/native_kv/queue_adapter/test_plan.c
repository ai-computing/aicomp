#include <assert.h>
#include "coalesced_plan.h"
int main(void){
 unsigned reason=0;
 struct segment s[3]={{128,1024,64},{64,1088,64},{0,1152,64}};
 struct plan *p=make_plan(s,3,64,192,&reason);
 assert(p && p->runs==1 && p->bytes==192 && p->transfers[0].src==0);
 assert(p->ordered[0].dst==1152 && p->ordered[2].dst==1024);
 free_plan(p);
 assert(!make_plan(s,3,64,128,&reason));
 assert(!make_plan(s,3,0,192,&reason));
 return 0;
}
