#define _GNU_SOURCE
#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <unistd.h>
#include <pthread.h>
#include <sys/mman.h>
#include <stdatomic.h>

/* Experimental ABI: only the SHA256-guarded 2026.4 engine in bootstrap.py. */
#include "coalesced_plan.h"
struct cqe {uint64_t tag;int32_t result;uint32_t flags;};
static void (*real_submit)(void*,void*,size_t);
static void *(*real_clone)(void*,const void*);
static pthread_mutex_t lock=PTHREAD_MUTEX_INITIALIZER;

static struct pending {int used;uint64_t tag;struct plan *plan;} pending[1024];
/* First five counters retain the original ABI. Remaining: input/output segments,
   in-flight staging bytes, insufficient reduction, unsupported geometry. */
_Atomic uint64_t stats[10];
static const size_t staging_budget=256*1024*1024;
static uint64_t get64(const void *p){uint64_t x;memcpy(&x,p,8);return x;}
static void set64(void *p,uint64_t x){memcpy(p,&x,8);}
static int read_channel(int fd){
#ifdef ADAPTER_TEST
 (void)fd;return 1;
#else
 char path[64],target[128];snprintf(path,sizeof(path),"/proc/self/fd/%d",fd);
 ssize_t n=readlink(path,target,sizeof(target)-1);if(n<0)return 0;target[n]=0;
 return !strcmp(target,"/dev/rngd/npu0ch4") || !strcmp(target,"/dev/rngd/npu0ch5") || !strcmp(target,"/dev/rngd/npu0ch6") || !strcmp(target,"/dev/rngd/npu0ch7");
#endif
}
static void *completion_hook(void *dst,const void *src){
 if(!stats[3])return real_clone(dst,src);
 struct cqe c;memcpy(&c,src,sizeof(c));
 pthread_mutex_lock(&lock);
 for(size_t i=0;i<1024;i++)if(pending[i].used && pending[i].tag==c.tag){
  struct plan *p=pending[i].plan;
  if(c.result>=0){
   size_t offset=0;
   for(size_t j=0;j<p->count;j++){
    memcpy((void*)(uintptr_t)p->ordered[j].dst,p->staging+offset,p->ordered[j].len);
    offset+=p->ordered[j].len;
   }
   stats[1]++;
  }else stats[2]++;
  stats[7]-=p->bytes;free_plan(p);pending[i].used=0;stats[3]--;break;
 }
 pthread_mutex_unlock(&lock);
 return real_clone(dst,src);
}
static void submit_hook(void *ring,void *entries,size_t n){
 if(!n || n>2048){real_submit(ring,entries,n);return;}
 unsigned char *copy=NULL;
 for(size_t i=0;i<n;i++){
  unsigned char *e=(unsigned char*)entries+i*64;
  uint32_t op;int fd;memcpy(&op,e+8,4);memcpy(&fd,e+4,4);
  uint64_t count=get64(e+48);
  if(e[0]!=46 || op!=0 || count<2 || count>2048 || !read_channel(fd))continue;
  unsigned reason=0;
  struct plan *p=make_plan((void*)(uintptr_t)get64(e+56),count,2048,4*1024*1024,&reason);
  if(reason)stats[reason]++;
  if(!p)continue;
  if(!copy){copy=malloc(n*64);if(!copy){free_plan(p);break;}memcpy(copy,entries,n*64);}
  uint64_t tag=get64(e+32);int slot=-1,duplicate=0;
  pthread_mutex_lock(&lock);
  for(int j=0;j<1024;j++){
   if(!pending[j].used && slot<0)slot=j;
   if(pending[j].used && pending[j].tag==tag)duplicate=1;
  }
  if(slot<0 || duplicate || stats[7]+p->bytes>staging_budget){
   stats[4]++;pthread_mutex_unlock(&lock);free_plan(p);continue;
  }
  pending[slot].tag=tag;pending[slot].plan=p;pending[slot].used=1;
  stats[0]++;stats[3]++;stats[5]+=count;stats[6]+=p->runs;stats[7]+=p->bytes;
  set64(copy+i*64+48,p->runs);set64(copy+i*64+56,(uintptr_t)p->transfers);
  pthread_mutex_unlock(&lock);
 }
 real_submit(ring,copy?copy:entries,n);
 free(copy);
}
int install_adapter(uintptr_t base){
 /* Install once, after module loading and before constructing any engine/device. */
 void **submit=(void**)(base+0x2de7b68),**clone=(void**)(base+0x2df5620);
 if((uintptr_t)*submit!=base+0x2b0aee0 || (uintptr_t)*clone!=base+0x2b1de00)return -1;
 size_t page=(size_t)sysconf(_SC_PAGESIZE);
 uintptr_t a=(uintptr_t)submit&~(page-1),b=(uintptr_t)clone&~(page-1);
 if(mprotect((void*)a,page,PROT_READ|PROT_WRITE))return -2;
 if(mprotect((void*)b,page,PROT_READ|PROT_WRITE)){mprotect((void*)a,page,PROT_READ);return -3;}
 real_submit=(void*)*submit;real_clone=(void*)*clone;
 *submit=(void*)submit_hook;*clone=(void*)completion_hook;
 if(mprotect((void*)a,page,PROT_READ) || mprotect((void*)b,page,PROT_READ))return -4;
 return 0;
}
