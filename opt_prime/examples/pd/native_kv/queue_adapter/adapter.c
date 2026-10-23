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
struct segment {uint64_t src,dst,len;};
struct cqe {uint64_t tag;int32_t result;uint32_t flags;};
static void (*real_submit)(void*,void*,size_t);
static void *(*real_clone)(void*,const void*);
static pthread_mutex_t lock=PTHREAD_MUTEX_INITIALIZER;
static struct pending {int used;uint64_t tag;struct segment transfer;} pending[1024];
_Atomic uint64_t stats[5]; /* rewritten, restored, failed, in-flight, capacity/duplicate fallback */
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
 struct cqe c;memcpy(&c,src,sizeof(c));
 pthread_mutex_lock(&lock);
 for(size_t i=0;i<1024;i++)if(pending[i].used && pending[i].tag==c.tag){
  struct segment *s=&pending[i].transfer;
  if(c.result>=0){
   unsigned char temp[2048],*p=(void*)(uintptr_t)s->dst;
   for(size_t a=0,b=s->len-2048;a<b;a+=2048,b-=2048){
    memcpy(temp,p+a,2048);memcpy(p+a,p+b,2048);memcpy(p+b,temp,2048);
   }
   stats[1]++;
  }else stats[2]++;
  pending[i].used=0;stats[3]--;break;
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
  struct segment *s=(void*)(uintptr_t)get64(e+56);
  if(s[0].len!=2048 || s[0].src<(count-1)*2048 || s[0].dst>UINT64_MAX-count*2048)continue;
  int valid=1;
  for(size_t j=1;j<count;j++)if(s[j].len!=2048 || s[j].src!=s[0].src-j*2048 || s[j].dst!=s[0].dst+j*2048){valid=0;break;}
  if(!valid)continue;
  if(!copy){copy=malloc(n*64);if(!copy)break;memcpy(copy,entries,n*64);}
  uint64_t tag=get64(e+32);int slot=-1,duplicate=0;
  pthread_mutex_lock(&lock);
  for(int j=0;j<1024;j++){
   if(!pending[j].used && slot<0)slot=j;
   if(pending[j].used && pending[j].tag==tag)duplicate=1;
  }
  if(slot<0 || duplicate){stats[4]++;pthread_mutex_unlock(&lock);continue;}
  pending[slot].tag=tag;
  pending[slot].transfer=(struct segment){s[count-1].src,s[0].dst,count*2048};
  pending[slot].used=1;stats[0]++;stats[3]++;
  set64(copy+i*64+48,1);set64(copy+i*64+56,(uintptr_t)&pending[slot].transfer);
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
