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
struct plan {size_t count,runs,bytes;struct segment *ordered,*transfers;unsigned char *staging;};
static struct pending {int used;uint64_t tag;struct plan *plan;} pending[1024];
/* First five counters retain the original ABI. Remaining: input/output segments,
   in-flight staging bytes, insufficient reduction, unsupported geometry. */
_Atomic uint64_t stats[10];
static const size_t staging_budget=256*1024*1024;
static void free_plan(struct plan *p){if(p){free(p->ordered);free(p->transfers);free(p->staging);free(p);}}
static int compare_source(const void *a,const void *b){
 const struct segment *x=a,*y=b;return (x->src>y->src)-(x->src<y->src);
}
static struct plan *make_plan(const struct segment *s,size_t count){
 size_t bytes=0;
 for(size_t i=0;i<count;i++){
  if(!s[i].len || s[i].len%2048 || s[i].len>4*1024*1024-bytes ||
     s[i].src>UINT64_MAX-s[i].len || s[0].dst>UINT64_MAX-bytes-s[i].len ||
     s[i].dst!=s[0].dst+bytes){stats[9]++;return NULL;}
  bytes+=s[i].len;
 }
 struct plan *p=calloc(1,sizeof(*p));if(!p)return NULL;
 p->count=count;p->bytes=bytes;p->ordered=malloc(count*sizeof(*s));
 if(!p->ordered){free_plan(p);return NULL;}
 memcpy(p->ordered,s,count*sizeof(*s));qsort(p->ordered,count,sizeof(*s),compare_source);
 size_t runs=1;
 for(size_t i=1;i<count;i++){
  uint64_t end=p->ordered[i-1].src+p->ordered[i-1].len;
  if(p->ordered[i].src<end){stats[9]++;free_plan(p);return NULL;}
  if(p->ordered[i].src!=end)runs++;
 }
 /* Require at least 25% fewer driver segments to pay for sorting and host copying. */
 if(runs*4>count*3){stats[8]++;free_plan(p);return NULL;}
 p->transfers=malloc(runs*sizeof(*s));p->staging=malloc(bytes);
 if(!p->transfers || !p->staging){free_plan(p);return NULL;}
 size_t offset=0;
 for(size_t i=0;i<count;i++){
  struct segment x=p->ordered[i];
  if(p->runs && p->transfers[p->runs-1].src+p->transfers[p->runs-1].len==x.src)
   p->transfers[p->runs-1].len+=x.len;
  else p->transfers[p->runs++]=(struct segment){x.src,(uintptr_t)(p->staging+offset),x.len};
  offset+=x.len;
 }
 return p;
}
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
  struct plan *p=make_plan((void*)(uintptr_t)get64(e+56),count);
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
