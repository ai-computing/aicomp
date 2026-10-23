#define ADAPTER_TEST
#include "adapter.c"
#include <assert.h>
static int coalesced,fail;
static void *fake_clone(void *dst,const void *src){return memcpy(dst,src,16);}
static void fake_submit(void *ring,void *entries,size_t n){
 (void)ring;
 for(size_t i=0;i<n;i++){
  unsigned char *e=(unsigned char*)entries+64*i;
  uint64_t count=get64(e+48);struct segment *s=(void*)(uintptr_t)get64(e+56);
  coalesced+=(count==1);
  for(size_t j=0;j<count && !fail;j++)memcpy((void*)s[j].dst,(void*)s[j].src,s[j].len);
  struct cqe c={.tag=get64(e+32),.result=fail?-5:0,.flags=0},out;
  completion_hook(&out,&c);
  assert(out.tag==c.tag && out.result==c.result);
 }
}
int main(void){
 real_submit=fake_submit;real_clone=fake_clone;
 unsigned char src[3*2048],dst[3*2048];
 for(size_t i=0;i<sizeof(src);i++)src[i]=(i/2048)*71+i%251;
 struct segment seg[3];
 for(int i=0;i<3;i++)seg[i]=(struct segment){(uintptr_t)(src+(2-i)*2048),(uintptr_t)(dst+i*2048),2048};
 struct segment original[3];memcpy(original,seg,sizeof(seg));
 unsigned char e[64]={46,8};set64(e+32,123);set64(e+48,3);set64(e+56,(uintptr_t)seg);
 submit_hook(NULL,e,1);
 assert(coalesced==1);
 for(int i=0;i<3;i++)assert(!memcmp(dst+i*2048,src+(2-i)*2048,2048));
 assert(!memcmp(original,seg,sizeof(seg)) && get64(e+48)==3);
 assert(stats[0]==1 && stats[1]==1 && stats[3]==0);
 seg[1].src=(uintptr_t)src;
 submit_hook(NULL,e,1);assert(coalesced==1 && stats[0]==1);
 memcpy(seg,original,sizeof(seg));memset(dst,0,sizeof(dst));fail=1;
 submit_hook(NULL,e,1);
 assert(stats[2]==1 && stats[3]==0);
 for(size_t i=0;i<sizeof(dst);i++)assert(dst[i]==0);
 puts("PASS contiguous rewrite, completion before submit return, full bytes, immutable input, fallback");
}
