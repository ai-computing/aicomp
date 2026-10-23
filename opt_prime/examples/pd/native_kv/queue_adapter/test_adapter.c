#define ADAPTER_TEST
#include "adapter.c"
#include <assert.h>
static int coalesced,fail;
static size_t last_count;
static int defer;static struct cqe queued[4];static size_t queued_count;
static void *fake_clone(void *dst,const void *src){return memcpy(dst,src,16);}
static void fake_submit(void *ring,void *entries,size_t n){
 (void)ring;
 for(size_t i=0;i<n;i++){
  unsigned char *e=(unsigned char*)entries+64*i;
  uint64_t count=get64(e+48);struct segment *s=(void*)(uintptr_t)get64(e+56);
  last_count=count;coalesced+=(count==1);
  for(size_t j=0;j<count && !fail;j++)memcpy((void*)s[j].dst,(void*)s[j].src,s[j].len);
  struct cqe c={.tag=get64(e+32),.result=fail?-5:0,.flags=0},out;
  if(defer)queued[queued_count++]=c;
  else {completion_hook(&out,&c);assert(out.tag==c.tag && out.result==c.result);}
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
 unsigned char scattered[8*2048],out[8*2048];
 for(size_t i=0;i<sizeof(scattered);i++)scattered[i]=(i/2048)*29+i%197;
 int order[4]={6,2,5,1};struct segment partial[4];
 for(int i=0;i<4;i++)partial[i]=(struct segment){(uintptr_t)(scattered+order[i]*2048),(uintptr_t)(out+i*2048),2048};
 set64(e+48,4);set64(e+56,(uintptr_t)partial);
 submit_hook(NULL,e,1);assert(last_count==2);
 for(int i=0;i<4;i++)assert(!memcmp(out+i*2048,scattered+order[i]*2048,2048));
 assert(stats[3]==0 && stats[7]==0);
 struct segment variable[3]={
  {(uintptr_t)(scattered+5*2048),(uintptr_t)out,4096},
  {(uintptr_t)(scattered+1*2048),(uintptr_t)(out+4096),4096},
  {(uintptr_t)(scattered+3*2048),(uintptr_t)(out+8192),2048}};
 set64(e+48,3);set64(e+56,(uintptr_t)variable);
 submit_hook(NULL,e,1);assert(last_count==2);
 assert(!memcmp(out,scattered+5*2048,4096));
 assert(!memcmp(out+4096,scattered+2048,4096));
 assert(!memcmp(out+8192,scattered+3*2048,2048));
 assert(stats[3]==0 && stats[7]==0);
 set64(e+48,3);set64(e+56,(uintptr_t)seg);
 memcpy(seg,original,sizeof(seg));memset(dst,0,sizeof(dst));fail=1;
 submit_hook(NULL,e,1);
 assert(stats[2]==1 && stats[3]==0);
 for(size_t i=0;i<sizeof(dst);i++)assert(dst[i]==0);
 fail=0;defer=1;
 unsigned char dst_a[3*2048]={0},dst_b[3*2048]={0},batch[128];
 struct segment a[3],b[3];
 for(int i=0;i<3;i++){
  a[i]=(struct segment){(uintptr_t)(src+(2-i)*2048),(uintptr_t)(dst_a+i*2048),2048};
  b[i]=(struct segment){(uintptr_t)(src+(2-i)*2048),(uintptr_t)(dst_b+i*2048),2048};
 }
 memcpy(batch,e,64);memcpy(batch+64,e,64);
 set64(batch+32,901);set64(batch+56,(uintptr_t)a);
 set64(batch+64+32,902);set64(batch+64+56,(uintptr_t)b);
 submit_hook(NULL,batch,2);assert(stats[3]==2 && queued_count==2);
 memset(a,0,sizeof(a));memset(b,0,sizeof(b));
 struct cqe c;
 completion_hook(&c,&queued[1]);completion_hook(&c,&queued[0]);
 assert(stats[3]==0 && stats[7]==0);
 for(int i=0;i<3;i++){
  assert(!memcmp(dst_a+i*2048,src+(2-i)*2048,2048));
  assert(!memcmp(dst_b+i*2048,src+(2-i)*2048,2048));
 }
 puts("PASS partial runs, variable lengths, byte permutation, immutable inputs, fallback, early/failed completion cleanup");
}
