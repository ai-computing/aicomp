#ifndef COALESCED_PLAN_H
#define COALESCED_PLAN_H
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
/* Pure host-side scatter/gather planning. No driver ABI or completion hooks. */
struct segment {uint64_t src,dst,len;};
struct plan {size_t count,runs,bytes;struct segment *ordered,*transfers;unsigned char *staging;};
static void free_plan(struct plan *p){if(p){free(p->ordered);free(p->transfers);free(p->staging);free(p);}}
static int compare_source(const void *a,const void *b){
 const struct segment *x=a,*y=b;return (x->src>y->src)-(x->src<y->src);
}
static struct plan *make_plan(const struct segment *s,size_t count,size_t alignment,size_t max_bytes,unsigned *reason){
 *reason=0;
 if(!count || count>2048 || !alignment || !max_bytes)return NULL;
 size_t bytes=0;
 for(size_t i=0;i<count;i++){
  if(!s[i].len || s[i].len%alignment || s[i].len>max_bytes-bytes ||
     s[i].src>UINT64_MAX-s[i].len || s[0].dst>UINT64_MAX-bytes-s[i].len ||
     s[i].dst!=s[0].dst+bytes){*reason=9;return NULL;}
  bytes+=s[i].len;
 }
 struct plan *p=calloc(1,sizeof(*p));if(!p)return NULL;
 p->count=count;p->bytes=bytes;p->ordered=malloc(count*sizeof(*s));
 if(!p->ordered){free_plan(p);return NULL;}
 memcpy(p->ordered,s,count*sizeof(*s));qsort(p->ordered,count,sizeof(*s),compare_source);
 size_t runs=1;
 for(size_t i=1;i<count;i++){
  uint64_t end=p->ordered[i-1].src+p->ordered[i-1].len;
  if(p->ordered[i].src<end){*reason=9;free_plan(p);return NULL;}
  if(p->ordered[i].src!=end)runs++;
 }
 /* Require at least 25% fewer driver segments to pay for sorting and host copying. */
 if(runs*4>count*3){*reason=8;free_plan(p);return NULL;}
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

#endif
