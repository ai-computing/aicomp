"""Test-only notification timing control; source compute and SHM writes unchanged."""
def deferred(original,engine,request,slab,publish):
 messages=[]
 logits=original(engine,request,slab,messages.append)
 for message in messages:publish(message)
 return logits

def self_check():
 trace=[]
 def source(engine,request,slab,publish):
  for i in range(3):trace.append(('copy',i));publish({'i':i})
  trace.append(('logits',None));return 42
 result=deferred(source,None,None,None,lambda m:trace.append(('notify',m['i'])))
 assert result==42 and trace==[('copy',0),('copy',1),('copy',2),('logits',None),('notify',0),('notify',1),('notify',2)]
 if __name__=='__main__':print('deferred control: copies precede logits; notification order preserved after logits')
self_check()
