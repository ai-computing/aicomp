import sys,os,json
from furiosa.torch._register import _register
_register()
import rngd_pd.engines.fsdk as sdk
sdk.open_device=lambda *a,**k:(sdk.set_fusion(4),sdk.set_mesh_kind(sdk.MeshKind.Mesh(2)))
from opt_prime.accel.pd import prefill_server as server
from stream265_control import deferred
modes=json.loads(os.environ['STREAM265_MODES']);original=server.stream_result

def controlled(engine,request,slab,publish):
 mode=modes[request['id']//2]
 assert mode in ('early','deferred')
 if mode=='deferred':return deferred(original,engine,request,slab,publish)
 return original(engine,request,slab,publish)
server.stream_result=controlled
sys.argv=['prefill_server',*sys.argv[3:]]
server.main()
