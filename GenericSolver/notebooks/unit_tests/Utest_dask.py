import pyDaskVector
from dask_util import DaskClient
import pyDaskOperator as DaskOp
import pyOperator as pyOp
import pyVector as  Vec
import numpy as np


hostnames = ["192.168.66.166","192.168.66.66","192.168.66.67"]
client = DaskClient(hostnames)
vec = Vec.vectorIC((200,1))
chunks = (2,1,1)
# vecD = pyDaskVector.DaskVector(client,vector_template=vec,chunks=chunks)

# client.getClient().who_has(vecD.vecDask[0])
# client.getClient().who_has(vecD.vecDask[3])
# client.getClient().who_has(vecD.vecDask[5])
#
# vec1 = Vec.vectorIC((200,1))
# vectors = [vec1.clone(),vec1.clone()]
# vec1D = pyDaskVector.DaskVector(client,vectors=vectors)
#
# vec1D.rand()
# vec1D.norm()
# vecClone = vec1D.clone()
# vec1D.checkSame(vecClone)
# vec1D.checkSame(vec1D)
#
# #Testing getNdArray
# vec1 = Vec.vectorIC((800,500,1)); vec1.rand()
# vec2 = Vec.vectorIC((800,500,1)); vec2.rand()
# vec3 = Vec.vectorIC((800,500,1)); vec3.rand()
# vec2D = pyDaskVector.DaskVector(client,vectors=(vec1,vec2,vec3))
# arr = vec2D.getNdArray()
# vec1.norm()**2+vec2.norm()**2+vec3.norm()**2
# vec2D.norm()**2
#
# #Testing Dask Operator
# vec = Vec.vectorIC((100,1))
# chunks = (20,20,20)
# sc = 10.0
# vecD = pyDaskVector.DaskVector(client,vector_template=vec,chunks=chunks)
# scal_op_args = [(ivec,sc) for ivec in vecD.vecDask]
#
# scaleOp_constr = pyOp.scalingOp
# dask_scaling = DaskOp.DaskOperator(client,scaleOp_constr,scal_op_args,chunks)
#
# vecD1 = vecD.clone()
# vecD.set(1.0)
# dask_scaling.forward(False,vecD,vecD1)
# arr = vecD1.getNdArray()
#
#
# dask_scaling.dotTest(True)


#Testing Spreading/Stacking operator
S = DaskOp.DaskSpreadOp(client,vec,chunks)

try:
	#Testing with GenericIO
	import SepVector
	vecSep = SepVector.getSepVector(ns=[200,200])
	chunks = (1,1,1)
	vecD = pyDaskVector.DaskVector(client,vector_template=vecSep,chunks=chunks)
	vec1 = vecSep.clone(); vec1.rand()
	vec2 = vecSep.clone(); vec2.rand()
	vec3 = vecSep.clone(); vec3.rand()
	vectors = [vec1,vec2,vec3]
	vecD1 = pyDaskVector.DaskVector(client,vectors=vectors,chunks=chunks)
	vec1.norm()**2+vec2.norm()**2+vec3.norm()**2
	vecD1.norm()**2
	chunks = (4,5,2)
	sc = 10.0
	vecD = pyDaskVector.DaskVector(client,vector_template=vecSep,chunks=chunks)
	scal_op_args = [(ivec,sc) for ivec in vecD.vecDask]
	scaleOp_constr = pyOp.scalingOp
	dask_scaling = DaskOp.DaskOperator(client,scaleOp_constr,scal_op_args,chunks)
	vecD1 = vecD.clone()
	vecD.set(1.0)
	dask_scaling.forward(False,vecD,vecD1)
	arr = vecD1.getNdArray()
	dask_scaling.dotTest(True)
except ImportError:
	pass
