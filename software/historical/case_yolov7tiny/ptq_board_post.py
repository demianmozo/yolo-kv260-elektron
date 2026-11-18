"""NumPy translation of pinned YOLOv7 Detect and single-class NMS for ARM."""
import numpy as np
import cv2

def preprocess_image(im):
 h0,w0=im.shape[:2];r=1024/max(h0,w0)
 if r!=1:im=cv2.resize(im,(int(w0*r),int(h0*r)),interpolation=cv2.INTER_AREA if r<1 else cv2.INTER_LINEAR)
 h,w=im.shape[:2];dw=(1024-w)/2;dh=(1024-h)/2
 im=cv2.copyMakeBorder(im,int(round(dh-.1)),int(round(dh+.1)),int(round(dw-.1)),int(round(dw+.1)),cv2.BORDER_CONSTANT,value=(114,114,114))
 return np.ascontiguousarray(im[:,:,::-1].transpose(2,0,1))[None].astype(np.float32)/255

def decode(raw,anchors,strides):
 z=[]
 for i,a in enumerate(raw):
  _,_,h,w=a.shape;x=a.reshape(1,3,6,h,w).transpose(0,1,3,4,2)
  y=1/(1+np.exp(-x));gy,gx=np.meshgrid(np.arange(h,dtype=np.float32),np.arange(w,dtype=np.float32),indexing='ij');grid=np.stack((gx,gy),-1)[None,None]
  y[...,:2]=(y[...,:2]*2-.5+grid)*strides[i];y[...,2:4]=(y[...,2:4]*2)**2*anchors[i].reshape(1,3,1,1,2)
  z.append(y.reshape(-1,6))
 return np.concatenate(z)

def nms(pred,confidence,iou=.6):
 x=pred[pred[:,4]>=confidence];score=x[:,4]*x[:,5];keep=score>=confidence;x=x[keep];score=score[keep]
 boxes=np.concatenate((x[:,:2]-x[:,2:4]/2,x[:,:2]+x[:,2:4]/2),axis=1)
 order=np.argsort(-score,kind='stable')[:30000];chosen=[];area=(boxes[:,2]-boxes[:,0])*(boxes[:,3]-boxes[:,1])
 while len(order) and len(chosen)<300:
  j=order[0];chosen.append(j);rest=order[1:];lt=np.maximum(boxes[j,:2],boxes[rest,:2]);rb=np.minimum(boxes[j,2:],boxes[rest,2:]);wh=np.maximum(rb-lt,0);inter=wh[:,0]*wh[:,1];ov=inter/(area[j]+area[rest]-inter);order=rest[ov<=iou]
 return np.concatenate((boxes[chosen],score[chosen,None],np.zeros((len(chosen),1),np.float32)),axis=1)
