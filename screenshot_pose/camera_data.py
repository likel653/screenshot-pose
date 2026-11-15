"""Image-bound perspective intrinsics and shared projection conventions."""
import hashlib,math
import numpy as np


def pixel_digest(rgb):
    a=np.asarray(rgb,dtype=np.uint8)
    if a.ndim!=3 or a.shape[2]!=3:raise ValueError('相机标定需要 RGB 图片')
    return hashlib.sha256(str(a.shape).encode()+a.tobytes()).hexdigest()


def intrinsics(calibration,width,height):
    if calibration.get('schema')!='screenshot_pose/camera_v1' or not calibration.get('valid'):
        raise ValueError('没有有效的相机标定，请先标定当前图片')
    if calibration.get('image_size')!=[width,height]:raise ValueError('相机标定与截图尺寸不匹配')
    a=np.asarray(calibration.get('intrinsics'),float)
    if a.shape!=(4,) or not np.isfinite(a).all():raise ValueError('相机内参无效')
    fx,fy,cx,cy=a
    if not (.1*width<fx<15*width and .1*height<fy<15*height and -.5*width<cx<1.5*width and -.5*height<cy<1.5*height):
        raise ValueError('相机焦距或主点超出可用范围')
    return a


def project(points,camera):
    """Image units = pixels / width, axes X right / Y away / Z up."""
    p=np.asarray(points,float)@np.asarray(camera['rotation'],float).T
    if camera.get('type')=='PERSPECTIVE':
        p=p+np.asarray(camera['translation'],float)
        if np.min(p[:,1])<=1e-5:raise ValueError('透视求解把关节放在相机后方')
        fx,fy,cx,cy=camera['normalized_intrinsics']
        return np.column_stack((fx*p[:,0]/p[:,1]+cx,-fy*p[:,2]/p[:,1]+cy))
    return p[:,[0,2]]*[1.,-1.]*camera['scale']+np.asarray(camera['shift'])


def error(predicted,observed,weights):
    weights=np.asarray(weights,float)
    return float(np.sqrt(np.sum((np.asarray(predicted)-observed)**2*weights[:,None])/max(weights.sum(),1e-10)))
