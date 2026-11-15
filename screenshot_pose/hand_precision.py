"""Screenshot-only image fidelity and MANO bind-frame conversion."""
from pathlib import Path
import numpy as np
try:from . import orientation_math as om
except ImportError:import orientation_math as om

FINGERS=('thumb','index','middle','ring','pinky')
MANO_BASES=(13,1,4,10,7)


def _rgb(path):
    from PIL import Image,ImageOps
    with Image.open(path) as source:
        oriented=ImageOps.exif_transpose(source)
        if oriented.mode in ('RGBA','LA') or 'transparency' in oriented.info:
            rgba=oriented.convert('RGBA');result=Image.new('RGB',rgba.size,'white');result.paste(rgba,mask=rgba.getchannel('A'))
        else:result=oriented.convert('RGB')
    return result


def image(processed, original=None):
    """Use the original only after EXIF/aspect/thumbnail identity checks."""
    from PIL import Image
    reference=_rgb(processed);chosen=reference;scale=np.ones(2)
    info=dict(source=str(processed),processed_size=list(reference.size),used_size=list(reference.size),
              original_used=False,reason='no_larger_verified_original')
    if original and Path(original).is_file() and Path(original).resolve()!=Path(processed).resolve():
        try:
            candidate=_rgb(original)
            if candidate.width*candidate.height<=reference.width*reference.height*1.1:
                info['reason']='original_has_no_resolution_gain'
            elif abs((candidate.width/candidate.height)/(reference.width/reference.height)-1)>.005:
                info['reason']='original_aspect_mismatch'
            else:
                a=np.asarray(reference.resize((96,96),Image.Resampling.LANCZOS),float)
                b=np.asarray(candidate.resize((96,96),Image.Resampling.LANCZOS),float)
                error=float(np.mean(np.abs(a-b))/255)
                info['identity_thumbnail_mae']=error
                if error>.015:
                    info['reason']='original_pixels_mismatch'
                else:
                    # A matching large background must not hide a changed hand.
                    sampled=candidate.copy();sampled.thumbnail(reference.size,Image.Resampling.LANCZOS)
                    if sampled.size!=reference.size:
                        info['reason']='original_sampling_size_mismatch'
                    else:
                        difference=np.max(np.abs(np.asarray(sampled,float)-np.asarray(reference,float)),axis=2)/255
                        changed=float(np.mean(difference>.08));info['changed_pixel_fraction']=changed
                        if changed>.00005 or float(np.mean(difference))>.006:
                            info['reason']='original_local_pixels_mismatch'
                        else:
                            chosen=candidate;scale=np.array(chosen.size)/reference.size
                            info.update(source=str(original),used_size=list(chosen.size),original_used=True,reason='verified_original')
        except (OSError,ValueError) as exc:info['reason']='original_unreadable: '+str(exc)[:120]
    return np.ascontiguousarray(np.asarray(chosen)[:,:,::-1]),scale,info


def scaled_location(location,scale):
    import copy
    result=copy.deepcopy(location);scale=np.asarray(scale,float)
    for key in ('wrist','other_wrist','center'):result[key]=(np.asarray(result[key])*scale).tolist()
    result['box']=(np.asarray(result['box'])*np.tile(scale,2)).tolist()
    if result.get('refs') is not None:result['refs']=(np.asarray(result['refs'])*scale).tolist()
    if result.get('regional_search'):
        for key,points in result['regional_search'].items():
            if points:result['regional_search'][key]=(np.asarray(points)*scale).tolist()
    result['span']*=float(max(scale));result['forearm_length']=result.get('forearm_length',0)*float(np.mean(scale))
    result['width']*=float(scale[0]);result['height']*=float(scale[1])
    return result


def mano_frames(rest,global_rotations,side):
    """Mirror bind and pose together; all returned rotations remain proper."""
    rest=np.asarray(rest,float);rot=np.asarray(global_rotations,float)
    if rest.shape!=(21,3) or rot.shape!=(16,3,3) or not np.isfinite(rest).all() or not np.isfinite(rot).all():
        raise ValueError('MANO 原生旋转或静置关节无效')
    if side not in ('L','R'):raise ValueError('无效左右手')
    transform=om.C_DOWN@np.diag([1 if side=='R' else -1,1,1])
    points=(rest-rest[0])@transform.T
    source_palm=om.palm_frame(points)
    frames={}
    for fi,(finger,base) in enumerate(zip(FINGERS,MANO_BASES)):
        for segment in range(3):
            point=1+4*fi+segment;joint=base+segment
            frames[f'{finger}:{segment+1}']=dict(rest_head=points[point].tolist(),rest_tail=points[point+1].tolist(),
                rest_rotation=np.eye(3).tolist(),pose_rotation=om.proper_rotation(transform@rot[joint]@transform.T).tolist())
    return dict(native_source='MANO',source_rest_palm=source_palm.tolist(),bone_frames=frames)
