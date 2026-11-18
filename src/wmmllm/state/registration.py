"""CPU registration and strict-support endpoint blending.

Registration returns candidate pixels and diagnostics without committing state.
The blend preserves source pixels outside the caller's binary support exactly.
"""
from __future__ import annotations

import hashlib
import math

import cv2
import numpy as np

from . import panorama, projection

POLICY = {'min_inliers':16, 'max_rmse_pixels':4., 'min_inlier_ratio':.25,
          'min_source_hull_fraction':.08, 'min_source_span_fraction':.35,
          'min_source_grid_cells':4, 'min_unique_inlier_points':12, 'max_ray_condition':20.,
          'min_local_scale':.35, 'max_local_scale':2.8,
          'min_corner_area_fraction':.1, 'max_corner_area_fraction':8.}
SCHEMA = "wmmllm.endpoint_blend.v1"


class RegistrationFailure(ValueError):
    """A rejected fit with its actual CPU measurements retained for provenance."""
    def __init__(self, message, *, stage, candidates, source_shape, target_shape, extra=None):
        super().__init__(message)
        self.diagnostics = {'schema':'wmmllm.registration_failure.v1',
            'stage':stage,'reason':message,'source_shape':list(source_shape),
            'target_shape':list(target_shape),'policy':dict(POLICY),
            'candidates':candidates,'extra':extra or {},'accepted':False,
            'thresholds_relaxed':False,'additional_image_attempts':0}


def _metrics(H,src,dst,inlier,source_shape,target_shape,*,stage):
    sh,sw=source_shape[:2];th,tw=target_shape[:2]
    mask=np.asarray(inlier).ravel().astype(bool)
    out={'feature_matches':len(src),'inliers':int(mask.sum()),
         'inlier_ratio':float(mask.mean()) if len(mask) else 0.,
         'source_to_generated_H':None if H is None else H.tolist()}
    if H is None or not mask.any() or not np.isfinite(H).all() or abs(H[2,2])<1e-12:
        return out|{'passed':False,'legacy_geometry_pass':False,'failed_gates':['no_model']}
    H=np.asarray(H,dtype=np.float64)/H[2,2]
    err=np.linalg.norm(cv2.perspectiveTransform(np.float32(src)[None],H)[0]-dst,axis=1)
    p=src[mask]
    hull=float(cv2.contourArea(cv2.convexHull(np.float32(p))))/(sw*sh)
    span=np.ptp(p,axis=0)/[sw,sh]
    cells=set(map(tuple,np.minimum(np.floor(p/[sw,sh]*3).astype(int),2)))
    unique_source=len(np.unique(np.rint(p/3).astype(int),axis=0))
    unique_target=len(np.unique(np.rint(dst[mask]/3).astype(int),axis=0))
    corners=np.float32([[0,0],[sw-1,0],[sw-1,sh-1],[0,sh-1]])
    warped=cv2.perspectiveTransform(corners[None],H)[0]
    area=float(cv2.contourArea(warped,oriented=True))/(sw*sh)
    den=np.c_[corners,np.ones(4)]@H[2]
    # The true local Jacobian detects reflection/projective poles; det(H[:2,:2]) alone does not.
    jac=[]
    for x,y in np.vstack([corners,[(sw-1)/2,(sh-1)/2]]):
        z=H[2,0]*x+H[2,1]*y+H[2,2];q=H@[x,y,1.]
        if abs(z)<1e-8:
            jac.append(np.array([1e12,0.]));continue
        j=(H[:2,:2]*z-np.outer(q[:2],H[2,:2]))/(z*z)
        jac.append(np.linalg.svd(j,compute_uv=False))
    f=sw/2.;K=np.array([[f,0,(sw-1)/2],[0,-f,(sh-1)/2],[0,0,1.]])
    A=np.linalg.inv(K)@H@K
    rmse=float(np.sqrt(np.mean(err[mask]**2)))
    out.update(inlier_rmse_pixels=rmse,inlier_p95_pixels=float(np.percentile(err[mask],95)),
        source_hull_fraction=hull,source_span_fraction=span.tolist(),source_grid_cells=len(cells),
        unique_inlier_source_points=unique_source,unique_inlier_target_points=unique_target,
        source_corner_area_fraction=area,source_projective_denominators=den.tolist(),
        local_singular_values=np.asarray(jac).tolist(),ray_map_det=float(np.linalg.det(A)),
        ray_map_condition_number=float(np.linalg.cond(A)),source_to_generated_H=H.tolist(),
        mapped_source_corners=warped.tolist(),offline_model_calls=0,visual_acceptance_from_registration=False)
    gates={'inlier_count':mask.sum()>=POLICY['min_inliers'],
        'rmse':rmse<=POLICY['max_rmse_pixels'],
        'inlier_ratio':out['inlier_ratio']>=POLICY['min_inlier_ratio'],
        'spatial_hull':hull>=POLICY['min_source_hull_fraction'],
        'spatial_span':bool(np.all(span>=POLICY['min_source_span_fraction'])),
        'spatial_grid':len(cells)>=POLICY['min_source_grid_cells'],
        'unique_points':min(unique_source,unique_target)>=POLICY['min_unique_inlier_points'],
        'orientation':area>0 and np.all(den>.05) and out['ray_map_det']>0,
        'scale':min(np.asarray(jac).ravel())>=POLICY['min_local_scale'] and max(np.asarray(jac).ravel())<=POLICY['max_local_scale'],
        'corner_area':POLICY['min_corner_area_fraction']<=area<=POLICY['max_corner_area_fraction'],
        'ray_condition':stage!='erp' or out['ray_map_condition_number']<=POLICY['max_ray_condition']}
    legacy_pass=gates['inlier_count'] and gates['rmse'] and gates['orientation'] and gates['ray_condition']
    out.update(gates={k:bool(v) for k,v in gates.items()},legacy_geometry_pass=bool(legacy_pass),passed=all(gates.values()),failed_gates=[k for k,v in gates.items() if not v])
    return out


def _matches(source,target,extractor,ratio,reciprocal):
    gray=lambda x:cv2.cvtColor(x,cv2.COLOR_RGB2GRAY)
    if extractor=='SIFT':e=cv2.SIFT_create(nfeatures=12000);norm=cv2.NORM_L2
    elif extractor=='SIFT_dense':e=cv2.SIFT_create(nfeatures=16000,contrastThreshold=.015,edgeThreshold=15);norm=cv2.NORM_L2
    else:e=cv2.ORB_create(nfeatures=18000,scaleFactor=1.15,nlevels=10,fastThreshold=7);norm=cv2.NORM_HAMMING
    sk,sd=e.detectAndCompute(gray(source),None);tk,td=e.detectAndCompute(gray(target),None)
    if sd is None or td is None:return np.empty((0,2),np.float32),np.empty((0,2),np.float32)
    matcher=cv2.BFMatcher(norm)
    pair=matcher.knnMatch(sd,td,k=2)
    matches=[m for p in pair if len(p)==2 for m,n in [p] if m.distance<ratio*n.distance]
    if reciprocal:
        reverse={m.queryIdx:m.trainIdx for p in matcher.knnMatch(td,sd,k=2) if len(p)==2 for m,n in [p] if m.distance<ratio*n.distance}
        matches=[m for m in matches if reverse.get(m.trainIdx)==m.queryIdx]
    src=np.float32([sk[m.queryIdx].pt for m in matches]).reshape(-1,2)
    dst=np.float32([tk[m.trainIdx].pt for m in matches]).reshape(-1,2)
    return src,dst


def registration_candidates(source,target,*,stage='erp',include_fallbacks=True):
    source,target=panorama.rgb_array(source),panorama.rgb_array(target)
    rows=[]
    recipes=[('SIFT',.72,False)]
    if include_fallbacks:recipes += [('SIFT',.8,True),('SIFT_dense',.8,True),('ORB',.78,True)]
    for extractor,ratio,reciprocal in recipes:
        src,dst=_matches(source,target,extractor,ratio,reciprocal)
        for model in (['homography','affine','similarity'] if include_fallbacks else ['homography']):
            if len(src)<4:H=None;mask=np.zeros(len(src),bool)
            elif model=='homography':
                cv2.setRNGSeed(0)
                if extractor=='SIFT' and not reciprocal:
                    H,mask=cv2.findHomography(src,dst,cv2.RANSAC,4.)
                else:
                    H,mask=cv2.findHomography(src,dst,cv2.RANSAC,4.,maxIters=10000,confidence=.999)
            else:
                fit=cv2.estimateAffine2D if model=='affine' else cv2.estimateAffinePartial2D
                cv2.setRNGSeed(1701)
                a,mask=fit(src,dst,method=cv2.RANSAC,ransacReprojThreshold=4.,maxIters=10000,confidence=.999,refineIters=10)
                H=None if a is None else np.vstack([a,[0,0,1]])
            if mask is None:mask=np.zeros(len(src),bool)
            metrics=_metrics(H,src,dst,mask,source.shape,target.shape,stage=stage)
            metrics.update(extractor=extractor,descriptor_ratio=ratio,reciprocal=reciprocal,model=model,stage=stage,
                method=f'{extractor}_{model}_'+('reciprocal' if reciprocal else 'original'))
            rows.append(metrics)
    return rows


def select_registration(source,target,*,stage='erp'):
    rows=registration_candidates(source,target,stage=stage,include_fallbacks=False)
    incumbent=rows[0] if rows[0].get('legacy_geometry_pass',False) else None
    if incumbent is not None:
        meta=dict(incumbent);meta.update(policy=POLICY,alternatives=rows,fallback_used=False,acceptance='unchanged_legacy_support_with_orientation_and_ray_checks')
        return np.asarray(incumbent['source_to_generated_H']),meta
    rows=registration_candidates(source,target,stage=stage)
    passed=[r for r in rows if r['passed'] and r['reciprocal']]
    if not passed:raise RegistrationFailure('source registration fails v5 distributed support/orientation gates',stage=stage,candidates=rows,source_shape=panorama.rgb_array(source).shape,target_shape=panorama.rgb_array(target).shape)
    # Preserve a valid incumbent; then prefer reciprocal distributed support with a simpler model.
    incumbent=next((r for r in passed if r['extractor']=='SIFT' and not r['reciprocal'] and r['model']=='homography'),None)
    selected=incumbent or max(passed,key=lambda r:(r['inliers']*min(r['source_hull_fraction'],.6),{'similarity':2,'affine':1,'homography':0}[r['model']],-r['inlier_rmse_pixels']))
    meta=dict(selected);meta.update(policy=POLICY,alternatives=rows,fallback_used=selected is not incumbent)
    return np.asarray(selected['source_to_generated_H']),meta


def perspective_outpaint_input(source,*,source_metadata=None):
    """Retain API-compatible /16 canvas and compensate odd content in the crop.

    Odd content is placed with floor/ceil margins differing by one pixel.
    register_native_context centers the final texture on the original principal
    point using matching content/crop parity. No source texel is shifted/scaled.
    """
    g=panorama;full=g.rgb_array(source)
    profile=g.source_profile(source,source_metadata=source_metadata)
    l,t,r,b=profile['content_box'];content=np.ascontiguousarray(full[t:b,l:r]);ch,cw=content.shape[:2]
    width=math.ceil((cw+192)/16)*16;height=math.ceil((full.shape[0]+128)/16)*16
    x0,y0=(width-cw)//2,(height-ch)//2
    canvas=np.full((height,width,3),127,np.uint8);canvas[y0:y0+ch,x0:x0+cw]=content
    mask=np.full((height,width),255,np.uint8);mask[y0:y0+ch,x0:x0+cw]=0
    record={'method':'native_perspective_masked_expansion_parity_compensated_v5',
        'canvas_size':[width,height],'official_content_box':[x0,y0,x0+cw,y0+ch],
        'official_content_size':[cw,ch],'source_profile':profile,'official_hfov_degrees':90.,
        'expanded_hfov_degrees':math.degrees(2*math.atan(width/cw)),
        'focal_length_pixels':cw/2.,'source_pixel_sha256':g.pixel_sha256(full),
        'official_content_pixel_sha256':g.pixel_sha256(content),'padding_is_world_texture':False,
        'mask_rule':'255 editable; 0 preserved',
        'canvas_principal_point':[(width-1)/2,(height-1)/2],
        'official_principal_point':[x0+(cw-1)/2,y0+(ch-1)/2],
        'principal_point_offset_pixels':[(width-cw)%2*-.5,(height-ch)%2*-.5],
        'final_context_rule':'center on official principal point, crop parity matches source'}
    return canvas,mask,record


def _native_warp_and_crop(raw,reference,known,H):
    ys,xs=np.nonzero(known)
    if not len(xs):raise ValueError('native source mask is empty')
    l,t,r,b=int(xs.min()),int(ys.min()),int(xs.max())+1,int(ys.max())+1
    if not np.all(known[t:b,l:r]) or known.sum()!=(b-t)*(r-l):raise ValueError('native source mask must be one rectangle')
    ch,cw=b-t,r-l;oh,ow=reference.shape[:2];rh,rw=raw.shape[:2]
    T=np.array([[1,0,l],[0,1,t],[0,0,1.]],np.float64)
    W=T @ np.linalg.inv(H)
    aligned=cv2.warpPerspective(raw,W,(ow,oh),flags=cv2.INTER_LINEAR,borderValue=(127,127,127))
    covered=cv2.warpPerspective(np.full((rh,rw),255,np.uint8),W,(ow,oh),flags=cv2.INTER_NEAREST,borderValue=0)
    # A bilinear colour sample also needs all interpolation neighbours in-bounds.
    float_covered=cv2.warpPerspective(np.ones((rh,rw),np.float32),W,(ow,oh),flags=cv2.INTER_LINEAR,borderValue=0)
    valid=(covered==255)&(float_covered>=1-1e-6)
    invalid=cv2.integral((~valid).astype(np.uint8))
    maxw=cw+2*min(l,ow-r);maxh=ch+2*min(t,oh-b)
    selected=None
    # Independent height search fixes the fixed-canvas-aspect crop failure.
    # Two-pixel steps exactly retain source principal point and odd/even parity.
    for nw in range(maxw,cw+31,-2):
        left=l-(nw-cw)//2
        for nh in range(maxh,ch+31,-2):
            top=t-(nh-ch)//2
            count=invalid[top+nh,left+nw]-invalid[top,left+nw]-invalid[top+nh,left]+invalid[top,left]
            if count==0:
                if selected is None or nw*nh>selected[2]*selected[3]:selected=(left,top,nw,nh)
                break
    return aligned,valid,W,selected


def _legacy_native_crop(raw,reference,source,valid):
    rawh,raww=raw.shape[:2]
    if raw.shape!=reference.shape or (reference.shape[1]-source.shape[1])%2 or (reference.shape[0]-source.shape[0])%2:return None
    for nw in range(raww,source.shape[1]-1,-16):
        nh=min(rawh,int(math.floor((nw*reference.shape[0]/reference.shape[1])/2)*2))
        left,top=(raww-nw)//2,(rawh-nh)//2
        if nh>=source.shape[0] and np.all(valid[top:top+nh,left:left+nw]):return (left,top,nw,nh)
    return None


def register_native_context(generated,canvas,mask,*,collar_px=32):
    g=panorama;raw,reference=panorama.rgb_array(generated),panorama.rgb_array(canvas);known=np.asarray(mask)==0
    ys,xs=np.nonzero(known)
    if not len(xs):raise ValueError('native source mask is empty')
    l,t,r,b=int(xs.min()),int(ys.min()),int(xs.max())+1,int(ys.max())+1
    source=reference[t:b,l:r]
    # Successful requests keep the original extractor/model/crop and avoid all fallback work.
    candidates=registration_candidates(source,raw,stage='native',include_fallbacks=False)
    c=candidates[0];selected=None;selected_index=None;legacy_crop=None
    if c.get('legacy_geometry_pass',False):
        aligned,valid,W,crop=_native_warp_and_crop(raw,reference,known,np.asarray(c['source_to_generated_H']))
        legacy_crop=_legacy_native_crop(raw,reference,source,valid)
        if legacy_crop is not None:selected=c;selected_index=0;crop=legacy_crop
    if selected is None:
        candidates=registration_candidates(source,raw,stage='native');passed=[];warps={}
        for i,c in enumerate(candidates):
            if not c['passed'] or (i!=0 and not c['reciprocal']):continue
            aa,vv,ww,cc=_native_warp_and_crop(raw,reference,known,np.asarray(c['source_to_generated_H']))
            c['gates']['complete_native_coverage']=cc is not None
            if cc is None:c['passed']=False;c['failed_gates'].append('complete_native_coverage');continue
            ll,tt,nw,nh=cc;c['coverage_crop']=list(cc)
            c['outside_context_pixels']=nw*nh-source.shape[0]*source.shape[1]
            passed.append((i,c));warps[i]=(aa,vv,ww,cc)
        if not passed:raise RegistrationFailure('registered context fails v5 support/orientation/complete coverage gates',stage='native',candidates=candidates,source_shape=source.shape,target_shape=raw.shape)
        incumbent=next(((i,c) for i,c in passed if i==0),None)
        selected_index,selected=incumbent or max(passed,key=lambda ic:(ic[1]['inliers']*min(ic[1]['source_hull_fraction'],.6),ic[1]['outside_context_pixels']))
        aligned,valid,W,crop=warps[selected_index]
    h,w=reference.shape[:2];yy,xx=np.indices((h,w));cx=np.clip(xx,l,r-1);cy=np.clip(yy,t,b-1)
    distance=np.maximum(np.maximum(l-xx,xx-(r-1)),np.maximum(t-yy,yy-(b-1)))
    weight=np.clip(1-distance/max(1,collar_px),0,1)**2
    residual=reference[cy,cx].astype(np.float32)-aligned[cy,cx].astype(np.float32)
    corrected=np.clip(np.rint(aligned.astype(np.float32)+residual*weight[...,None]),0,255).astype(np.uint8)
    corrected[known]=reference[known]
    left,top,nw,nh=crop;final=np.ascontiguousarray(corrected[top:top+nh,left:left+nw])
    metadata=dict(selected);metadata.update(policy=POLICY,alternatives=candidates,fallback_used=selected_index!=0,
        crop_fallback_used=legacy_crop is None,legacy_crop_valid=legacy_crop is not None,
        generated_to_official_canvas_W=W.tolist(),registered_crop=list(crop),native_context_size=[nw,nh],
        context_hfov_degrees=math.degrees(2*math.atan(nw/source.shape[1])),outside_only_colour_collar_pixels=collar_px,
        protected_region_changed_pixels=0,source_exact_after_restoration=bool(np.array_equal(corrected[known],reference[known])),
        complete_native_coverage=True,context_principal_point=[(nw-1)/2,(nh-1)/2],
        protected_returned_error=g.image_error(reference[t:b,l:r],raw[t:b,l:r]) if raw.shape==reference.shape else None)
    return final,metadata


def register_erp_to_source(pano,source,*,source_metadata=None):
    g=panorama;erp=g.rgb_array(pano);full=g.rgb_array(source)
    profile=g.source_profile(source,source_metadata=source_metadata);l,t,r,b=profile['content_box'];official=full[t:b,l:r]
    h,w=official.shape[:2];generated=g._render_content_view(erp,0,0,90,w,h)
    H,meta=select_registration(official,generated,stage='erp')
    f=w/2.;K=np.array([[f,0,(w-1)/2],[0,-f,(h-1)/2],[0,0,1.]],np.float64);A=np.linalg.inv(K)@H@K
    if np.linalg.det(A)<=0 or np.linalg.cond(A)>POLICY['max_ray_condition']:
        raise RegistrationFailure('ERP source camera registration is singular or orientation reversing',stage='erp',candidates=meta['alternatives'],source_shape=official.shape,target_shape=generated.shape,extra={'ray_map_determinant':float(np.linalg.det(A)),'ray_map_condition_number':float(np.linalg.cond(A))})
    ph,pw=erp.shape[:2];output=np.empty_like(erp);geom=projection
    lon=((np.arange(pw,dtype=np.float64)+.5)/pw-.5)*2*math.pi
    for first in range(0,ph,128):
        last=min(ph,first+128);lat=(.5-(np.arange(first,last,dtype=np.float64)+.5)/ph)*math.pi
        dx=np.cos(lat[:,None])*np.sin(lon[None,:]);dy=np.broadcast_to(np.sin(lat[:,None]),dx.shape);dz=np.cos(lat[:,None])*np.cos(lon[None,:])
        directions=np.stack([dx,dy,dz],axis=-1)@A.T;u,v=geom.dir_to_uv(directions)
        samples=g._sample_xy(erp,u*pw-.5,v*ph-.5,wrap_x=True);output[first:last]=np.clip(np.rint(samples),0,255).astype(np.uint8)
    meta.update(method='v5_distributed_front_registration_then_global_ERP_ray_map',official_to_generated_ray_map=A.tolist(),
        ray_map_condition_number=float(np.linalg.cond(A)),original_ERP_pixel_sha256=g.pixel_sha256(erp),registered_ERP_pixel_sha256=g.pixel_sha256(output))
    return output,meta


def _inputs(source, generated, support):
    for label, value in (("source", source), ("generated", generated)):
        if not isinstance(value, np.ndarray) or value.dtype != np.uint8 or value.ndim != 3 or value.shape[2] != 3:
            raise ValueError(label + " must be uint8 HxWx3 RGB")
    if source.shape != generated.shape:
        raise ValueError("Source and generated RGB canvases must have identical native shape")
    if not isinstance(support, np.ndarray) or support.dtype != np.uint8 or support.shape != source.shape[:2]:
        raise ValueError("Support must be uint8 HxW on the native RGB canvas")
    if not set(np.unique(support)).issubset({0, 255}):
        raise ValueError("Support must be binary0/255; soft alpha is computed inward only")
    return source, generated, support > 0


def _pixel_sha(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def _mae(a, b, mask):
    if not np.any(mask): return None
    return float(np.abs(a.astype(np.float32) - b.astype(np.float32))[mask].mean())


def align_background(source_rgb, generated_rgb, support_uint8, *,
                     max_translation_pixels=8.0, max_rotation_degrees=0.25,
                     max_corner_displacement_pixels=12.0,
                     exclusion_margin_pixels=32, iterations=60):
    """Return aligned generated RGB and numeric registration provenance.

    ECC estimates one global Euclidean transform on stable pixels outside a
    dilated entity union. It is rejected if any motion bound is exceeded or if
    source/background RGB error does not improve. The output is for subsequent
    strict-support composition; aligned generated background is never state.
    """
    source, generated, entity = _inputs(source_rgb, generated_rgb, support_uint8)
    if any(not math.isfinite(v) or v < 0 for v in (max_translation_pixels, max_rotation_degrees, max_corner_displacement_pixels)):
        raise ValueError("Registration bounds must be finite and nonnegative")
    if not isinstance(exclusion_margin_pixels, int) or not 0 <= exclusion_margin_pixels <= 256 or not isinstance(iterations, int) or not 1 <= iterations <= 300:
        raise ValueError("Invalid bounded registration margin/iterations")
    height, width = entity.shape
    kernel = np.ones((2 * exclusion_margin_pixels + 1,) * 2, np.uint8)
    stable = cv2.dilate(entity.astype(np.uint8), kernel) == 0
    border = min(20, height // 8, width // 8)
    if border:
        stable[:border] = False; stable[-border:] = False
        stable[:, :border] = False; stable[:, -border:] = False
    identity = np.array([[1., 0., 0.], [0., 1., 0.]], np.float32)
    before = _mae(source, generated, stable)
    metadata = {"schema": SCHEMA, "method": "global_Euclidean_ECC_only_no_flow",
                "status": "IDENTITY_RETAINED", "reason": None,
                "template_to_generated_matrix_2x3": identity.tolist(),
                "stable_background_pixels": int(stable.sum()),
                "background_mask_sha256": _pixel_sha(stable.astype(np.uint8)),
                "exclusion_margin_pixels": exclusion_margin_pixels,
                "background_RGB_MAE_before": before, "background_RGB_MAE_after": before,
                "bounds": {"translation_pixels": max_translation_pixels,
                           "rotation_degrees": max_rotation_degrees,
                           "corner_displacement_pixels": max_corner_displacement_pixels},
                "warp_convention": "ECC maps source(template) to generated(input); warpAffine WARP_INVERSE_MAP samples generated into source coordinates.",
                "registration_does_not_authorize_background_edits": True}
    if stable.sum() < max(1024, width * height * .05):
        metadata["reason"] = "Insufficient stable background; no camera warp"
        return generated.copy(), metadata
    # Half-resolution ECC is a CPU diagnostic; map its one rigid estimate back
    # to the exact native coordinate grid before checking all native bounds.
    size = (max(16, width // 2), max(16, height // 2))
    gray_source = cv2.cvtColor(source, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255
    gray_generated = cv2.cvtColor(generated, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255
    small_source = cv2.resize(gray_source, size, interpolation=cv2.INTER_AREA)
    small_generated = cv2.resize(gray_generated, size, interpolation=cv2.INTER_AREA)
    small_stable = (cv2.resize(stable.astype(np.float32), size, interpolation=cv2.INTER_AREA) >= .999).astype(np.uint8) * 255
    try:
        correlation, small_warp = cv2.findTransformECC(small_source, small_generated,
            identity.copy(), cv2.MOTION_EUCLIDEAN,
            (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, iterations, 1e-6),
            inputMask=small_stable, gaussFiltSize=5)
    except cv2.error:
        metadata["reason"] = "ECC did not converge; original generated RGB retained"
        return generated.copy(), metadata
    scaling = np.diag([size[0] / width, size[1] / height, 1.])
    transform = np.eye(3); transform[:2] = small_warp
    warp = (np.linalg.inv(scaling) @ transform @ scaling)[:2].astype(np.float32)
    angle = math.degrees(math.atan2(float(warp[1, 0]), float(warp[0, 0])))
    translation = math.hypot(float(warp[0, 2]), float(warp[1, 2]))
    corners = np.array([[0, 0, 1], [width - 1, 0, 1], [0, height - 1, 1], [width - 1, height - 1, 1]], np.float32)
    displacement = float(np.linalg.norm(corners @ warp.T - corners[:, :2], axis=1).max())
    metadata.update(estimated_matrix_2x3=warp.tolist(), ECC_correlation=float(correlation),
                    estimated_rotation_degrees=angle, estimated_translation_pixels=translation,
                    estimated_max_corner_displacement_pixels=displacement)
    if not np.isfinite(warp).all() or translation > max_translation_pixels or abs(angle) > max_rotation_degrees or displacement > max_corner_displacement_pixels:
        metadata["reason"] = "Estimated camera motion exceeds a bound; no warp applied"
        return generated.copy(), metadata
    aligned = cv2.warpAffine(generated, warp, (width, height),
                            flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                            borderMode=cv2.BORDER_REFLECT_101)
    after = _mae(source, aligned, stable)
    metadata["estimated_background_RGB_MAE_after"] = after
    if after is None or before is None or after > before - max(.05, before * .005):
        metadata["reason"] = "Stable background did not improve sufficiently; no warp applied"
        return generated.copy(), metadata
    metadata.update(status="WARP_APPLIED", reason="Bounded rigid registration reduces stable-background RGB error",
                    template_to_generated_matrix_2x3=warp.tolist(), background_RGB_MAE_after=after,
                    interpolation="bilinear global geometry; no per-object deformation")
    return aligned, metadata


def _boundary_gradient(image, support):
    total = count = 0
    for axis in (0, 1):
        if axis == 0: a, b, boundary = image[:-1], image[1:], support[:-1] != support[1:]
        else: a, b, boundary = image[:, :-1], image[:, 1:], support[:, :-1] != support[:, 1:]
        if np.any(boundary):
            values = np.abs(a.astype(np.float32) - b.astype(np.float32))[boundary]
            total += float(values.sum()); count += values.size
    return total / count if count else None


def blend_endpoint(source_rgb, generated_rgb, support_uint8, *, feather_pixels=12, **registration_options):
    """Return a candidate RGB and proof; outside supplied support is exact source.

    The supplied binary mask must already be the complete actual old/new entity
    union. A native margin of at least feather_pixels+1 keeps object interiors
    fully generated. Caller support is never enlarged or clipped to a predicted
    bbox. Identity/interior pixels undergo only the accepted single rigid warp.
    """
    source, generated, entity = _inputs(source_rgb, generated_rgb, support_uint8)
    if not isinstance(feather_pixels, int) or not 0 <= feather_pixels <= 64:
        raise ValueError("feather_pixels must be an integer from0 to64")
    aligned, registration = align_background(source, generated, support_uint8, **registration_options)
    if feather_pixels:
        distance = cv2.distanceTransform(entity.astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
        alpha = np.clip((distance - 1) / feather_pixels, 0, 1)
    else:
        alpha = entity.astype(np.float32)
    alpha[~entity] = 0
    blended = np.rint(source.astype(np.float32) * (1 - alpha[..., None]) + aligned.astype(np.float32) * alpha[..., None]).clip(0, 255).astype(np.uint8)
    blended[~entity] = source[~entity]
    outside_error = np.abs(blended.astype(np.int16) - source.astype(np.int16))[~entity]
    hard = source.copy(); hard[entity] = aligned[entity]
    metadata = {"schema": SCHEMA, "status": "CPU_CANDIDATE_REQUIRES_ROOT_VISUAL_REVIEW",
                "source_RGB_sha256": _pixel_sha(source), "generated_RGB_sha256": _pixel_sha(generated),
                "actual_support_sha256": _pixel_sha(support_uint8), "candidate_RGB_sha256": _pixel_sha(blended),
                "native_size": [source.shape[1], source.shape[0]], "registration": registration,
                "actual_support_pixels": int(entity.sum()), "outside_support_pixels": int((~entity).sum()),
                "outside_changed_pixels": int(np.count_nonzero(np.any(outside_error != 0, axis=-1))),
                "outside_max_RGB_error": int(outside_error.max()) if outside_error.size else 0,
                "outside_exact": bool(not outside_error.size or not outside_error.any()),
                "mask_policy": "Caller-supplied actual old/new entity union, never clipped to predicted bbox; no support inference or expansion by helper.",
                "feather": {"direction": "inward_only", "pixels": feather_pixels,
                            "fully_generated_pixels": int((alpha == 1).sum()),
                            "transition_pixels": int(((alpha > 0) & (alpha < 1)).sum())},
                "boundary_RGB_gradient_hard": _boundary_gradient(hard, entity),
                "boundary_RGB_gradient_candidate": _boundary_gradient(blended, entity),
                "API_calls": 0, "GPU_calls": 0, "optical_flow": False,
                "source_or_generated_inputs_modified": False}
    if not metadata["outside_exact"]: raise RuntimeError("Strict outside-support preservation failed")
    return blended, metadata


def _source_homography(source: np.ndarray, target: np.ndarray):
    """Estimate classical feature registration; no learned image model runs."""
    import cv2
    extractor=cv2.SIFT_create(nfeatures=12000)
    sk,sd=extractor.detectAndCompute(cv2.cvtColor(source,cv2.COLOR_RGB2GRAY),None)
    tk,td=extractor.detectAndCompute(cv2.cvtColor(target,cv2.COLOR_RGB2GRAY),None)
    if sd is None or td is None:raise ValueError('source registration has no feature descriptors')
    pairs=cv2.BFMatcher().knnMatch(sd,td,k=2)
    matches=[m for pair in pairs if len(pair)==2 for m,n in [pair] if m.distance < .72*n.distance]
    if len(matches)<16:raise ValueError('source registration has fewer than16 matches')
    src=np.float32([sk[m.queryIdx].pt for m in matches])
    dst=np.float32([tk[m.trainIdx].pt for m in matches])
    H,inlier=cv2.findHomography(src,dst,cv2.RANSAC,4.0)
    if H is None:raise ValueError('source registration has no homography')
    accepted=inlier[:,0].astype(bool)
    errors=np.linalg.norm(cv2.perspectiveTransform(src[None,...],H)[0]-dst,axis=1)
    rmse=float(np.sqrt(np.mean(errors[accepted]**2)))
    if accepted.sum()<16 or rmse>4:raise ValueError('source registration fails feature support')
    return H,{'method':'classical_SIFT_RANSAC_homography','feature_matches':len(matches),
              'inliers':int(accepted.sum()),'inlier_rmse_pixels':rmse,'source_to_generated_H':H.tolist(),
              'offline_model_calls':0,'visual_acceptance_from_registration':False}
