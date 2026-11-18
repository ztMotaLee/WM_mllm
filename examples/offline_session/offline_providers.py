"""Explicit deterministic offline demo providers: not a model, never a live selector."""
from __future__ import annotations
import hashlib,json
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
from wmmllm.providers.types import ChatResponse,ImageEditResult,CallReceipt,AttemptReceipt

def receipt(kind,role,payload):
    digest=hashlib.sha256(json.dumps(payload,sort_keys=True,default=str).encode()).hexdigest()
    return CallReceipt(kind,role,'offline-procedural-not-a-model','offline-procedural-not-a-model','known',None,'unknown','complete','offline',0.,digest,digest,(),(AttemptReceipt(1,'complete',0.),))

class OfflineProceduralChat:
    """Not a model. Fixed fixture decisions exercise the real engine's routes."""
    route='offline-procedural-not-a-model'
    def __init__(self):self.requests=[];self.verifier_values=[];self.observer_errors=[]
    def complete(self,request):
        self.requests.append(request)
        name=request.response_format['json_schema']['name'];payload=request.payload
        caption=payload.get('current_caption',payload.get('instruction',''))
        if name=='pano_scene_bootstrap':
            value={'scene_description':'A procedural tiled room with geometric marks.','style_tokens':'Flat geometric colors','entity_identity_properties':'Repeated squares and colored circles'}
        elif name=='reachable_camera_schedule':
            yaw=0.;path=[]
            for item in payload['instructions']:
                caption=item['caption'].lower()
                if 'right' in caption:yaw=45.
                if 'back' in caption or 'initial' in caption:yaw=0.
                path.append({'segment':item['segment'],'yaw':yaw,'pitch':0.})
            value={'camera_path':path}
        elif name=='reachable_seam_verdict':
            value={'verdict':'PASS','visible_seam':False,'duplicated_object':False,'cut_object':False,'detail':'Offline procedural fixture geometry.','repair_views':[]}
        elif name=='aiwm_operation_plan':
            pose=dict(payload['current_camera']);text=caption.lower()
            local='paint' in text
            if 'right' in text:pose['yaw']=45.
            if 'back' in text or 'initial' in text:pose['yaw']=0.
            operation='local_edit' if local else ('revisit' if 'back' in text else 'rotate')
            value={'operations':[{'operation':operation,'targets':['yellow marker'] if local else [],'scope':'local' if local else 'camera_only','instruction':caption}],
                'ending_camera':pose,'editing_camera':dict(payload['current_camera']),'editing_hfov':90.,'restore_node':'','physical_state_changed':local,
                'endpoint_instruction':'Paint a yellow square on the middle tile.' if local else 'Show the room at the ending camera orientation.','decision_reason':'Fixed offline fixture route; not a model.'}
        elif name in ('source_grounded_transition_action','visible_state_remaining_action'):
            value={'current_state':'A tiled room with geometric marks.','remaining_action':caption,'endpoint_instruction':'Paint a yellow square on the middle tile.',
                'scope_bbox':[.32,.25,.68,.75],'support_polygons':[[[.35,.30],[.65,.30],[.65,.70],[.35,.70]]],
                'final_view_entity_bboxes':[{'entity':'yellow marker','bbox':[.42,.40,.58,.60]}]}
        elif name=='actual_generated_edit_support':
            value={'support_polygons':[[[.35,.30],[.65,.30],[.65,.70],[.35,.70]]]}
        elif name=='endpoint_semantic_verification':
            status=self.verifier_values.pop(0) if self.verifier_values else 'pass'
            value={key:status for key in ('state_matches_source','action_preserves_request','endpoint_matches_action','scene_identity_preserved')};value['reason']='Deterministic offline fixture verdict; not a model.'
        elif name=='accepted_state_memory_observation':
            if self.observer_errors:raise self.observer_errors.pop(0)
            value={'summary':'The procedural room remains visible.','facts':[{'subject_id':'room','label':'room','key':'appearance','kind':'appearance','value':'geometric tiles','evidence':'Visible tiles in both supplied endpoints.','frame_indices':[0,1]}]}
        else:raise ValueError('Unsupported offline fixture schema: '+name)
        return ChatResponse(json.dumps(value),'offline-procedural-not-a-model',receipt('chat',request.role,value))
    def close(self):pass

class OfflineDrawnImageEdit:
    """Not a model. Draws fixture pixels and fills only explicitly supplied masks."""
    route='offline-drawn-not-a-model'
    def __init__(self):self.requests=[]
    def edit(self,request):
        self.requests.append(request)
        image=Image.open(request.images[0].path).convert('RGB')
        size=request.output_size or ((2048,1024) if request.panorama else image.size)
        if image.size!=size:image=image.resize(size,Image.Resampling.LANCZOS)
        if request.mask is not None:
            mask=Image.open(request.mask.path).convert('L').resize(size,Image.Resampling.NEAREST)
            x=np.arange(size[0])[None,:];y=np.arange(size[1])[:,None]
            fill=np.stack(np.broadcast_arrays(68+(x//80%2)*14,84+(y//80%2)*14,108+(x+y)//100%30),axis=-1).astype(np.uint8)
            pixels=np.asarray(image).copy();pixels[np.asarray(mask)>0]=fill[np.asarray(mask)>0];image=Image.fromarray(pixels)
        else:
            draw=ImageDraw.Draw(image);w,h=image.size
            draw.rectangle((int(w*.42),int(h*.4),int(w*.58),int(h*.6)),fill=(245,205,40))
        output=Path(request.output_dir);output.mkdir(parents=True,exist_ok=True);path=output/'endpoint.png'
        with path.open('xb') as stream:image.save(stream,format='PNG')
        return ImageEditResult(path,receipt('image','editor',{'pixels':hashlib.sha256(np.asarray(image).tobytes()).hexdigest()}))
    def close(self):pass
