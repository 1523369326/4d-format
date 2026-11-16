"""Upstream 4DGS-compression texture math + native decoder + FFV1/H.264.

Adapter changes: uniform atlas size for variable counts, vectorized texture fill,
sequential FFmpeg video decode instead of per-frame seeking. No upstream edits.
"""
from pathlib import Path
import sys
import tempfile
import numpy as np
from .common import ROOT, PROPERTIES, read_json, write_json, save_frame, save_sequence, run_ffmpeg


def modules():
    sys.path.insert(0,str(ROOT/'third_party/4DGS-compression'))
    import ply_texture_encoder as enc
    import ply_texture_decoder as dec
    return enc,dec


def make_encoder(side):
    enc,_=modules()
    class UniformVectorizedEncoder(enc.PlyTextureEncoder):
        def _select_texture_size(self,entry_count):
            return side,side

        def _write_property_group(self,element,property_names,writer,start_index):
            values=np.column_stack([element.data[name] for name in property_names]).astype(np.float32)
            writer._data.reshape(-1,len(property_names))[start_index:start_index+len(values)]=values
            return start_index+len(values)

    return UniformVectorizedEncoder()


def encode(paths,timestamps,folder,ffmpeg,threads=4):
    from plyfile import PlyData
    import cv2
    cv2.setNumThreads(threads)
    enc,_=modules()
    folder=Path(folder)
    folder.mkdir(parents=True,exist_ok=True)
    # Header-only prescan; PlyData memory maps binary vertices.
    counts=[int(PlyData.read(p)['vertex'].count) for p in paths]
    side=int(np.ceil(np.sqrt(max(counts))))
    encoder=make_encoder(side)
    metadata=[]
    with tempfile.TemporaryDirectory(prefix='texture-png-',dir=folder.parent) as temp:
        temp=Path(temp)
        for i,path in enumerate(paths):
            ply=PlyData.read(path)
            encoded=encoder.encode(ply)
            mapping={}
            for group in encoded.groups:
                name=f'FRAME_{i:06d}_{group.name}.png'
                mapping[group.name]=name
                enc.save_texture(temp/name,group.texture,group.min_values,group.max_values,
                                 png_compression=3,bit_depth=group.bit_depth)
            meta=encoded.to_metadata(path.name,mapping)
            metadata.append(meta)
            if (i+1)%10==0:
                print(f'Texture PNG encoded {i+1}/{len(paths)}',flush=True)
        for group in metadata[0]['groups']:
            name=group['name']
            base=['-y','-filter_threads',str(threads),'-threads',str(threads),'-framerate','30','-i',temp/f'FRAME_%06d_{name}.png','-frames:v',len(paths),'-threads',str(threads)]
            if name=='xyz':
                args=base+['-vf','format=gbrp16le','-c:v','ffv1','-level','3','-g','1','-pix_fmt','gbrp16le','-color_range','pc',folder/f'{name}.mkv']
            else:
                args=base+['-c:v','libx264','-preset','medium','-pix_fmt','yuv444p','-crf','18',folder/f'{name}.mp4']
            run_ffmpeg(ffmpeg,args)
        write_json(folder/'manifest.json',{'timestamps':timestamps,'metadata':metadata,'width':side,'height':side,
            'adapter':'uniform size, vectorized fill, native normalization; PNG level 3; FFV1 xyz; H264 crf18 yuv444p',
            'native_source':'third_party/4DGS-compression'})
    return {'sh_degree':3,'implementation':'upstream property groups/quantization/decoder; vectorized fill and uniform atlas adapter',
            'settings':'xyz uint16 + FFV1; remaining attrs uint8 + H264 CRF18 yuv444p medium'}


def decode(folder,output,ffmpeg,threads=4):
    _,dec=modules()
    folder=Path(folder)
    manifest=read_json(folder/'manifest.json')
    w,h=manifest['width'],manifest['height']
    frames=len(manifest['timestamps'])
    decoder=dec.PlyTextureDecoder()
    with tempfile.TemporaryDirectory(prefix='texture-decode-',dir=folder.parent) as temp:
        temp=Path(temp)
        videos={}
        for group in manifest['metadata'][0]['groups']:
            name=group['name']
            video=folder/(f'{name}.mkv' if name=='xyz' else f'{name}.mp4')
            raw=temp/f'{name}.raw'
            pix='bgr48le' if name=='xyz' else 'bgr24'
            run_ffmpeg(ffmpeg,['-y','-filter_threads',str(threads),'-threads',str(threads),'-i',video,'-frames:v',frames,'-f','rawvideo','-pix_fmt',pix,'-threads',str(threads),raw])
            dtype='<u2' if name=='xyz' else 'u1'
            expected=frames*h*w*3*(2 if name=='xyz' else 1)
            if raw.stat().st_size!=expected:
                raise ValueError(f'Wrong video frame count/size for {name}')
            videos[name]=np.memmap(raw,dtype=dtype,mode='r',shape=(frames,h,w,3))
        for i,meta in enumerate(manifest['metadata']):
            textures={}
            for group in meta['groups']:
                values=videos[group['name']][i]
                textures[group['name']]=values[:,:,0] if len(group['properties'])==1 else values
            ply=decoder.decode_from_textures(meta,textures)
            vertex=ply['vertex'].data
            table=np.column_stack([vertex[name] for name in PROPERTIES])
            save_frame(output,i,table)
        # Close Windows mappings before TemporaryDirectory removes raw files.
        for array in videos.values():
            array._mmap.close()
        videos.clear()
    save_sequence(output,manifest['timestamps'])
    return {'frame_count':frames,'sh_degree':3}
