"""v5 non-codebook control: reversible SH color and/or spatial integer prediction.

All original 59 uint16 attribute codes remain represented in lossless AV1 planes.
This independent family measures whether SH decorrelation can help without a VQ
or exact dictionary. No learned model, retraining, pruning or new quantizer.
"""
from pathlib import Path
import math
import numpy as np
from . import av1_planes as v1
from .common import read_json,write_json

MODES=['spatial','sh_color','sh_color_spatial']


def forward(codes,mode):
    values=codes.copy()
    if mode in ['sh_color','sh_color_spatial']:
        # R,G,B each have 15 AC terms in the standard table.
        values[:,36:51]=codes[:,36:51]-codes[:,21:36]
        values[:,21:36]=codes[:,21:36]-codes[:,6:21]
    if mode in ['spatial','sh_color_spatial']:
        values[1:]=values[1:]-values[:-1]
    return values


def inverse(codes,mode):
    values=codes.copy()
    if mode in ['spatial','sh_color_spatial']:
        values=np.cumsum(values,axis=0,dtype=np.uint64).astype(np.uint16)
    if mode in ['sh_color','sh_color_spatial']:
        values[:,21:36]+=values[:,6:21]
        values[:,36:51]+=values[:,21:36]
    return values


def pack(codes,config):
    side=config['tile_side']
    planes=np.zeros((config['tile_rows']*config['tile_columns'],side*side),np.uint8)
    planes[:118:2,:len(codes)]=(codes&255).T
    planes[1:118:2,:len(codes)]=(codes>>8).T
    return planes.reshape(config['tile_rows'],config['tile_columns'],side,side).transpose(0,2,1,3).reshape(config['height'],config['width'])


def unpack(atlas,count,config):
    side=config['tile_side']
    planes=atlas.reshape(config['tile_rows'],side,config['tile_columns'],side).transpose(0,2,1,3).reshape(-1,side*side)
    return planes[:118:2,:count].T.astype(np.uint16)|(planes[1:118:2,:count].T.astype(np.uint16)<<8)


def encode(paths,timestamps,folder,ffmpeg,threads=1,prediction='sh_color'):
    if threads!=1 or prediction not in MODES:
        raise ValueError('Require one CPU thread and a valid reversible prediction')
    original=v1.atlas_from_table
    def atlas(table,config):
        lo=np.asarray(config['mins'],np.float64)
        hi=np.asarray(config['maxs'],np.float64)
        order=v1.morton_order(table[:,:3],lo[:3],hi[:3])
        values=table[np.ix_(order,v1.COLUMNS)].astype(np.float64)
        codes=np.rint(np.clip((values-lo)/np.where(hi>lo,hi-lo,1),0,1)*65535).astype(np.uint16)
        return pack(forward(codes,prediction),config)
    # Same private-process v1 AV1 transport, with only the reversible plane transform changed.
    # Restore immediately; no source file is modified and no parallel codecs are allowed.
    v1.atlas_from_table=atlas
    try:
        details=v1.encode(paths,timestamps,folder,ffmpeg,threads=1)
    finally:
        v1.atlas_from_table=original
    config=read_json(Path(folder)/'manifest.json')
    config.update({'format':'morton-av1-predictive','version':5,'prediction':prediction,
                   'preserves_v1_quantizer':True,'preserves_v1_morton_order':True,'sh_dictionary':False})
    write_json(Path(folder)/'manifest.json',config)
    return {**details,'algorithm':f'v1 quantizer/Morton, reversible {prediction}, no SH codebook, lossless AV1',
            'prediction':prediction,'sh_dictionary':False,'preserves_v1_quantizer':True}


def decode(folder,output,ffmpeg,threads=1):
    if threads!=1:
        raise ValueError('Require one CPU thread')
    config=read_json(Path(folder)/'manifest.json')
    if config.get('format')!='morton-av1-predictive' or config.get('version')!=5 or config.get('prediction') not in MODES:
        raise ValueError('Unsupported predictive format/version')
    original=v1.table_from_atlas
    def table(atlas,count,config):
        codes=inverse(unpack(atlas,count,config),config['prediction'])
        return original(pack(codes,config),count,config)
    v1.table_from_atlas=table
    try:
        return v1.decode(folder,output,ffmpeg,threads=1)
    finally:
        v1.table_from_atlas=original
