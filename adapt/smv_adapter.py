"""Local SMV payload prototype, with explicit timeline (not original site playback)."""
from pathlib import Path
import tempfile
from .common import write_json,save_frame,save_sequence
import smv_codec


def encode(paths,timestamps,folder,ffmpeg,threads=1):
    folder=Path(folder)
    folder.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='smv-input-',dir=folder.parent) as temp:
        sequence=Path(temp)/'sequence.json'
        write_json(sequence,{'frames':[{'file':str(p),'timestamp':t} for p,t in zip(paths,timestamps)]})
        stats=smv_codec.encode(sequence,folder/'scene.smv.zip',ffmpeg,threads=threads)
    # These reports are benchmark diagnostics, not bytes required to play the archive.
    (folder/'scene.smv.zip.report.json').replace(folder.parent/'native_encode_report.json')
    return {'sh_degree':3,'threads':threads,'implementation':'local SMV prototype: NPZ + lossless HEVC + explicit GOP timeline',
            'original_website_direct_playback':False,'native_stats':stats}


def decode(folder,output,ffmpeg,threads=1):
    timestamps=[]
    for i,timestamp,table in smv_codec.decoded_frames(Path(folder)/'scene.smv.zip',ffmpeg,threads=threads):
        save_frame(output,i,table)
        timestamps.append(timestamp)
    save_sequence(output,timestamps)
    return {'frame_count':len(timestamps),'sh_degree':3,'threads':threads}
