"""Fixed full-point screening frames spanning motion and topology boundaries."""
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
sequence=json.loads((ROOT/'benchmark/input_first60.json').read_text(encoding='utf-8-sig'))
indices=[0,1,26,27,28,51,52,53,58,59]
output=Path(__file__).parent/'screen_sequence.json'
if output.exists():
    raise FileExistsError(output)
frames=[]
for i in indices:
    frame=sequence['frames'][i]
    frames.append({'file':str((ROOT/'benchmark'/frame['file']).resolve()),'timestamp':frame['timestamp']})
output.write_text(json.dumps({'purpose':'screening only, not a final 60-frame quality result','source_frame_indices':indices,'frames':frames},indent=2)+'\n',encoding='utf-8')
print(output)
