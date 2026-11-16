"""Compatibility entry point; the canonical implementation is /av1_codec.py."""
import sys

if __name__=='__main__':
    # Execute the root CLI so its pre-NumPy thread limits are applied as well.
    import runpy
    from pathlib import Path
    runpy.run_path(str(Path(__file__).resolve().parents[1]/'av1_codec.py'),run_name='__main__')
else:
    import av1_codec as _implementation
    from av1_codec import *
    # Normal imports refer to the same module, including instrumentation globals.
    sys.modules[__name__]=_implementation
