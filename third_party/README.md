# Third-party dependencies

The original projects are Git submodules pinned to the locally verified commits:

| Path | Upstream | Commit |
|---|---|---|
| `4DGS-compression` | https://github.com/X13-A/4DGS-compression | `722a984b4f5dc1bc796d2c7137b92d57d978e35c` |
| `4dgs-converter` | https://github.com/DazaiStudio/4dgs-converter | `0230c3a0a14116609d139a30032abaf3d7a5e67e` |
| `tsog` | https://github.com/OpsiClear-4DGS/tsog | `1813a0a1882208ea6600b8fea2d4e8ad827fde81` |

Use `git submodule update --init --recursive` after cloning. Each upstream retains
its own license and notices. The TSOG dependency is the OpsiClear Python project.

`smv-codec` is a local experimental implementation, tracked directly in this
repository. It is not an upstream checkout. Its archive payload follows the SMV
decoder equations with an explicit timeline extension.
