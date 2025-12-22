# Voxtral RunPod Serverless (Pinned, conflict-free)

This build removes `accelerate` and other optional deps to avoid pip resolution conflicts on RunPod.
Diarization (pyannote) is optional and disabled unless you add it yourself.

## Build
Use the included Dockerfile and requirements.txt.

If pip fails during build, the Dockerfile prints the tail of `/tmp/pip_install.log` in the build logs.
