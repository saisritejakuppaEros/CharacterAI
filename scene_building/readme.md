0. Extract frames (adaptive fps)
1. Location clustering on RAW frames (DINOv2/CLIP embeddings + HDBSCAN)
2. VLM captioning per cluster → verify/merge/split clusters, tag floor/room relationships
3. People removal + video inpainting (per cluster, after clustering is locked)
4. Per-cluster SfM (glomap/COLMAP or DUSt3R for sparse-overlap shots) → camera poses
5. Per-cluster Gaussian Splatting for renderable 3D
6. Identify connective frames (VLM flags shots with stairs/doors/windows implying adjacency)
7. Video-diffusion interpolation (CameraCtrl-style) using edge frames of adjacent clusters as anchors, to synthesize the missing transition
8. Stitch into full navigable 3D set



Task 1 Installations:
pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu118
pip install open_clip_torch hdbscan
utils
pip install streamlit

The clustring looks good on tanu weds manu clip

Task 2
VLM Captioning for video understanding to get to know flow of things
VLM for image understanding
VLM to estimate the shots and then connect with the pictures
Once the flow is done by the llm understanding the scene

Task 3
