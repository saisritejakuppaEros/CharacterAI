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
Use the person detection and check hwo much its occupied in the iamge
1. drop the clusters which has more than 75% of the person alone where most of the images
2. keep the images with least person in the images in the csv file(top 5 images) from each cluster


# generated output
cluster 1 - consists of ground floor of the hall
cluster 2 - other side of the ground floor of a room 
cluster 3 - shot of top view of the house
cluster 4-  inside manu room
cluster 5 - shot from isometric conenction
cluster 7 -  manu room
cluster 8 - open hall in ground floor
cluster 10 - open hall in ground floor
cluster 12 - manu room
cluster 13 - manu room
cluster 14,19,20,28 - hall in the ground floor
cluster 15,16,17,18,22,23,24,25,26,29,30,31 - manu room


anchors for the movies and on the fly scene building
1. enter the building u find the wide space - open hall in ground floor images(cluter 8)
2. move towards a corner of the ground hall which has a attached room - get the hall in the ground floor(cluster 2)
3. movie forward u get stairs in with walls left and right and climb to the top - to be hallucinated
4. walk a limited distance on the top - to be hallicanated
5. once walked up move left  to enter manu room - (cluster 15,16,17,18,22,23,24,25,26,29,30,31)