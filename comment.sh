conda activate df_3dgs

###step1 Get Lseg feature
cd lseg_encoder

CUDA_VISIBLE_DEVICES=0  python -u encode_images.py --backbone clip_vitl16_384 --weights checkpoint/demo_e200.ckpt --widehead --no-scaleinv  --outdir ../dataset/room_0/rgb_feature_langseg --test-rgb-dir ../dataset/room_0/train_images --workers 0
###step2 Adaptive Data Compression
cd autoencoder
CUDA_VISIBLE_DEVICES=0  python 0_Adaptive_Data_Compression.py --sence_name  room_0

###step3 Get_index
CUDA_VISIBLE_DEVICES=0  python 1_Get_index.py --sence_name  room_0

### step4 train autoencoder
CUDA_VISIBLE_DEVICES=0  python 2_train.py --sence_name  room_0

### step5 test autoencoder
CUDA_VISIBLE_DEVICES=0  python 3_test.py --sence_name  room_0


### step6 train DF_3DGS
cd DF_3DGS
echo "train" | CUDA_VISIBLE_DEVICES=0  python 4_train.py -s dataset/room_0 -m output/room_0  --speed

### step7 test DF_3DGS

echo "test"  | CUDA_VISIBLE_DEVICES=0  python 5_test.py -s dataset/room_0 -m output/room_0  --speed



