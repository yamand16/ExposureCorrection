modelName=("20" "50" "100" "150" "200" "latest")
eOut=("1" "2" "3" "4" "5" "6")
#path=05_05_2021_exposure_illumination_direct_translation2/
path=29_07_2021_exposured_correction_1024/
#for s in ${folderName[@]}; do
#for m in ${modelName[@]}; do

CUDA_VISIBLE_DEVICES="1" python3 test.py --ntest 5905 --results_dir exposure_CVPRW_test --cluster_path test_examples_1_encoded_features.npy --name $path --loadSize 1024 --fineSize 1024 --no_instance --dataroot ../dataset/exposure/test/ --dir_A INPUT_IMAGES --no_flip --label_nc 0 --how_many 5905 --phase_test_type test_all --which_epoch 100 --netG global --ngf 64 --n_downsample_global 4 --n_blocks_global 9 --batchSize 1
