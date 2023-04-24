# Exposure Correction Model to Enhance Image Quality

[F. Irem Eyiokur](https://github.com/iremeyiokur)<sup>1,</sup>\*, Dogucan Yaman<sup>1,</sup>\*, [Hazım Kemal Ekenel](https://web.itu.edu.tr/ekenel/)<sup>2</sup>, [Alexander Waibel](https://isl.anthropomatik.kit.edu/english/21_74.php)<sup>1,3</sup>

\*Equal contribution.

<sup>1</sup>Karlsruhe Institute of Technology, <sup>2</sup>Istanbul Technical University, <sup>3</sup>Carnegie Mellon University

This paper is published in [CVPR 2022](https://cvpr2022.thecvf.com/) [NTIRE Workshop](https://data.vision.ee.ethz.ch/cvl/ntire22/). If you use our code or the paper is helpful for your research, please cite our paper.

### TODO ###

- Demos

## Prerequests ##

## Datasets ##

### Exposure correction dataset ###
In order to download the corresponding exposure correction dataset [1], please visit [this link.](https://github.com/mahmoudnafifi/Exposure_Correction#dataset) 

Test setup   | Number of images
:------ | :--------------:
Well- and overexposed | 3543
Underexposed | 2362
Altogether | 5905

### Low-light image datasets ###
We utilized four different low-light image datasets to test and evaluate our model and compare with other works.

Dataset   | Number of images
:------- | :--------------:
LIME [2] | 10 
NPE [3]  | 75
VV [4]   | 24
DICM [5] | 44

You can download all these datasets from [this link.](https://daooshee.github.io/BMVC2018website/) or from [this link](https://github.com/VITA-Group/EnlightenGAN). We followed the literature and used the same setup.

## Demo ##

## Training ##

```python
CUDA_VISIBLE_DEVICES="0" python train.py --name experiment_name --checkpoints_dir checkpoints --batchSize 32 --loadSize 256 --fineSize 256 --label_nc 3 --output_nc 3 --dataroot data_folder_name --no_flip --read_from_file --file_name_A train_input.txt --file_name_B train_gt_full.txt --ngf 64 --n_downsample_global 4 --n_blocks_global 9 --spectral_normalization_D --l1_image_loss --num_D 3 --ndf 64 
```

## Testing ##


```python
CUDA_VISIBLE_DEVICES="0" python test.py --ntest 1000 --results_dir exposure_output --checkpoints_dir checkpoints --name exposure_correction_experiment --loadSize 256 --fineSize 256 --dataroot dataset/exposure/test --dir_A INPUT_Images --no_flip --label_nc 3 --how_many 1000 --phase_test_type test_all --which_epoch 100 --netG global --ngf 64 --n_downsample_global 4 --n_blocks_global 9 --batchSize 1
```

## Evaluation ##

We implemented PSNR, SSIM, and PI metrics using Python. You can also find the original Matlab implementation of PI metric [in this GitHub project.](https://github.com/roimehrez/PIRM2018) Please follow the corresponding instructions to run the original implementation.

Although we did not provide some common metrics for portrait matting such as SAD, Gradient, Connectivity due to space limitation, we observed the same situation that we presented with MSE and MAE. When we change the exposure setting, all these metrics increase. After we correct the exposure setting, portrait matting models work better and all these metrics decrease.

```python
python evaluate_exposure.py --images_path1 ../dataset/exposure_correction/test/expert_a_testing_set --images_path2 predicted_images --metric all
```

## Results ##

HR versions of the presented figures in the paper are available under *images* folder.

![Results1](/images/comparison3.png)

## Application - Portrait Matting ##

We also tested the effect of our exposure correction model on portrait matting task. For this, we utilized four real-world portrait matting dataset and manipulated the exposure settings of these images by using Adobe Photoshop Lightroom. As in exposure correction dataset, we used -1.5, -1, +1, +1.5 EVs. We also utilized -2.5, -2, +2, +2.5 EVs to perform further test to evaluate the generalization performance of the exposure correction model. 

Dataset   | Number of images
:------- | :--------------:
[PPM-100](https://github.com/ZHKKKe/PPM) [6] | 100
[P3M500](https://github.com/JizhiziLi/P3M) [7]  | 500
[RWP636](https://github.com/yucornetto/MGMatting) [8]  | 636
[AIM500](https://github.com/JizhiziLi/AIM) [9] \*  | 100

\*Please note that AIM500 dataset has 500 images, however, this dataset is proposed for image matting. Since we performed portrait matting test, we evaluated the images that are not suitable for the portrait matting (e.g., images without person). In the end, we have 100 images.

We also employed three different state-of-the-art portrait matting models that do not need additiona input. These models are [MODNet](https://github.com/ZHKKKe/MODNet) [6], [MGMatting](https://github.com/yucornetto/MGMatting) [8], and [GFM](https://github.com/JizhiziLi/GFM) [10].

![portrait matting](/images/matting.png)

## Acknowledgement

This code is based on [Pix2PixHD](https://github.com/NVIDIA/pix2pixHD) implementation. We would like to thanks the authors for sharing their valuable codes.

## References ##

[1] Afifi, Mahmoud, et al. "Learning multi-scale photo exposure correction." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2021.

[2] Xiaojie Guo, Yu Li, and Haibin Ling. Lime: Low-light image enhancement via illumination map estimation. IEEE Transactions on image processing, 26(2):982–993, 2016.

[3] Shuhang Wang, Jin Zheng, Hai-Miao Hu, and Bo Li. Naturalness preserved enhancement algorithm for non-uniform illumination images. IEEE transactions on image processing, 22(9):3538–3548, 2013.

[4] Vassilios Vonikakis. Busting image enhancement and tonemapping algorithms

[5] Chulwoo Lee, Chul Lee, and Chang-Su Kim. Contrast enhancement based on layered difference representation. In 2012 19th IEEE international conference on image processing, pages 965–968. IEEE, 2012.

[6] Ke, Zhanghan, et al. "MODNet: Real-Time Trimap-Free Portrait Matting via Objective Decomposition." AAAI, 2022.

[7] Li, Jizhizi, et al. "Privacy-preserving portrait matting." Proceedings of the 29th ACM International Conference on Multimedia. 2021.

[8] Yu, Qihang, et al. "Mask guided matting via progressive refinement network." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2021.

[9] Li, Jizhizi, Jing Zhang, and Dacheng Tao. "Deep automatic natural image matting." arXiv preprint arXiv:2107.07235 (2021).

[10] Li, Jizhizi, et al. "Bridging composite and real: towards end-to-end deep image matting." International Journal of Computer Vision (2022): 1-21.
