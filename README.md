# Exposure Correction

### TODO ###

1. Additional results
2. Evaluation metrics and Python implementation of PI metric
3. Test code
4. Pretrained model
5. Training code

## Prerequests ##

## Datasets ##

### Exposure correction dataset ###
In order to download the corresponding exposure correction dataset [1], please visit [this link.](https://github.com/mahmoudnafifi/Exposure_Correction#dataset)

### Low-light image datasets ###
We utilized four different low-light image datasets to evaluate our model and compare with other works. You can use the following links to access to these datasets.

1. LIME [2]
2. NPE [3]
3. VV [4]
4. DICM [5]

## Training ##

## Testing ##

## Evaluation ##

We implemented PSNR, SSIM, and PI metrics using Python. You can also find the original Matlab implementation of PI metric [in this GitHub project.](https://github.com/roimehrez/PIRM2018) Please follow the corresponding instructions to run the original implementation.

## Results ##

## References ##

[1] Afifi, Mahmoud, et al. "Learning multi-scale photo exposure correction." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2021.

[2] Xiaojie Guo, Yu Li, and Haibin Ling. Lime: Low-light image enhancement via illumination map estimation. IEEE Transactions on image processing, 26(2):982–993, 2016.

[3] Shuhang Wang, Jin Zheng, Hai-Miao Hu, and Bo Li. Naturalness preserved enhancement algorithm for non-uniform illumination images. IEEE transactions on image processing, 22(9):3538–3548, 2013.

[4] Vassilios Vonikakis. Busting image enhancement and tonemapping algorithms

[5] Chulwoo Lee, Chul Lee, and Chang-Su Kim. Contrast enhancement based on layered difference representation. In 2012 19th IEEE international conference on image processing, pages 965–968. IEEE, 2012.
