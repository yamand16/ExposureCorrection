# Exposure Correction

[F. Irem Eyiokur](https://github.com/iremeyiokur)<sup>1,</sup>\*, Dogucan Yaman<sup>1,</sup>\*, [Hazım Kemal Ekenel](https://web.itu.edu.tr/ekenel/)<sup>2</sup>, [Alexander Waibel](https://isl.anthropomatik.kit.edu/english/21_74.php)<sup>1,3</sup>

\*Equal contribution.

<sup>1</sup>Karlsruhe Institute of Technology, <sup>2</sup>Istanbul Technical University, <sup>3</sup>Carnegie Mellon University

This paper will be published in [CVPR 2022](https://cvpr2022.thecvf.com/) [NTIRE Workshop](https://data.vision.ee.ethz.ch/cvl/ntire22/). If you use our code or the paper is helpful for your research, please cite our paper.

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
We utilized four different low-light image datasets to test and evaluate our model and compare with other works.

Dataset   | Number of images
:------- | :--------------:
LIME [2] | 10 
NPE [3]  | 75
VV [4]   | 24
DICM [5] | 44

You can download all these datasets from [this link.](https://daooshee.github.io/BMVC2018website/) or from [this link](https://github.com/VITA-Group/EnlightenGAN). We followed the literature and used the same setup.

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
