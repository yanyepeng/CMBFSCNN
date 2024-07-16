# CMBFSCNN: Cosmic Microwave Background Foreground Subtraction with Convolutional Neural Network



CMBFSCNN is  a method for component separation of CMB using CNN (convolutional neural network). It can efficiently remove the foregrounds of CMB temperature and polarization.





## Attribution

If you use this code or  find this code useful in your research,  please cite the following papers:

* CMBFSCNN: Cosmic Microwave Background Polarization Foreground Subtraction with Convolutional Neural Network, Ye-Peng Yan, Si-Yu Li, Guo-Jian Wang, Zirui Zhang, Jun-Qing Xia, 2024, arXiv:[2406.17685](https://arxiv.org/abs/2406.17685)
* Recovering Cosmic Microwave Background Polarization Signals with Machine Learning, Ye-Peng Yan, Guo-Jian Wang, Si-Yu Li, Jun-Qing Xia, 2023, ApJ, 947, 29. [doi:10.3847/1538-4357/acbfb4](https://iopscience.iop.org/article/10.3847/1538-4357/acbfb4)
* Recovering the CMB Signal with Machine Learning, Guo-Jian Wang, Hong-Liang Shi, Ye-Peng Yan, Jun-Qing Xia, Yan-Yun Zhao, Si-Yu Li, and Jun-Feng Li, 2022, ApJS, 260, 13. [doi:10.3847/1538-4365/ac5f4a](https://iopscience.iop.org/article/10.3847/1538-4365/ac5f4a)





## Requirements

Install the following packages:

- [PyTorch](https://pytorch.org/)
- [CUDA](https://developer.nvidia.com/cuda-downloads)
- [PySM](https://github.com/bthorne93/PySM_public): No need to install. It is included in CMBFSCNN.
- [CAMB](https://github.com/cmbant/CAMB)
- [Healpy](https://github.com/healpy/healpy)
- [NaMaster](https://github.com/LSSTDESC/NaMaster)







## Contributors

* Ye-Peng Yan
* Guo-Jian Wang
* Si-Yu Li

## Installing

You can install CMBFSCNN by using:

```
$ git clone https://github.com/yanyepeng/CMBFSCNN
$ cd cmbfscnn
$ python setup.py install
```



## Quick Start

You can directly run the code by using

```
cd examples
python main.py 'config'
```

You can also modify the configuration file according to your needs.



In script `Tutorial.ipynb`, we provide a demo for use. 



The computational process of CMBFSCNN is divided into five steps: 

1) Simulating sky radiation data: We use the [PySM](https://github.com/bthorne93/PySM_public) software package to simulate foreground components, CMB, instrument beam, and white noise. We can simulate a large number of multi-frequency sky maps and divide them into training, validation, and testing sets. In this code, we also provide the computation of NILC noise for CMB polarization, which is smaller than the noise of each frequency band and can reduce the noise level in the CNN output.

2)  Data preprocessing: It involves transforming the healpix sky maps into two-dimensional flat sky maps for CNN processing. 

   ![figure1](images/figure1.png)

3) Establishing and training the CNN model. 

4) Predicting CMB maps from  the polluted sky-maps and calculating the relevant power spectra. 

5) Evaluating the CMBFSCNN method at the sky map and power spectra levels.



