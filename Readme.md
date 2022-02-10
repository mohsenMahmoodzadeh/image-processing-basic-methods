
# Fundamentals of Computer Vision

This project is developed in Python + OpenCV for implementing some techinques of image processing and feature extraction algorithms. 


## Outline
### [01 Image Fundamentals](https://github.com/mohsenMahmoodzadeh/Fundamentals-of-Computer-Vision-projects/blob/master/01%20Image%20Fundamentals.ipynb)
- Geometric Transformations(e.g. rotation, image stitching)
- Histogram Equaliztion
- Interpolation

### [02 Contrast Adjustment](https://github.com/mohsenMahmoodzadeh/Fundamentals-of-Computer-Vision-projects/blob/master/02%20Contrast%20Adjustment.ipynb)
- Image Adjusting
- Histogram Equaliztion
- Local Histogram Equalization(LHE)

### [03 Filters](https://github.com/mohsenMahmoodzadeh/Fundamentals-of-Computer-Vision-projects/blob/master/03%20Filters.ipynb)
- Box Filter
- Median Filter
- Noises(e.g. salt-and-pepper, gaussian)
- Sharpening, Blurring and Noise Removal
- Edge Detection(first-order difference, robert, sobel)
- Unsharp Masking

### [04 Frequency Domain](https://github.com/mohsenMahmoodzadeh/Fundamentals-of-Computer-Vision-projects/blob/master/04%20Frequency%20Domain.ipynb)
- Discrete Fourier Transform(DFT)
- Separable Filters
- Filtering

### [05 Color](https://github.com/mohsenMahmoodzadeh/Fundamentals-of-Computer-Vision-projects/blob/master/05%20Colors.ipynb)
- Color Spaces(e.g. RGB, HSI)
- Quantization
- Clustering

### [06 Wavelet](https://github.com/mohsenMahmoodzadeh/Fundamentals-of-Computer-Vision-projects/blob/master/06%20Wavelet.ipynb)
- Discrete Wavelet Transform(DWT)
- Multi-level DWT
- Approximation & Prediction Residual Pyramids
- Denoising with Wavelet

### [07 Features](https://github.com/mohsenMahmoodzadeh/Fundamentals-of-Computer-Vision-projects/blob/master/07%20Features.ipynb)

- Estimate Geometry
- Corner Detection(with Harris algorithm)


## Environment
- Python: 3.9
- OpenCV: 4.5
- Numpy: 1.22
- Pandas: 1.3
- Matplotlib: 3.4
- Scikit-image: 0.18
- PyWavelets: 1.2
  
## Setup Guide
Clone the repository:

```
git clone https://github.com/mohsenMahmoodzadeh/Fundamentals-of-Computer-Vision-projects.git
```
Create a virtual environement (to avoid conflicts):

```
virtualenv -p python3.9 fcv

# this may vary depending on your shell
. recsys/bin/activate
```

Install the dependencies:
```
pip install -r requirements.txt
```

If you want to work on jupyter notebook, you may need to setup a kernel on your virtual environment to make sure all your modules execute correctly.
```
python -m ipykernel install --name fcvkernel

# Now you get a kernel named `fcvkernel` in your jupyter notebook
```  

## Future Works
- Correct and improve the image registration algorithm in [01 Image Fundamentals.ipynb](https://github.com/mohsenMahmoodzadeh/Fundamentals-of-Computer-Vision-projects/blob/master/01%20Image%20Fundamentals.ipynb) notebook.

- Improve the algoirthms in [02 Contrast Adjustment.ipynb](https://github.com/mohsenMahmoodzadeh/Fundamentals-of-Computer-Vision-projects/blob/master/02%20Contrast%20Adjustment.ipynb) and [03 Filter.ipynb](https://github.com/mohsenMahmoodzadeh/Fundamentals-of-Computer-Vision-projects/blob/master/03%20Filters.ipynb)

- Apply OOP to algorithms and visualization plots to gain reusability.


## Contributing

Fixes and improvements are more than welcome, so raise an issue or send a PR!
