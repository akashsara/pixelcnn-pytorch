# pixelcnn-pytorch
A PyTorch implementation of different PixelCNN models. 
Based on [Pixel Recurrent Neural Networks](https://arxiv.org/abs/1601.06759) by van den Oord et. al.
This by no means serve to reproduce the original results in the paper.

## PixelCNN
PixelCNNs are a type of autoregressive generative models which try to model the generation of images as a sequence of generation of pixels. More formally, PixelCNN model the joint distribution of pixels over an image x as the following product of conditional distributions, where x<sub>i</sub> is a single pixel:

<img src="https://i.imgur.com/pP3SLRU.png" width="250"/>

The ordering of the pixel dependencies is in raster scan order: row by row and pixel by pixel within every row. Every pixel therefore depends on all the pixels above and to the left of it, and not on any other pixels. We see this autoregressive property in other autoregressive models such as MADE. The difference lies in the way the conditional distributions are constructed. With PixelCNN every conditional distribution is modelled by a CNN with masked convolutions. 

<img src="https://i.imgur.com/qGTXtcl.png" width="300" hspace="60"/> <img src="https://i.imgur.com/Hrr2Ynq.png" width="200"/>         

The left figure visualizes how the PixelCNN maps a neighborhood of pixels to prediction for the next pixel. To generate pixel x<sub>i</sub> the model can only condition on the previously generated pixels x<sub>1</sub>, ..., x<sub>i-1</sub>. This conditioning is done by masking the convolutional filters, as shown in the right figure. This is a type A mask, in contrast to type B mask where the weight for the middle pixel also is set to 1. 

## Datasets
The four datasets used:

Binary Shapes | Binary MNIST | Colored Shapes | Colored MNIST
:--- | :--- | :--- | :--- 
![](https://i.imgur.com/4iU3eDY.png) | ![](https://i.imgur.com/mlO1TuB.png) | ![](https://i.imgur.com/F23XE4t.png) | ![](https://i.imgur.com/bvtHHQm.png)


## PixelCNN models
#### Regular PixelCNN
This model followes a simple PixelCNN architecture to model binary MNIST and shapes images. 
It has the following network design: 
- A  7×7  masked type A convolution
- 5  7×7  masked type B convolutions
- 2  1×1  masked type B convolutions
- Appropriate ReLU nonlinearities and Batch Normalization in-between
- 64 convolutional filters

#### PixelCNN with independent color channels (PixelRCNN)
This model supports RGB color channels, but models the color channels independently. More formally, we model the following parameterized distribution:

<img src="https://i.imgur.com/uzd19aT.png" width="300"/>

Trained on color Shapes and color MNIST
It uses the following architecture:
- A 7×7  masked type A convolution
- 8 residual blocks with masked type B convolutions
- Appropriate ReLU nonlinearities and Batch Normalization in-between
- 128 convolutional filters

#### PixelCNN with dependent color channels (Autoregressive PixelRCNN)
This PixelCNN models dependent color channels. This is done by changing the masking scheme for
the center pixel. The filters are split into 3 groups, only allowing each group to see the groups before (or including the current group, for type B masks) to maintain the autoregressive property. More formally, we model the parameterized distribution:

<img src="https://i.imgur.com/zD81GA7.png" width="300"/>

For computing a prediction for pixel x<sub>i</sub> in channel R we only use previous pixels x<sub><i</sub> in channel R (mask type A). Then, when predicting pixel x<sub>i</sub> in the G channel we use the previous pixels x<sub><i</sub> in both G and R, but since we at this time also have a prediction for x<sub>i</sub> in the R channel, we may use this as well (mask type B). Similarly, when predicting x<sub>i</sub> in channel B, we can use previous pixels for all channels, along with current pixel x<sub>i</sub> for channel R and G.
This way, the predictions are now dependent on colored channels. 

<img src="https://i.imgur.com/kCByD1A.png" width="300"/>

Figure above shows the difference between type A and type B mask. The 'context' refers to all the previous pixels (x<sub><i</sub>).

#### Conditional PixelCNNs
This PixelCNN is class-conditional on binary MNIST and binary Shapes.
Class labels are conditioned on by adding a conditional bias in each convolutional layer.
Similar architecture as normal PixelCNN. 

## Generated samples from the models
Below are samples generated by the different PixelCNN models after training.
#### PixelCNN

Binary Shapes | Binary MNIST 
:--- | :--- 
![](https://i.imgur.com/vV7OM3T.png) | ![](https://i.imgur.com/ZLmO1CK.png)
 
#### PixelRCNN
Colored Shapes | Colored MNIST
:--- | :--- 
![](https://i.imgur.com/FJxxt1l.png) |  ![](https://i.imgur.com/4tp9mF6.png)

#### Autoregressive PixelRCNN
Colored Shapes | Colored MNIST
:--- | :---
 ![](https://i.imgur.com/poxJoWA.png) |  ![](https://i.imgur.com/EB0b3wx.png) 

#### Conditional PixelCNN
Binary Shapes | Binary MNIST
:--- | :---
 ![](https://i.imgur.com/JcR1pVS.png) |  ![](https://i.imgur.com/qLcP3n6.png)
