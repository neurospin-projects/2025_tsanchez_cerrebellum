## BasicNet 

Model used by Louise in the initial implementation of the BetaVAE : 
- Block arcitechture : 
Conv3d(kernel = 3, stride=1, padding=1) -> BatchNorm3d -> LeakyReLU -> Conv3d(kernel_size = 4, stride = 2, padding = 1)
- Each block divide each dimension by 2

Need to have an adaptation of the size of the input (padding done at the beginning of the computation)

**Default settings** : 
- `depth` : 3

## ConvNet : 

Encoder used for Champollion : 
**Block**
- 1st convolution : Conv3d(kernel_size = 7, stride=1, padding = 1) -> BatchNorm -> LeakyReLU -> Dropout3D 
- 2nd convolution : Conv3d with the same padding as the previous one

