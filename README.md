# ECE69500ILGM_project_diffusionOOD

Based on [Denoising Diffusion Models for Out-of-Distribution Detection](https://openaccess.thecvf.com/content/CVPR2023W/VAND/papers/Graham_Denoising_Diffusion_Models_for_Out-of-Distribution_Detection_CVPRW_2023_paper.pdf) by M. S. Graham et al.

It is well known that reconstruction based methods for evaluating OOD samples on VAEs is ill-advised. VAEs can reconstruct similar distributions faithfully, e.g. FashionMNIST and MNIST. In contrast, with diffusion models we can gradually destroy the structure of the image using noise, and then attempt to reconstruct it. Clearly if the sample is all noise (all structure destroyed by noise) then we will only generate samples from our distribution. 
