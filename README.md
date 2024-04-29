# ECE69500ILGM_project_diffusionOOD

Based on [Denoising Diffusion Models for Out-of-Distribution Detection](https://openaccess.thecvf.com/content/CVPR2023W/VAND/papers/Graham_Denoising_Diffusion_Models_for_Out-of-Distribution_Detection_CVPRW_2023_paper.pdf) by M. S. Graham et al.

It is well known that reconstruction-based methods for evaluating OOD samples on VAEs is ill-advised. VAEs can reconstruct similar distributions faithfully, e.g. FashionMNIST and MNIST. In contrast, with diffusion models we can gradually destroy the structure of the image using noise, and then attempt to reconstruct it. Clearly if the sample is all noise (all structure destroyed by noise) then we will only generate samples from our distribution, but it is not guaranteed that the generated sample is going to be similar to our original class, e.g. original class is a sneaker and we generated a sweater. Another thing to note is that if our model is good, it can denoise images that contain low amounts of noise. We conduct experiments to test if we can correctly identify OOD samples using reconstruction-based methods at different timesteps *t*.

## Experiment setup

We use a cosine noise scheduler with a time-aware Attention UNet taken from [here](https://github.com/labmlai/annotated_deep_learning_paper_implementations/blob/master/labml_nn/diffusion/ddpm/unet.py). We made some changes such as changing out the Swish function by the PyTorch one, nn.SiLU, and reducing the number of groups in the resnetblock to 2. We also changed the channels of the initial feature map to be 16. This is to reduce the complexity of the model, with a result of a 10M parameter model compared to 41M without them. 

We mainly evaluate two metrics, the MSE between the reconstructed image and the original image. The other metric is a perception score, which we get from using an ImageNet-pretrained AlexNet model on both the reconstructed image and the original image. We then add both scores into the resulting reconstruction error. We do this for 1000 samples at different timesteps.

## Results

We found that at low timesteps the OOD samples were reconstructed better due to some detail being lost on the ID FashionMNIST samples and it was impossible to tell the difference between them. As we gradually increase t, the distributions start to shift, with OOD MNIST having higher reconstruction error, and a binary classifier gradually improves, as shown in the ROC curves. However, after certain time t, all the structure is destroyed thus we are not able to get meaninful reconstructions, so the performance of the binary classifier drops. This indicates that there is a certain ideal time t that can maximize the performance of the binary classifier, however finding this seems difficult without brute force search.

## Code

I have included the 2 notebooks used for this project. [This](asd) is for training the model, and [the other](asd) is to conduct the experiments. To reproduce the experiments, download the [weights](https://drive.google.com/file/d/1aSl74xNuBoa8KkGUeVfUC_q0PPXf2PfX/view?usp=sharing) for the model and change the path to the downloaded weights in [experiments.ipynb](asd).
