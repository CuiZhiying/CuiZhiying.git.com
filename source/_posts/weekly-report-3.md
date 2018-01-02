---
title: weekly report 3
date: 2017-12-24 20:24:03
categories:
tags:
---

## Summary
This week, I have shared some tips with my friends on the usage of Python and tensorflow. A hour's share, hardly two days's preparation.

And I read some code which achieved FCNs in tensorflow and trid to run it. Although, it was still some problems that I couldn't sold, I understood FCN much better with the help of seniors. Base on it, I trid to read the paper _**Semantic Image Segmentation With Deep Convolutional Nets and Fully Connected CRFs**_, also known as "**Deeplab v.1**".

Addtional, I has learned how to make segmentation labels, because Doctor Chu told us they would give us some MRI data with new labels. I think it will be better if we also know how to create the labels. Doctor Chu invited us to make a deep talk with them next week.

Saturday, Jiaxin and I attended am AI report on which I learn some autopolit technology and some usage of AR. Both of them make me exicted.

## Paper Reading
This week, what I learn is _**Semantic Image Segmentation With Deep Convolutional Nets and Fully Connected CRFs**_, also known as "**Deeplab v.1**".   
This paper realizes that there are two technical hurdles in the application of DCNNs to image labeling task: signal downsampling, and spatial invariance. And they trid to sold the first problem with "atrous"(with holes) algorithm, which allows effcient dens computation of DCNN responses in a scheme substantially simpler than earlier solution. Then, they boost their model's ability to capture fine details by employing a fully-connected Conditional Random Field (CRF) to solve the socond problem.

### Advantages
- This paper reduce the stride;
- Try to use hole algorithm to replace upsampling;
- The usage of multi-scale;
- Use CRFs for every pixel after traning.

### Disadvantages
- Multi-scale perform excellent, and they could make a better use of it.
- In fact, I don't think I understand how to do with CRF.

## Next Week's Plan
Prepare some matrial with details to talk with Doctor Chu and make some better rules to label MRI data files.

Maybe I should continue to learn **DeepLab v.2** and **DeepLab v.3**

I think it's time for me to go back Changsha and prepare for my final exams.
