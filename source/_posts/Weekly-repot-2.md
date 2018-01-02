---
title: Weekly repot 2
date: 2017-12-17 19:30:41
categories:
tags: [report, FCN]
---

## Summary
This week, I tried to sort out my code in CNN, but this was a little boring and time-consuming, which nearly took me a day to push a blog. So I had to slow down and continued to learn new thing, such as tensorboard( visualization tools for Tensoflow graph) and new papers.

One day, am undergraduate in our laboratory told me that I should take more time to read the latest papers and learn from them rather than just coding. He said he saw me just coding most of time and noticed me that it's important to improve the coding ability, but it's more important to learn how to think. It made me think a lot, and I thought he was right. Maybe I should trid to read more papers, codes just help me learn and understand better.

Friday, I took part in the group meeting for Weishi's laboratories, which is a paper sharing meeting.But I just could not get what they were talking about.

I just copyed the MRI data from Doctor Chu.
## Paper reading
This week, what I read was **_Fully Convolutional Networks for Semantic Segmentation_**, and the code achieved FCN in Tensorflow is [here](https://github.com/shekkizh/FCN.tensorflow/blob/master/FCN.py)
### Features
1. Change the fully connected layers in classical CNN networks into equivalent convolutional layers
2. Deconv(Upsampling) the final feature maps in the above result, sothat get the heatmap of object.
3. Use skip architecture to improve the prediction resultes.
### advantages
- It can accept any sizes of images as its input.
- It is more effective, as it avoids the repeate calculation from pixels blocks
### disadvantages
- The result is not enough detail. Although 8 times the upsampling is much better than 32 times, but the upsampled results are still relatively vague and smooth, insensitive to the details of the image.
- The classification of each pixel, did not give full consideration to the relationship between pixels and pixels. It lacks spatial consistency 
## Next week's plan
- Prepare to share what I learn in tensorflow.
- Continue to learn the FCN series paper.
- Test it in some MRI data.
