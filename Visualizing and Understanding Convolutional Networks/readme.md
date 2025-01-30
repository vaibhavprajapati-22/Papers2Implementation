# Visualizing and Understanding Convolutional Networks

## Introduction :
Convolutional Networks have been used for image classification, segmentation, generation, and many other tasks, performing exceptionally well. 
This repository contains code for visualizing what each layer in a CNN actually captures for a given input image. The model used is AlexNet, trained on the 
ImageNet dataset. The model is modified according to the paper: "Visualizing and Understanding Convolutional Networks".

[Paper Link](https://arxiv.org/abs/1311.2901)

## Approach :
To examine the convnet, a deconvnet is attached to it each layer. It includes three function :
1. Unpooling : Since maxpooling is invertible operation, we can't get the exact original information before the maxpooling was applied but we can get the approximate
   content of the image by storing the location where the operation was applied in form of switch variable, so that during reversing the operation we set values in according
   to indices stored in switch.
2. Rectification : We apply relu function to remove the negative pixel values present.
3. Filtering : To invert the effect of the convent filter, we use transpose version of the same filters to the rectified maps.

## Results :
 ![image](https://github.com/user-attachments/assets/a72806a9-a347-45e8-9075-a0a253c7daa6)
 ![image](https://github.com/user-attachments/assets/bf6a79f9-c890-4546-981b-ebdbd8db22b8)
 ![image](https://github.com/user-attachments/assets/219f210b-2d9d-4583-9c4c-a698f2e701d4)
 ![image](https://github.com/user-attachments/assets/f20a5832-fa09-4df2-b36f-e4b68cbec05f)
 ![image](https://github.com/user-attachments/assets/3067d268-c255-45be-8d9e-8faeff330978)
 ![image](https://github.com/user-attachments/assets/0aaee449-f154-4d2e-86d9-762692f7ec64)

## How To Run :gun:
  1. Clone the repository :
    <pre>
    <code class="python">
    git clone [https://github.com/vaibhavprajapati-22/Fast-Neural-Style-Transfer](https://github.com/vaibhavprajapati-22/Papers2Implementation.git)
    </code>
    </pre>
  2. Get in correct directory :
     <pre>
      <code class="python">
        cd Fast-Neural-Style-Transfer
        cd Visualizing and Understanding Convolutional Networks
      </code>
     </pre>
  3. Run main script
     <pre>
      <code class="python">
        python main.py
      </code>
     </pre>

## References :
* [Visualizing and Understanding Convolutional Networks](https://arxiv.org/abs/1311.2901).



