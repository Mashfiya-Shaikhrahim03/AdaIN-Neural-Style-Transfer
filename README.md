# AdaIN Neural Style Transfer 🎨

## About the Project

This project is about transferring the artistic style of one image to another using deep learning.

The application takes two images: a content image and a style image. The content image provides the main structure of the output, while the style image provides its visual appearance, such as colors, textures, and patterns.

For this project, I am using Adaptive Instance Normalization (AdaIN) to perform neural style transfer. I have also built a Flask web application where users can upload images, see the generated result, and download it.

The main idea behind this project is to understand how content and style can be represented separately and then combined to generate a new image.

## What I Wanted to Explore

While working on this project, I wanted to understand a few things:

- How can a neural network capture the content of an image?
- How is artistic style represented in the network?
- What are the different ways to perform neural style transfer?
- Why is AdaIN useful when we want to use different style images?
- How do speed and output quality differ between NST methods?

## Technologies Used

- Python
- PyTorch
- Flask
- HTML, CSS and JS
- VGG-based feature extraction
- Adaptive Instance Normalization (AdaIN)

## How Neural Style Transfer Works

Neural Style Transfer, or NST, combines the content of one image with the artistic style of another.

For example, we can take a photograph as the content image and a painting as the style image. The output should retain the important objects and structure of the photograph while taking on some of the painting's colors and textures.

The two main parts are:

**Content:** The objects, shapes, and general structure we want to preserve.

**Style:** The colors, textures, patterns, and other visual characteristics we want to transfer.

Getting both parts to work well together is one of the main challenges of neural style transfer.

## Different Approaches to NST

Before choosing AdaIN, it is useful to understand the main approaches to neural style transfer and their differences.

### 1. Optimization-Based NST

This approach is associated with the work of Gatys and colleagues. It starts with an output image and repeatedly updates it to reduce a combination of content and style losses.

The method can produce good artistic results, but the repeated optimization can take a long time. A new content and style combination generally requires another optimization process.

### 2. Feed-Forward Style Transfer

Feed-forward methods use a trained neural network to generate the stylized image in a forward pass.

Once the model is trained, generating an image is usually much faster than optimizing each output from scratch. However, some feed-forward models are designed for a particular style, so supporting another style may require retraining or a different model.

### 3. Arbitrary-Style Transfer

Arbitrary-style transfer aims to use different style images without training a completely separate model for each one.

This is useful for an application where users can choose their own content and style images. AdaIN is one method that supports this kind of style transfer.

## Why I Chose AdaIN

I chose AdaIN because I wanted to work with different style images while keeping the image-generation process relatively efficient.

AdaIN stands for **Adaptive Instance Normalization**. It works with the feature representations extracted from the content and style images.

In simple terms, it adjusts the channel-wise mean and standard deviation of the content features to match those of the style features. These statistics help represent aspects of the style, although they do not capture every detail of an artistic image.

The adapted features are then passed through a decoder to generate the final image.

The important point is that the content image and style image have different roles, but their feature representations work together in the same pipeline.

## How AdaIN Works in This Project

The basic process is:

1. The user provides a content image and a style image.
2. Both images are processed by the encoder to extract their features.
3. AdaIN adapts the content features using the style features' statistics.
4. The decoder converts the adapted features back into an image.
5. The application displays the generated image and allows the user to download it.

The encoder, AdaIN operation, and decoder each have a different role in generating the final result.


## Running the Project

First, install the required dependencies:

```bash
python -m pip install -r requirements.txt
```

Make sure the required pretrained model weights are available at the paths expected by the application.

Then start Flask:

```bash
python app.py
```

Open the local address printed in the terminal.

