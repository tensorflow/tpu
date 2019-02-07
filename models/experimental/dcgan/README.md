## Overview

This example uses a DCGAN architecture to learn to produce MNIST digits and
CIFAR10 images. It trains on Google Cloud TPUs. It uses an open source library
called TF-GAN to abstract away many of the GAN and TPU infrastructure details.

To run this example, be sure to install TF-GAN with:

    pip install tensorflow-gan
