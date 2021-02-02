# Try it

Try to run our pre-trained COCO Mask R-CNN using [Colab](https://colab.sandbox.google.com/github/tensorflow/tpu/blob/master/models/official/mask_rcnn/mask_rcnn_demo.ipynb).

# Installing extra packages

Mask R-CNN requires a few extra packages.  We can install them now:

```
sudo apt-get install -y python-tk && \
pip3 install --user Cython matplotlib opencv-python-headless pyyaml Pillow && \
pip3 install --user 'git+https://github.com/cocodataset/cocoapi#egg=pycocotools&subdirectory=PythonAPI'
```
