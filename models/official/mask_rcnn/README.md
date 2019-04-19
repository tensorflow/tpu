# Installing extra packages

Mask R-CNN requires a few extra packages.  We can install them now:

```
sudo apt-get install -y python-tk && \
pip install Cython matplotlib opencv-python-headless pyyaml Pillow && \
pip install 'git+https://github.com/cocodataset/cocoapi#egg=pycocotools&subdirectory=PythonAPI'
```
