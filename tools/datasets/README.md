# Tools for preparing datasets

## imagenet_to_gcs.py
Downloads [Image-Net](http://image-net.org/) dataset, transforms data into
`TFRecords`, and uploads to the specified GCS bucket. The script also has flags to
skip the GCS bucket upload and utilize an existing download of ImageNet.
Common to the various options are the following commands:

```bash
pip install gcloud google-cloud-storage
pip install tensorflow
```

**Image-Net to GCS**

Downloads the files from [Image-Net](http://image-net.org/), processes them into
`TFRecords` and uploads them to the specified GCS bucket.

```bash
python imagenet_to_gcs.py \
  --project="TEST_PROJECT" \
  --gcs_output_path="gs://TEST_BUCKET/IMAGENET_DIR" \
  --local_scratch_dir="./imagenet" \
  --imagenet_username=FILL_ME_IN \
  --imagenet_access_key=FILL_ME_IN \
```

**Image-Net to local only**

Downloads the files from [Image-Net](http://image-net.org/) and processes them
into `TFRecords` but does not upload them to GCS.

```bash
# `local_scratch_dir` will be where the TFRecords are stored.`
python imagenet_to_gcs.py \
  --local_scratch_dir=/data/imagenet \
  --nogcs_upload

```

**Image-Net with existing .tar files from Image-Net**

Utilizes already downloaded .tar files of the images


```bash
export IMAGENET_HOME=FILL_ME_IN
# Setup folders
mkdir -p $IMAGENET_HOME/validation
mkdir -p $IMAGENET_HOME/train

# Extract validation and training
tar xf ILSVRC2012_img_val.tar -C $IMAGENET_HOME/validation
tar xf ILSVRC2012_img_train.tar -C $IMAGENET_HOME/train

# Extract and then delete individual training tar files This can be pasted
# directly into a bash command-line or create a file and execute.
cd $IMAGENET_HOME/train

for f in *.tar; do
  d=`basename $f .tar`
  mkdir $d
  tar xf $f -C $d
done

cd $IMAGENET_HOME # Move back to the base folder

# [Optional] Delete tar files if desired as they are not needed
rm $IMAGENET_HOME/train/*.tar

# Download labels file.
wget -O $IMAGENET_HOME/synset_labels.txt \
https://raw.githubusercontent.com/tensorflow/models/master/research/slim/datasets/imagenet_2012_validation_synset_labels.txt

# Process the files. Remember to get the script from github first. The TFRecords
# will end up in the --local_scratch_dir. To upload to gcs with this method
# leave off `nogcs_upload` and provide gcs flags for project and output_path.
python imagenet_to_gcs.py \
  --raw_data_dir=$IMAGENET_HOME \
  --local_scratch_dir=$IMAGENET_HOME/tf_records \
  --nogcs_upload
```
