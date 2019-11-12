# Tools to manipulate datasets for Cloud TPU users.

licenses(["notice"])  # Apache 2.0

py_binary(
    name = "tfrecords_to_bigtable",
    srcs = ["tfrecords_to_bigtable.py"],
    python_version = "PY2",
    deps = [
        "//third_party/py/tensorflow",
        "//third_party/tensorflow/core/platform/cloud:gcs_file_system",
    ],
)
