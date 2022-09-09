#!/usr/bin/python3

import os
from configs import MODEL_LINKS

tf_models = "data/models"
if os.path.isdir(tf_models) is False:
    os.mkdir(tf_models)
for m, link in MODEL_LINKS.items():
    final_path = f"{tf_models}/{m}"
    tar_file = final_path + ".tar.gz"
    if os.path.isfile(tar_file):
        os.remove(tar_file)
    os.system(f"wget {link} -O {tar_file}")
    if os.path.isdir(final_path) is False:
        os.mkdir(final_path)
    os.system(f"tar xzf {final_path}.tar.gz -C {final_path}/")
