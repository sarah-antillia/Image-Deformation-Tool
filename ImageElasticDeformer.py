# Copyright 2024 antillia.com Toshiyuki Arai
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# ImageElasticTransformer.py

# This method has been taken from the following code.
# https://github.com/MareArts/Elastic_Effect/blob/master/Elastic.py
#
# https://cognitivemedium.com/assets/rmnist/Simard.pdf
#
# See also
# https://www.kaggle.com/code/jiqiujia/elastic-transform-for-data-augmentation/notebook


import os
import sys
import cv2
import glob 
#from scipy import ndimage as ndi
from scipy.ndimage.interpolation import map_coordinates

from scipy.ndimage.filters import gaussian_filter
import numpy as np
import shutil
from ConfigParser import ConfigParser

import traceback


DEFORMATION    = "deformation"

"""
;deformation3.config
[deformation]
images_dir  = "./images"
output_dir  = "./deformed"
alpha       = 1300
sigmoids    = [4,8,12]
"""

class ImageElasticDeformer:

  def __init__(self, config_file):
    print("=== ImageElasticDeformer ===")
    self.seed = 137
    np.random.seed(self.seed)
    
    self.config = ConfigParser(config_file)

    self.images_dir = self.config.get(DEFORMATION, "images_dir", dvalue="./images")
    if not os.path.exists(self.images_dir):
      error = "Not found " + self.images_dir
      raise Exception(error)

    self.output_dir = self.config.get(DEFORMATION, "output_dir", dvalue="./distorted") 

    if os.path.exists(self.output_dir):
      shutil.rmtree(self.output_dir)
    if not os.path.exists(self.output_dir):
      os.makedirs(self.output_dir)
    self.config.dump_all()

    self.alpha    = self.config.get(DEFORMATION,  "alpha",   dvalue=1300)
    self.sigmoids  = self.config.get(DEFORMATION, "sigmoids", dvalue=[10, 20])


  def deform(self):
    print("=== deform ===")
    with os.scandir(self.images_dir) as it:
      for entry in it:
        if entry.is_dir():
          print("---dir path {} name {}".format(entry.path, entry.name))
          self.deform_dir(entry.path, subdir=entry.name)
          pass
        elif entry.is_file():
          print("---file path {} name {}".format(entry.path, entry.name))
          self.deform_onefile(entry.path, self.output_dir)          

  def deform_dir(self, images_dir, subdir=""):
    print("=== deform_dir imagesdir:{} subdir:{}".format(images_dir, subdir))
    output_subdir = os.path.join(self.output_dir, subdir)
    if os.path.exists(output_subdir):
      shutil.rmtree(output_subdir)
    if not os.path.exists(output_subdir):
      os.makedirs(output_subdir)

    image_files  = glob.glob(images_dir + "/*.jpg")
    image_files += glob.glob(images_dir + "/*.png")
    image_files += glob.glob(images_dir + "/*.bmp")
    image_files += glob.glob(images_dir + "/*.tif")
    if len(image_files) == 0:
      print("Not found image_files "+ self.images_dir)
      return
    for image_file in image_files:
      self.deform_onefile(image_file, output_subdir)

  def deform_onefile(self, image_file, output_dir):
    print("--- deform_onefile {}".format(image_file))
    image    = cv2.imread(image_file)
    basename = os.path.basename(image_file)

    """Elastic deformation of images as described in [Simard2003]_.
    .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
       Convolutional Neural Networks applied to Visual Document Analysis", in
       Proc. of the International Conference on Document Analysis and
       Recognition, 2003.
    """
    random_state = np.random.RandomState(self.seed)
    
    shape = image.shape
   
    for sigmoid in self.sigmoids:

      dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigmoid, mode="constant", cval=0) * self.alpha
      dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigmoid, mode="constant", cval=0) * self.alpha
      #dz = np.zeros_like(dx)

      x, y, z = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]), np.arange(shape[2]))
      indices = np.reshape(y+dy, (-1, 1)), np.reshape(x+dx, (-1, 1)), np.reshape(z, (-1, 1))
      deformed = map_coordinates(image, indices, order=1, mode='nearest')  
      deformed = deformed.reshape(image.shape)

      filename = "deformed" + "_alpha_" + str(self.alpha) + "_sigmoid_" +str(sigmoid) + "_" + basename
      output_filepath  = os.path.join(output_dir, filename)
      cv2.imwrite(output_filepath, deformed)
      print("=== Saved {}".format(output_filepath))

  
if __name__ == "__main__":
  try:
    config_file = "./deformation.config"
    if len(sys.argv) == 2:
      config_file = sys.argv[1]
    
    if not os.path.exists(config_file):
       error = "Not found " + config_file
       raise Exception(error)

    deformer = ImageElasticDeformer(config_file)
    deformer.deform()
  except:
    traceback.print_exc()


