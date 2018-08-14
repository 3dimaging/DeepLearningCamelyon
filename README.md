
1. Install ASAP (https://github.com/DIDSR/DeepLearningCamelyon/blob/master/ASAP%20installation%20(Ubuntu%2016.04))
2. Image Visulization (https://github.com/DIDSR/DeepLearningCamelyon/blob/master/Display%20annotation%20over%20image)
3. Mask Creation (DeepLearningCamelyon/mask creatation--updated)
4. Set up machine learning environment (DeepLearningCamelyon/Setup Machine Learning Environment)









# DeepLearningCamelyon

use Keras 2.0.0 for googlenet(very important---may not be necessary)

pip install keras==2.0.0.


If you need more information about how ElementTree package handle XML file, please follow the link: 
https://docs.python.org/3/library/xml.etree.elementtree.html

It took me long time to figure it out:
the model compiled under python 3.5 can not be loaded under python 3.6


export PATH="$PATH:/usr/local/cuda-8.0/bin"
export LD_LIBRARY_PATH="/usr/local/cuda-8.0/lib64"
export PYTHONPATH=”$PYTHONPATH:/opt/ASAP/bin”

# install h5py package for model saving and revi
sudo apt-get install libhdf5-serial-dev
pip3 install h5py
