# DeepLearningCamelyon

use Keras 2.0.0 for googlenet(very important)

pip install keras==2.0.0.


If you need more information about how ElementTree package handle XML file, please follow the link: 
https://docs.python.org/3/library/xml.etree.elementtree.html

It took me long time to figure it out:
the model compiled under python 3.5 can not be loaded under python 3.6


export PATH="$PATH:/usr/local/cuda-8.0/bin"
export LD_LIBRARY_PATH="/usr/local/cuda-8.0/lib64"
export PYTHONPATH=”$PYTHONPATH:/opt/ASAP/bin”
