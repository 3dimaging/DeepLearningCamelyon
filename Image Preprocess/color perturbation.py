import tensorflow as tf

def color_pertubation(image):
      
      image = tf.image.random_brightness(image, max_delta=64./ 255.)
      image = tf.image.random_saturation(image, max_delta=0.25)
      image = tf.image.random_hue(image, max_delta=0.04)
      image = tf.image.random_contrast(image, max_delta=0.75)
      
      
      
      return image
