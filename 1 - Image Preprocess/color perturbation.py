import tensorflow as tf
import matplotlib.pyplot as plt

def color_perturbation(image):
      
      image = tf.image.random_brightness(image, max_delta=64./ 255.)
      image = tf.image.random_saturation(image, lower=0, upper=0.25)
      image = tf.image.random_hue(image, max_delta=0.04)
      image = tf.image.random_contrast(image, lower=0, upper=0.75)
      
      
      
      
      return image


image_color_perturb = color_perturbation(image)

init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)
im = sess.run(image_color_perturb)
plt.imshow(npf)
