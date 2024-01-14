# pytorch_forward_forward
Implementation of forward-forward (FF) training algorithm - an alternative to back-propagation
---

The base of the supervised version is taken from https://github.com/mohammadpz/pytorch_forward_forward.
This is an extension with the unsupervised data, with the idea of also implementing the top-down approach described by Hinton.

Below is my understanding of the FF algorithm presented at [Geoffrey Hinton's talk at NeurIPS 2022](https://www.cs.toronto.edu/~hinton/FFA13.pdf).\
The conventional backprop computes the gradients by successive applications of the chain rule, from the objective function to the parameters. FF, however, computes the gradients locally with a local objective function for each layer, so there is no need to backpropagate the errors.


The supervised approach imposes the label on top of the image by using the top row of pixels. This way, the networks learns connections between certain selected pixels, and the corresponding number features. The pictures are fed through the network in a way where positive samples pushes the layer weights upwards and negative examples pushes the corresponding weights downwards. In order to classify a new image, we have to impose all of the possible labels on top of the image and run the image through the network. We then check which label creates the largest "goodness", or the largest L2-norm of the latter feature space layers. Weighing all layers decisions together will provide the classification.

Positive sample: A number with the correct corresponding selected pixel.<br />
Negative sample: A number with the incorrect corresponding selected pixel.

---

The unsupervised approach creates a random bitmask which is then used to "fuse" two correct examples into an incorrect example as described by Hinton. The network is trained without any labels on good and bad samples. However, in order to get the classification, we need to teach the network which label belongs to what class. This is done by creating a linear classifier, which takes all of the the intermediate, normalized layers of the network as input, and gets trained on the target label. This might seem like a supervised approach with a label, but we are not affecting the layers of the network itself in any way regardless of the classification, only the linear classifier itself. In other words, we are combining the trained states of the intermediate layers to run inference on. Thanks to https://github.com/rmwkwok for helping me get my head around this.

Positive sample: Any good image in the dataset<br />
Negatiive sample: Two images of the dataset combined together with a randomly generated bitmask

---

#Smaller nets result:

Net([784, 64, 32])
Supervised test error: 0.09550005197525024

Net([784, 64, 64, 64, 64])
Unsupervised test error: 0.6461000144481659

-----------------------------------------------------

#Larger nets result:

Net([784, 2000, 2000])
Supervised test error: 0.06480002403259277

Net([784, 1000, 1000, 1000, 1000])
Unsupervised test error: 0.6809000074863434

---
