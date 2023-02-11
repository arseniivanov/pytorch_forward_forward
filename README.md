# pytorch_forward_forward
Implementation of forward-forward (FF) training algorithm - an alternative to back-propagation
---

The base of the supervised version is taken from https://github.com/mohammadpz/pytorch_forward_forward.
This is an extension with the unsupervised data, with the idea of also implementing the top-down approach.

Below is my understanding of the FF algorithm presented at [Geoffrey Hinton's talk at NeurIPS 2022](https://www.cs.toronto.edu/~hinton/FFA13.pdf).\
The conventional backprop computes the gradients by successive applications of the chain rule, from the objective function to the parameters. FF, however, computes the gradients locally with a local objective function, so there is no need to backpropagate the errors.



Smaller nets result:

Net([784, 64, 32])
Supervised test error: 0.09550005197525024

Net([784, 64, 64, 64, 64])
Unsupervised test error: 0.6461000144481659

-----------------------------------------------------

Larger nets result:

Net([784, 2000, 2000])
Supervised test error: 0.06480002403259277

Net([784, 1000, 1000, 1000, 1000])
Unsupervised test error: 0.6809000074863434