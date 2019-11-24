# How many samples to estimate a convolution filter.
Here, I'll run some quick experiments to test the claims made by [How many samples regarding the representation ability of a convolution filter](https://papers.nips.cc/paper/7320-how-many-samples-are-needed-to-estimate-a-convolutional-neural-network). This will help me better understand how CNNs are superior to their fully connected counterparts.

Result: Failed to replicate exact results.
Observation: I obtained far better error rates. I suspect I am optimizing the wrong target function. Perhaps if the details of the experiments were made clearer I can get better results.
Consolation: I kind of got the asymptotic error bounds for the CNN and I did find the CNN to perform better than the FNN in general.