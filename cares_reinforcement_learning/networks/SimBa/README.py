'''
Computer Vision and Natural Language Processing models
have had success by increasing the number of parameters
that they use. Despite fears of overfitting, including a
simplicity bias in these algorithms prevents that from
happening.

SimBaTD3 does the same thing (introducing a simplicity
bias). There are three ways it does that:
    1. Observed Normalization Layer
    Instead of imputing the parameters, this layer
    finds the difference from the average mean and variance
    and uses that difference as a parameter.
    2. Residual Feedforward block
    This adds a linear pathway from the input to the output,
    so instead of y=F(x), y=x+F(x) (research more on pathways)
    3. A layer normalization to control feature magnitude
    This prevents the values from getting too large (research!!)

I want to try implement SimBa with TD3 first.


Notes:
    1. Actor - critic algorithm
    2. Uses a deterministic policy gradient (SAC/TD3 is a good alg for SimBaTD3)

https://docs.agilerl.com/en/latest/tutorials/custom_networks/agilerl_simba_tutorial.html

PROGRESS:

I have used Sony's github to copy over the normalization work they had, and the agent/runningmean code
which supports it.

Notes on the things I've done so far:
1. I've coded a version of what I think SimBa's normalization function should do. I used the formula's
in Sony's paper to get what the running normalization should do.
2. I've copied across some normalization code from Sony, which I think happens before it is part of the
neural network, but I don't understand most of the code.

'''