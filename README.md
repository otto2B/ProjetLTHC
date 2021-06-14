# ProjetLTHC
Semester project at EPFL by Théo &amp; Constantin

The project is an inference problem. Given a generated matrix we want to reconstruct its main components by minimizing a loss function by gradient descent. 
To avoid local minimum we add to the descent a perturbation in the form of a Brownian motion. In a first time we will implement a framework that will allow 
us in a second time to test different parameters’ combinations on large generated data set over infinite time. This will allow us to confront theory and 
understand better how Brownian motion and stochastic differential equations (SDE) can help us on this particular problem.

Our framework architecture is as follows:
Basic functions and definitions are implemented in the util.py file. You will find how we generate u∗ and v∗,
how we compute the overlaps, how we generate the matrix Y and compute the projector that helps us fix the
vectors on the hypershpere during the descent. Finally we defined two versions of the gradients for u(t) and
v(t).
Then, in the gradient descent.py file we defined three gradient descents for each type of gradient, one using
projections, one without projections and one without projections but with normalization at each step.
We ran the simulations on two different jupyter notebooks simulation2D.ipynb which displays MSEs or overlaps
in functions of βu and βv for a certain learning rate ratio (λu/λv). The second one, simulation3D.ipynb compares λv
two different simulations (different learning speed ratios) on the same graph.
