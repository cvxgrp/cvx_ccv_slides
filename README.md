# Convex-Concave Lecture Slides and Code

This repository contains the [slides on the convex-concave procedure](https://github.com/cvxgrp/cvx_ccv_slides/blob/main/cvxccv.pdf) from [EE 364a](https://web.stanford.edu/class/ee364a/),
together with example code.
To see the animations in the PDF file, you need to select a viewer that supports animations, such as Adobe Acrobat Reader.

For more theoretical background, please refer to Thomas Lipp and Stephen Boyd's paper
on [Variations and Extensions of the Convex-Concave Procedure](https://web.stanford.edu/~boyd/papers/cvx_ccv.html).

Most of the examples use [DCCP](https://web.stanford.edu/~boyd/papers/dccp.html),
which is short for disciplined convex-concave programming. DCCP is based on the domain-specific language
for convex optimization [CVXPY](https://www.cvxpy.org).

Here is an example on collision avoidance, using the convex-concave procedure (the code is [here](code/collision_avoidance.py)). Each colored vehicle aims
to travel to the opposite side of the circle of initial positions.
We minimize the total distance travelled by all vehicles, subject to
no collisions.  The convex-concave procedure discovers the roundabout.

![Collision Avoidance](img/collision_avoidance.gif)
