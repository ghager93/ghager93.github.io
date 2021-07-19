---
title: "Enclosed Curve Shortening Flow"
permalink: /enclosed-curve-shortening-flow/
mathjax: True
---

# Enclosed Curve Shortening Flow

## Motivation

Testing latex $\sqrt 2$

I love beautiful data.  Clean, succinct and informative data.  Goldilocks data: Not too much, not too little.  The kind of data that deserves to be hung in a gallery.  Specifically, I am talking about data visualisation.

This project was first inspired by this picture on Reddit;

<img src="C:\Users\ghage\Pictures\typora\curve_shortening.jpg" style="zoom: 33%;" />

It shows a crude outline of Ireland (the island).  I can't remember what it was representing, I just remember thinking it didn't look very good.  However, it could be an interesting way to visualise data relating to proportions of a country.  For instance, if you wanted to display the demographic breakdown of a country, you could use cumulative polygonal areas as a kind of stack chart.  The following diagram gives a rough example of this; 

<img src="C:\Users\ghage\Pictures\typora\brazil-silhouette_percent.bmp" style="zoom:15%;" />

This is a silhouette of Brazil, the different colours represent cumulative proportions of the silhouette.  Importantly, the different shapes maintain some resemblance to the original shape, i.e. rather than just becoming circles.  This topological relation gives a much nicer appearance.

There are several rules that I would like this algorithm to follow:

1. All smaller proportions are a subset of any larger proportion, i.e. the shape that corresponds to $x\%$ of the full-size shape is completely contained with the shape that corresponds to $(x+\epsilon)\%$ of the full-size shape for an arbitrarily small $\epsilon$;
2. The shapes retain some level of topological relation to the original shape; and 
3. The proportions reduce to a single point corresponding to $\epsilon\%$ of the original shape for an arbitrarily small $\epsilon$.



## Related Work

### Curve Shortening Flow

Curve shortening flow moves the points of a smooth curve perpendicular to the curve at a speed proportional to the curvature.  It is also known as Euclidean shortening flow, geometric heat flow, or arc length evolution.

<img src="C:\Users\ghage\Pictures\typora\Convex_curve_shortening.png" style="zoom:67%;" />



As the curve evolves it remains simple and smooth.  It loses area at a constant rate.  If the curve is non-convex it will evolve to a convex curve.  Once convex, the curve will reduce to a circle.

###### Gage-Hamilton-Grayson Result

*If a smooth simple closed curve undergoes the curve-shortening flow, it  remains smoothly embedded without self-intersections. It will eventually become convex, and once it does so it will remain convex. After this time, all points  of the curve will move inwards, and the shape of the curve will converge to a circle as the whole curve shrinks to a single point. This behaviour is  sometimes summarized by saying that every simple closed curve shrinks to a "round point".*



##### Curvature

Let $C(s) = (x(s), y(s))$ be a **smooth curve**, where $s$ is a parameter describing the length along the curve, and $(x(s), y(s))$ is the Cartesian point of the curve at arc length $s$;

<img src="C:\Users\ghage\Pictures\typora\parametric curve.png" style="zoom:50%;" />

The **unit tangential vector** $\vec t$ for each point is given by

$$
\vec {t(s)} = \frac{1}{\sqrt{x'^2(s) + y'^2(s)}}\left<x'(s), y'(s)\right>
$$

The **unit normal vector** $\vec n$ for each point is given by

$$
\vec {n(s)} = \frac {\vec {t'(s)}} {\lvert \vec {t'(s)} \rvert}
$$

<img src="C:\Users\ghage\Documents\Mathematics & Statistics\Curve Shortening\Convex Curve Shortening.assets\tangent_normal.png" style="zoom:50%;" />

The **curvature** $k(s)$ is a scalar that describes how much curve is occurring at the point $\left<x(s), y(s)\right>$;

$$
k(s) = \frac {\vec {t'(s)}} {\vec {n(s)}} = \lvert \vec {t'(s)} \rvert
$$

 It is related to the radius of the osculating circle at $\left<x(s), y(s)\right>$.  The osculating circle is the circle mapped out by extending the curvature at the point:

<img src="C:\Users\ghage\Documents\Mathematics & Statistics\Curve Shortening\Convex Curve Shortening.assets\osculating_circle.png" style="zoom:60%;" />

The **curve shortening flow** is defined such that each point $\bf x = \left<x(s), y(s)\right>$ on the curve moves according to the following differential equation:

$$
\frac {d \bf x} {dt} = -k(\bf x)\vec n(\bf x) = -\vec t(\bfsymbol x)
$$

<img src="C:\Users\ghage\Documents\Mathematics & Statistics\Curve Shortening\Convex Curve Shortening.assets\csf_vectors-1622729749489.png" style="zoom:60%;" />



##### Discrete Curve Shortening Flow

For a polygon made up of discrete vertices, the curvature $k(\bf v)$ at vertex $\bf v$ is 

$$
k(\bf v) = \pi - \alpha
$$

where $\alpha$ is the interior angle at $\bf v$.

The normal vector $\vec {n(\bf v)}$ points in the direction of the angle bisector at $\bf v$;

<img src="C:\Users\ghage\Documents\Mathematics & Statistics\Curve Shortening\Convex Curve Shortening.assets\discrete_csf.png" style="zoom:60%;" />



#### Finite Difference Method

Finite difference methods involve using discrete samples of a function to approximate the function's derivatives.

There are three basic types of finite difference: *forward*, *backward* and *central*.

![](C:\Users\ghage\Documents\Mathematics & Statistics\Curve Shortening\Convex Curve Shortening.assets\Finite_difference_method.svg)

###### Forward Difference

Forward difference calculates the gradient between the current value of the curve and a value to the right;

$$
\Delta_h f (x) = f(x+h) - f(x)
$$


###### Backward Difference

Backward difference uses a value to the left of the current value to calculate the difference;

$$
\nabla_h f(x) = f(x) - f(x-h)
$$


###### Central Difference

Central difference uses values both to the left and to the right. Thus, central difference can be thought of as an average of the forward and backward differences.  Unlike forward and backward, the central difference has no directional bias.

$$
\delta_h f(x) = f(x+\frac h 2) - f(x-\frac h 2)
$$


##### First-Order Derivatives

The gradient of the curve can be approximated by dividing the finite difference by the interval $h$, i.e.

$$
\begin{align}
	f'(x) &= \Delta_h f(x) + O(h) \\
	&= \nabla_h f(x) + O(h) \\
	&= \delta_h f(x) + O(h^2)
\end{align}
$$

Hence, the central difference results in a better approximation of the derivative.



##### Second-Order Derivatives

The second-order central difference derivative approximation is given by

$$
f''(x) \approx \frac {f(x+h) - 2f(x) + f(x-h)} {h^2}
$$


##### Non-Uniform Grid

If the finite difference grid is not equidistant, then the interval $h$ is instead a function of $x$: $h_i = h(x_i)$. In this case the second-order central derivative is calculated differently:

$$
f''(x_i) \approx \frac {f(x_i + h_i) - 2f(x) + f(x_i - h_{i-1})} {\frac 1 2 (h_i + h_{i-1})}
$$

where $h_i = x_{i+1} - x_i$

This is the case when using arc length as the interval for a curve defined on a pixelated grid, as shown below.

<img src="C:\Users\ghage\Documents\Mathematics & Statistics\Curve Shortening\Convex Curve Shortening.assets\pixel_distant.jpg" style="zoom:67%;" />

The orange squares represent pixels in a curve, with the blue line connecting their centres.  For any horizontally or vertical adjacent pixels, the change in arc length will be a  unit. For a diagonal movement, however, the change is equal to $\sqrt 2$.



##### Finite Difference Curve Shortening

Let `points` be a list of the locations of pixels representing the curve;

```python
points = [[0, 0], [1, 1], [2, 1], [2, 2], [2, 3], [1, 4], [2, 5], [3, 5], [4, 5]]
```

The arc length $\bf s$ is calculated as the cumulative Euclidean distance between the points;

```python
def arc_length(points):
    return np.cumsum(np.concatenate(([0], np.linalg.norm(points[1:] - points[:-1], axis=1))))

s = [0.   , 1.414, 2.414, 3.414, 4.414, 5.828, 7.243, 8.243, 9.243]
```

 $\bf s$ can be normalised to lie in the region $\bf s_n \in [0, 1]$.

```python
sn = [0.   , 0.153, 0.261, 0.369, 0.478, 0.631, 0.784, 0.892, 1.   ]
```

The backward finite difference is used to find the first-order differential of the curve, 

$$
\bf x'(s_n) = \left< x'(s_n), y'(s_n) \right>
$$

```python
def delta_array(points, arc_length):
    delta = (points[1:] - points[:-1]) / (arc_length[1:] - arc_length[:-1])[:, None]
    return np.vstack((delta[0, :], delta))

dpoints =  [[ 6.536,  6.536],
            [ 8.121,  2.707],
            [ 4.621,  4.621],
            [ 0.   ,  9.243],
            [-2.707,  8.121],
            [ 0.   ,  6.536],
            [ 8.121,  2.707],
            [ 9.243,  0.   ],
            [ 9.243,  0.   ]]
```

The unit tangent vectors of the curve are found by normalising each point of the derivative,

$$
\bf t = \frac 1 {\sqrt{x'^2(s) + y'^2(s)}}\bf x'(s_n)
$$


```python
t =    [[ 0.707,  0.707],
        [ 0.949,  0.316],
        [ 0.707,  0.707],
        [ 0.   ,  1.   ],
        [-0.316,  0.949],
        [ 0.   ,  1.   ],
        [ 0.949,  0.316],
        [ 1.   ,  0.   ],
        [ 1.   ,  0.   ]]
```

The second derivative is found using the central finite difference,

$$
\bf x''(s_n) = \left< x''(s_n), y''(s_n) \right>
$$

```python
def delta_delta_array(dpoints, arc_length):
    delta_delta = 2 * (dpoints[2:] - dpoints[1:-1]) / (arc_length[2:] - arc_length[:-2])[:, None]
    first = (dpoints[1] - dpoints[0]) / (arc_length[1] - arc_length[0])
    last = (dpoints[-1] - dpoints[-2]) / (arc_length[-1] - arc_length[-2])
    
    return np.vstack((first, delta_delta, last))
    
ddpoints = [[ 10.364, -25.021],
            [-14.657,   0.   ],
            [-37.531,  30.203],
            [-33.867,  16.175],
            [ -7.328, -10.364],
            [ 35.385, -17.692],
            [ 28.056, -25.021],
            [  5.182, -12.51 ],
            [  0.   ,   0.   ]]   
```

The unit normal vectors of the curve are the normalised vectors of the second derivative;

$$
\bf n = \frac {\bf t'} {\left| \bf t' \right|}
$$

```python
n =    [[ 0.383, -0.924],
        [-1.   ,  0.   ],
        [-0.779,  0.627],
        [-0.902,  0.431],
        [-0.577, -0.816],
        [ 0.894, -0.447],
        [ 0.746, -0.666],
        [ 0.383, -0.924],
        [ 0.   ,  0.   ]]
```

<img src="C:\Users\ghage\Documents\Mathematics & Statistics\Curve Shortening\Convex Curve Shortening.assets\tangents_normals.png" style="zoom:67%;" />

##### Problems with Discretisation

Calculating derivatives on a discrete grid will lead to inaccurate vector measurements, as shown below;

<img src="C:\Users\ghage\Documents\Mathematics & Statistics\Curve Shortening\Convex Curve Shortening.assets\discretisation.png" style="zoom:60%;" />

Since the only movements possible on the curve are horizontal, vertical and diagonal, the tangent and normal vector cannot accurately describe the overall curvature.

The effect can be mitigated by using a low-pass filter on both the tangent and normal arrays;

<img src="C:\Users\ghage\Documents\Mathematics & Statistics\Curve Shortening\Convex Curve Shortening.assets\filtered_heart.png" style="zoom:60%;" />

In the left image the tangent vector has been filtered with a square kernel 10 samples long. The normal vector has been filtered with a square kernel 20 samples long;

```python
moving_avg_tangent = ndimage.convolve1d(tangent, np.ones(10)/10, mode='wrap', axis=0)
normal = delta_delta_array(moving_avg_tangent, arc_length)
moving_avg_normal = ndimage.convolve1d(normal, np.ones(20)/20, mode='wrap', axis=0)
```

The normal window is twice as long as the tangent window since the normal calculation uses values two samples apart.

The right image shows the tangent and normal vectors convolved with Gaussian kernels with standard deviations $\sigma = 5$ and $\sigma = 10$, respectively.  For the normal vector, the Gaussian filter gives a noticeably smoother result.

```python
gauss_tangent = ndimage.gaussian_filter1d(tangent, 5, mode='wrap', axis=0)
normal = delta_delta_array(gauss_tangent, arc_length)
gauss_normal = ndimage.gaussian_filter1d(normal, 10, mode='wrap', axis=0)
```

###### Curvature

The curvature can be calculated as the cross-product of the normal and the tangent;

$$
\begin{align}
	\kappa(s) &= \bf t(s) \times \bf n(s) \\
	\\
	&= x'(s)y''(s) - x''(s)y'(s)
\end{align}
$$

 Below is shown the curvature for each of the filter methods;

![](C:\Users\ghage\Documents\Mathematics & Statistics\Curve Shortening\Convex Curve Shortening.assets\curvature.png)

A positive (red) curvature indicates an inward curve, while negative (blue) indicates an outward curve.  As can be seen, the Gaussian filter leads to smaller extremes than the moving average. 



##### Shortening

By moving the each point of the curve in the direction of its normal, the curvature will move toward a constant.  That is, outward curves will move outward and inward curves inward, until the curvature at each point is equal.  At this point, the curve will be a circle, and subsequent moves along the normal will shrink the circle to a singularity.  This can be simulated via iterative update;

$$
\bf x^{(i+1)}(s) \leftarrow \bf x^{(i)}(s) + \beta\bf n^{(i)}(s)
$$

where  $\beta$ is some step size constant.  However, this process can be (and often is) problematic, as shown below;

<img src="C:\Users\ghage\Documents\Mathematics & Statistics\Curve Shortening\Convex Curve Shortening.assets\mov_avg_and_gauss_iterations.png" style="zoom:60%;" />

The picture shows ten iterations of additive updates to curve, both with moving average and Gaussian filters.  Even with a small step ($\beta = 0.001$) the curve quickly becomes unstable.



######  Morphological Smoothing

One reason for this is the sharp curves at the top and bottom.  Being the most extreme points of curvature, these points should have the largest movements and thus round-off quickly.  Instead, they are hidden by the filtering and propagate through iterations.  As the difference in curvature between these points and their neighbours increases, instability is formed and cascades around the curve.

<img src="C:\Users\ghage\Documents\Mathematics & Statistics\Curve Shortening\Convex Curve Shortening.assets\gauss_it_zoom.png" style="zoom:60%;" />

These edges can be removed by applying a binary opening and closing to the image before calculating its edge;

<img src="C:\Users\ghage\Documents\Mathematics & Statistics\Curve Shortening\Convex Curve Shortening.assets\open_and_close.png" style="zoom:50%;" /><img src="C:\Users\ghage\Documents\Mathematics & Statistics\Curve Shortening\Convex Curve Shortening.assets\open_and_close_vectors.png" style="zoom:60%;" />

This stops the sharp edge propagating through the iterations, but does not fix the instability;

<img src="C:\Users\ghage\Documents\Mathematics & Statistics\Curve Shortening\Convex Curve Shortening.assets\op_cl_iter.png" style="zoom:50%;" />



#### Gaussian Convolution

As shown in Mokhtarian, Mackworth (1992), it is possible to achieve curve shortening flow through just the use of a Guassian filter.  Given a curve

$$
\Gamma = \lbrace(x(w), y(w)) \mid w \in [0, 1] \rbrace
$$

where $w$ is the normalised arc length, an *evolved* version of the curve is given by

$$
\Gamma_\sigma = \lbrace (X(u, \sigma), Y(u, \sigma)) \mid u \in [0, 1] \rbrace
$$

where

$$
X(u, \sigma) = x(u) \circledast g(u, \sigma) \qquad Y(u, \sigma) = y(u) \circledast g(u, \sigma)
$$

$g(u, \sigma)$ is the Gaussian of width $\sigma$, and $(A \circledast B)$  denotes the convolution operation. 

Shown below is the curvature for evolutions with various widths,

![](C:\Users\ghage\Documents\Mathematics & Statistics\Curve Shortening\Convex Curve Shortening.assets\log_abs_curv-1623725749062.png)

The right shows the logarithm of the absolute of the summation of the curvature for each width, i.e,

$$
\log \left\vert \sum_u \kappa_\sigma(u) \right\vert
$$

The curvature can be found as

$$
\kappa_\sigma(u) = \frac {X_u(u, \sigma)Y_{uu}(u, \sigma) - X_{uu}(u, \sigma)Y_{u}(u, \sigma)} 
	{(X_u(u, \sigma)^2 + Y_u(u, \sigma)^2)^{3/2}}
$$

Due to discretisation, the total absolute curvature initially increases.  It then drops as the shape becomes less concave and outward facing curves (that have negative curvature) are reduced.  Once the shape is convex, the absolute curvature increases exponentially as the shape is reduced to smaller and smaller circles.



## Enclosed Curve Shortening Algorithm

Restating the desired properties of the enclosed curve shortening flow (ECSF) algorithm:

1. **Subset property:** smaller proportions are a subset of any larger proportion, i.e. the shape that corresponds to $x\%$ of the full-size shape is completely contained with the shape that corresponds to $(x+\epsilon)\%$ of the full-size shape for an arbitrarily small $\epsilon$;
2. **Similarity property:** The shapes retain some level of topological relation to the original shape; and 
3. **Singularity property:** The proportions reduce to a single point corresponding to $\epsilon\%$ of the original shape for an arbitrarily small $\epsilon$.



#### Problem with Conventional Curve Shortening

According to the Gage-Hamilton-Grayson theorem, any shape that undergoes curve shortening will eventually become convex.  However, conventional curve shortening shifts concave curves outward as well as convex curves inward.  Thus, as shown above, any concave shape will first overlap its previous iterations before becoming a convex shape and shrinking to a point.  This violates the subset property of the ECSF algorithm that each iteration is a subset of previous iterations.

#### Inward Vector

Clearly, to satisfy this property, the curve should only move inward.  Assuming a clockwise arc, the inward vector can be found by rotating the tangent vector $90^\circ$ clockwise;

$$
\bf v_{inward}(s) = 
	\left(
		\begin{matrix}
			0 & 1 \\
			-1 & 0 \\
   		\end{matrix}
   	\right)
    \bf t(s)
$$

<img src="C:\Users\ghage\Documents\Mathematics & Statistics\Curve Shortening\Convex Curve Shortening.assets\inward_vector.png" style="zoom:60%;" />

However, since the movements are not weighted by the amount of curvature, the curve will overlap at its sharpest points, as shown below;

<img src="C:\Users\ghage\Documents\Mathematics & Statistics\Curve Shortening\Convex Curve Shortening.assets\inward_vector_development.png" style="zoom:60%;" />



#### Curvature Weighting

To prevent singular points emerging where curvature is sharpest, the update can be weighted by the curvature $\kappa$;

$$
\bf v_{step}(s) = 
	\kappa(s)
	\left(
		\begin{matrix}
			0 & 1 \\
			-1 & 0 \\
		\end{matrix}	
	\right)
	\bf t(s)
$$

<img src="C:\Users\ghage\Documents\Mathematics & Statistics\Curve Shortening\Convex Curve Shortening.assets\weighted_inward_vector.png" style="zoom:60%;" />

The problem with this is now the outward curves have outward facing vectors.  As such, the weighting function needs to be altered such that it remains non-negative.  Furthermore, the magnitude of the weighting function at points where the curvature is negative should be scaled down, so that parts of the edge that curve outwards (i.e. have negative curvature) move inwards much less (or not at all) compared to points with inward curve.

Three possible scaling functions are the **exponential linear unit**, **softplus** and **sigmoid**.



##### Exponential Linear Unit (ELU)

The ELU function remains linear for positive inputs, while negative inputs decay at the rate of an inverse exponential;

$$
\text{ELU}(x) = 
	\begin{cases}
		\alpha x &\text{if} \qquad x > 0 \\ 
		a (\exp(x) - 1) &\text{otherwise} \\
    \end{cases}    
$$

$\alpha$ controls the gradient of the linear part of the function for positive inputs.  Setting $a$ to be non-positive ensures the function is non-negative.



##### Softplus

The softplus function is continuous, non-negative approximation of a rectifier.  That is, it approximates the non-linear function $f(x) = \max(0, x)$;

$$
\text{softplus}(x) = a \log\left(1 + \exp\left(\frac {\alpha x} a \right) \right)
$$

$\alpha$ determines the gradient of the linear component of the function.  $a$ determines the sharpness of the curve at zero; 

$$
\lim_{a \rightarrow 0} \ \text{softplus}(x) = \max(0, x)
$$

**Sigmoid**

The sigmoid function is a continuous approximation of the Heaviside function, that returns one for positive inputs and zero for negative;

$$
\text{sigmoid}(x) = \frac 1 {1 + \exp\left(-\alpha \left(x - a\right) \right)}
$$

$\alpha$ determines the sharpness of the curve, while $a$ is the midpoint of the functions step.



##### Comparison

Shown below are the three curves in the range $\left[-1, 1 \right]$, and their effect as a weighting function for the heart shape.  Here the input is the normalised curvature, 

$$
\kappa_{norm}(s) = \frac {\kappa(s)} {\max \left( \lvert \kappa \rvert \right)}
$$

 ![](C:\Users\ghage\Documents\Mathematics & Statistics\Curve Shortening\Convex Curve Shortening.assets\scaling_func_comp-1624543124641.png)

Since the sigmoid function limits the positive curvature to one, it produces the most stable results.  Below is a comparison of the scaling functions for the curvature along the arc;

<img src="C:\Users\ghage\Documents\Mathematics & Statistics\Curve Shortening\Convex Curve Shortening.assets\scaled_curvature.png" style="zoom:60%;" /> 



#### Iterations

We turn now to implementation of ECSF.  The algorithm is an iterative update function.  For each step, the normalised curvature at each point, $\kappa_{norm}(s)$ is calculated.  This is then scaled by one of the scaling functions discussed above.  This scaling is used to weight the inward vector, $\bf v_{inward}(s)$.  The weighted vectors are passed through a Gaussian filter to smooth any discontinuities.  Finally the shape is shifted in the direction of the filtered vectors, with the magnitude determined by a step size coefficient.  This is summarised below;

$$
\begin{align}
	\kappa_{norm} &= \frac {\kappa} {\max \left( \lvert \kappa \rvert \right)} \\ \\
	\kappa_{scaled} &= f_{scaling}\left(\kappa_{norm}\right) \\ \\
	\bf v_{inward} &= 
		\left(
		\begin{matrix}
			0 & 1 \\
			-1 & 0 \\
		\end{matrix}	
		\right)
		\bf t \\ \\
	\bf x^{(i+1)} &= \bf x^{(i)} + 
		\beta_{step} \left(\kappa_{scaled} \bf v_{inward} \circledast \bf 
		g\left(\bf s , \sigma\right)\right)
		
\end{align}
$$


Shown below are the first 400 iterations of the algorithm with the different scaling methods.  Also shown is the total curvature for each iteration, $\sum_s \kappa^{(i)}(s)$.  For each scaling function, the total curvature initially grows almost linearly, as the shape becomes more convex.  However, at some point the iterative total curvature starts to oscillate, implying instability.  This is because the points in the shape are getting closer together with each iteration, in some places overlapping.

![](C:\Users\ghage\Documents\Mathematics & Statistics\Curve Shortening\Convex Curve Shortening.assets\scaling_func_iterations-1624543149372.png)



This is implied by the graphic below, which shows the mean distance between neighbouring vertices in the curve at each iteration.  For each scaling method, the curvature instability occurs when the mean edge length goes a small amount below one.

<img src="C:\Users\ghage\Documents\Mathematics & Statistics\Curve Shortening\Convex Curve Shortening.assets\mean_edge_length.png" style="zoom:100%;" />



#### Resampling

The above instability is due to the vertices of the curve bunching together.  The image below shows the vectors of individual vertices at different iterations of the algorithm.  Initially the vertices are uniformly spaced.   In later iterations, however, they are more clustered at points of high curvature.  Eventually the vertices become so close their vectors cross, leading to an overlap in the curve and instability that grows out of control, shown on the right.

<img src="C:\Users\ghage\Documents\Mathematics & Statistics\Curve Shortening\Convex Curve Shortening.assets\vertex_step_vectors.png" style="zoom:60%;" /> 



To prevent this, the curve can be resampled at each iteration, so as to maintain a uniform arc length between vertices.  Let $\bar s^{(i)}$ be the mean arc length between consecutive vertices for the $i$-th iteration of the algorithm, so that

$$
\bar s^{(i)} = \frac {L^{(i)}} {n^{(i)}}
$$

where $L$ is the total length of the curve and $n$ is the number of vertices.  To maintain a constant, uniform distance between vertices, the curve is resampled such that the mean arc length remains constant;

$$
\begin{align}
	\bar s^{(i)} &= \bar s^{(0)} \\ 
	\frac {L^{(i)}} {n^{(i)}} &= \frac {L^{(0)}} {n^{(0)}} \\ \\
	\therefore \quad n^{(i)} &= \left\lfloor n^{(0)} \frac {L^{(i)}} {L^{(0)}} \right\rfloor \\
\end{align}    
$$

where $\lfloor \cdot \rfloor$ is the floor function, since $n$ needs to be an integer.  The vertices are then resampled using linear interpolation.  Finally, the result is passed through a Gaussian filter to remove sharp artefacts.  The below code is a brief  implementation of the resampling method.

```python
interp_f = interpolate.interp1d(arc_length.cumsum(), curve, axis=0)
curve = interp_f(np.linspace(arc_length[0], arc_length.sum()-arc_length[0], 
                           int(n_verts0 * arc_length.sum() // arc_length0)))
curve = ndimage.gaussian_filter1d(curve, sigma, axis=0, mode='wrap')
```

![](C:\Users\ghage\Documents\Mathematics & Statistics\Curve Shortening\Convex Curve Shortening.assets\uniform_resampling.png)

The above image shows the resampling method in practice.  The mean arc length can been to be roughly constant until the curve becomes completely convex.  At this point, curvature and arc length begin to drop rapidly. 

It is important to note, once the curve is completely convex, the algorithm can be switched to the faster and more stable traditional curve shortening methods.



#### Discretisation

Another approach to the clustering problem is to occasionally discretise the vertices to integer values.  Each vertex is shifted to its nearest integer neighbour.  Duplicate values are removed, whilst making sure the order of the edges is maintained.  This discretisation can be done every set number of steps, or when the mean arc length falls below a threshold.  This method was found to be less stable than resampling.  Below is a Python function that implements discretisation.

```python
def discretise(edge):
    edge = np.around(edge)
    _, idx = np.unique(edge, axis=0, return_index=True)
    return edge[np.sort(idx)]
```



## Future Work

- step size
  - Too large of a step size will lead to discontinuities in the curve which will propagate as instability.
- Median Filtering 
- [Semidiscrete Geometric Flows of Polygons](C:/Users/ghage/Documents/Mathematics & Statistics/Curve Shortening/semidiscrete_curve_shortening.pdf)
- [Solitons_of_Discrete_Curve_Shortening.pdf](file:///C:/Users/ghage/Documents/Mathematics & Statistics/Curve Shortening/Solitons_of_Discrete_Curve_Shortening.pdf)
- Binary opening
- Multiplicative update
- 3D





# Appendix

## Appendix A - Convex Curve Shortening

Once the curve is completely convex, the remaining iterations of the algorithm can be more efficiently carried out using conventional curve shortening flow methods, such as Mokhtarian, Mackworth (1992).

Each iteration is found by applying a Gaussian filter with progressively larger standard deviation.  The relation between the Gaussians standard deviation and the mean radius of the curve, $\bar R_X$;

$$
\bar R_X = \frac 1 {\lvert X \rvert}\sum_s \lVert X(s) - \frac 1 {\lvert X \rvert}\sum_{s'} X(s')\rVert_2
$$

was found to closely match the positive side of a Normal distribution, scaled by the initial average radius of the curve;

$$
\mathcal N(x; \sigma) = \bar R_X^{(0)} \exp(-\frac {x^2} {2\sigma^2}), \quad \sigma = \frac \pi {20}
$$

<img src="C:\Users\ghage\Documents\Mathematics & Statistics\Curve Shortening\Convex Curve Shortening.assets\convex_approx.png" style="zoom:60%;" />

Thus, the inverse of the scaled normal distribution can be used to find values of the Gaussian standard deviation that give linear iteration curves between the original convex curve and the  singularity;

$$
\sigma_{step}^{(i)} = \left\lvert X^{(0)}\right\rvert \sqrt{2\left(\frac {\pi} {20}\right)^2 \ln \left(\frac {\left\lvert \bar R_X^{(0)} \right\rvert} {n_{step}}\right)} \ , \quad 0 \lt n_{step} \le \left\lvert \bar R_X^{(0)} \right\rvert
$$

where $\left\lvert \bar R_X^{(0)} \right\rvert$ is the average radius of the original curve.  A value of $n_{step}$ approaching  0 will give almost the original curve, while anything above 1 will be almost singular.  Values of $n_{step}$ in the range $(0, 1]$ give linear steps toward the singularity.



# Bibliography

- https://en.wikipedia.org/wiki/Curve-shortening_flow
- https://math.mit.edu/research/highschool/primes/materials/2017/conf/7-1-Cohen-Singh.pdf
- https://en.wikipedia.org/wiki/Finite_difference
- https://paperswithcode.com/method/softplus
- https://en.wikipedia.org/wiki/Rectifier_(neural_networks)
- Mokhtarian, Farzin & Mackworth, Alan. (1992). A Theory of Multiscale, Curvature-Based Shape Representation for Planar Curves. Pattern Analysis and Machine Intelligence, IEEE Transactions on. 14. 789-805. 10.1109/34.149591. 
