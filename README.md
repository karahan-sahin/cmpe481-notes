# Final Notes

## TODOs

- [x] Add sillhoutte coeff. details
  
- [ ] Learn to calculate eigen(vector/values)
  
- [ ] Check Spectral Clustering Algorithm Again
  
- [ ] Check Post-Pruning Algorithm Again
  

## Clustering

- Can we find?
  
  - Groups among obs.
    
  - Strange situation (**outliers**)
    
  - Relations among feats
    
  - Important factors
    
- **Visualization helps knowledge discovery**
  

<img src="file:///home/karahan/snap/marktext/9/.config/marktext/images/2022-12-31-13-16-21-image.png" title="" alt="" data-align="center">
- How many fish are there?

### Recommendation Systems

![](file:///home/karahan/snap/marktext/9/.config/marktext/images/2022-12-31-13-17-19-image.png?msec=1672605311550)

- Online shopping &rarr; identify shopper groups (similar browser)
  
- search results based on similar search patterns
  
- Color quantization (image reduction)
  

#### Data charactestics

1. Ball Shape
  
  ![](file:///home/karahan/snap/marktext/9/.config/marktext/images/2022-12-31-13-21-10-image.png?msec=1672605311434)
  
2. Elongated
  
  ![](file:///home/karahan/snap/marktext/9/.config/marktext/images/2022-12-31-13-21-26-image.png?msec=1672605311435)
  
3. Compact but not well separated
  
  ![](file:///home/karahan/snap/marktext/9/.config/marktext/images/2022-12-31-13-21-56-image.png?msec=1672605311435)
  

### 1. K-means Clustering

#### Definition

> **Clustering**: set of techniques for finding subgroups or clusters in a dataset

- Partition into distinct groups &rarr; similar obsv. into same groups
  
- **Unsupervised Problem**
  

> **k-Means:** partitions dataset into <mark>**k-distinct**, **non-overlapping**</mark> clusters

- Steps:
  
  - First specify **k**
    
  - Then will assign all data into one-of-k
    
- Let C<sub>1</sub> .. C<sub>k</sub> denote set of indices of the obsv. in each cluster
  
  - Belong only cluster

> Good clustering is one for the which the <mark>within-cluster variation</mark> is as <mark>small as possible</mark>

> **within-cluster variation**: Measure of the obsv. within a cluster differ from each other

$$
minimize_{C_1,..,C_k}\{\sum^K_{k=1}W(C_k)\}
$$

- Where $W(C_k)$ is a within-class variation of cluster $C_k$
  
- The within-cluster variation for the $k_{th}$ cluster is
  
  - Sum of the **all the pairwise squared Euclidean distance** between
    
    - the obsv. in the k cluster,
  - Divided by the total bumber of obsv. in the k_th cluster
    

$$
W(C_k) = \frac{1}{|C_k|} \sum_{i,i' \in C_k} \sum^p_{j=1} (x_{ij} - x_{i'j})^2
$$

- **For all clusters:**

$$
\frac{1}{|C_k|} \sum_{i,i' \in C_k} \sum^p_{j=1} (x_{ij} - x_{i'j})^2 = 2 \sum_{i \in C_k} \sum^p_{j=1} (x_{ij} - \bar{x}_{kj})^2 
$$$$
\bar{x}_{kj} = \frac{1}{|C_k|} \sum_{i \in C_k} x_{ij}
$$

- is the mean for the feature j in cluster $C_k$

#### Objective Function

> **Objective Function**: Total within class variation for all clusters $C_k$

$$
J = \sum_{n=1}^{N}\sum_{k=1}^{K} r_{nk} || x_n - \mu_k||^2
$$

- **Where**:
  
  - N: Number of samples
    
  - K: Number of clusters
    
  - $r_{nk}$: Cluster assignment if sample $n \in C_k$
    
- Very simple algorithm provides a local optimum to the k-Means problem
  
- Guarantees to decrease the value of obj. func. in each step
  

#### Training

```javascript
assigned_clusters <- map(random(K) , obsv) // or pick random K as heads

while (prev - new) > 0:
    for each K:
        P <- compute_centroids()
        for each obsv:
            assign_closest(obsv, P) 
```

- Loss plot

![](file:///home/karahan/snap/marktext/9/.config/marktext/images/2022-12-31-14-03-08-image.png?msec=1672605311435)

- k-Means is a
  
  - <mark>local search procedure</mark>,
    
  - the <mark>**final cluster centers** highly depend on the **initial cluster centers**</mark>
    

#### Drawbacks

1. Assumes clusters are convex
  
  ![](file:///home/karahan/snap/marktext/9/.config/marktext/images/2022-12-31-14-06-26-image.png?msec=1672605311435)
  
2. It responds poorly to **elongated clusters/ irregular shape**
  
  ![](file:///home/karahan/snap/marktext/9/.config/marktext/images/2022-12-31-14-07-18-image.png?msec=1672605311435)
  

#### Evaluation of the Clustering Quality

- How do measure the quality of k-Means
  
- If class labels are **not known**
  
  - **Silhoutte Co-efficient**
    
  - Calinski-Harabaz Index
    
  - Davies-Bouldin Index
    
- If class labels are **known**
  
  - Adjusted Rand Index
    
  - Mutual Information based Scores
    
    - Normalized MI
      
    - Adjusted MI
      
  - Homogenity, Completeness and V-measure
    
  - Fowlkes-Mallows Scores
    
- If the gr. labels are not knwon, eval must be use the model itself
  

> **Silhoutte coefficient** is composed of two scores:
> 
> - a: The mean distance between **a sample** and **all other points in the <mark>same class</mark>**
>   
> - b: The mean distance between a **a sample** and **all other points in the <mark>next nearest cluster</mark>**
>   

$$
s = \frac{b-a}{max(a,b)}
$$

- Mean silhoutte coff. for each sample is the sil. coeff or a set of samples
  
- **Higher silhoutte coeff. score means better clusters**
  
- range(sil) = [-1, 1]
  

#### To-do: Add Sill. Details

- Advantages
  
  - The **score is higher** when **clusters are dense and well separated**
- Disadvantages
  
  - The silhouette coefficient is generally **higher for convex clusters**

### 2. Spectral Clustering

> Spectral clustring is a **graph-based algorithm** for clustering data points

![](file:///home/karahan/snap/marktext/9/.config/marktext/images/2022-12-31-14-47-42-image.png?msec=1672605311435)

#### 2.1 𝜀-neighborhood graph

#### Definitions

##### 1. Similarity Matrix (S)

> **Similarity matrix:** stores pairwise similarities of data points $s_{ij}$
> 
> - Values are:
>   
>   - **non-negative**
>     
>   - **symmetric**
>     
> - Edge Values == Similarity
>   

$$
\begin{bmatrix}  
s_{AA}=0 & s_{AB} & s_{AC}\\  
s_{BA} & s_{BB}=0 & s_{BC}\\  
s_{CA} & s_{CB} & s_{CC}=0 
\end{bmatrix}
$$

![](file:///home/karahan/snap/marktext/9/.config/marktext/images/2022-12-31-15-19-35-image.png?msec=1672605311436)

##### 2. Degree of a Vertex (d<sub>i</sub>)

> **Degree of a Vertex:** Sum of similarities for a vertex

$$
d_i = \sum_{j=1}^{n} s_{ij}
$$

![](file:///home/karahan/snap/marktext/9/.config/marktext/images/2022-12-31-15-22-42-Screenshot%20from%202022-12-31%2015-22-32.png?msec=1672605311436)

##### 3. Degree Matrix (D)

> Degree Matrix is consist of diagonal values of degree of vertices

$$
\begin{bmatrix}  
d_A & 0 & 0\\  
0 & d_B & 0\\  
0 & 0 & d_C  
\end{bmatrix}
$$

##### 4. Graph Laplacian Matrix (D)

> **Unnormalized Graph Laplacian Matrix** is the difference between the degree martix and the similarity matrix
> 
> - $L = D - S$

$$
\begin{bmatrix}  
d_A-s(AA) & 0-s(AB) & 0-s(AC) \\ 
0-s(BA) & d_B-s(BB) & 0-s(BC) \\ 
0-s(CA) & 0-s(CB) & d_C-s(CC) 
\end{bmatrix}
$$

- **Property!!**: The **<mark>number of zero eigenvalues</mark>** of a Laplacian matrix <mark>**is equal to**</mark> the **<mark>number of connected components</mark>**

![](file:///home/karahan/snap/marktext/9/.config/marktext/images/2022-12-31-15-55-04-image.png?msec=1672605311436)

##### 5. Eigengap Method

- In linear algebra, eigengap is defined as the
  
  > **eigengap**: the diference between two successive eigenvalues, where eigenvalues are sorted in ascending order
  
- Determining the number of clusters (k), the position of the largest gap between sorted values, $|\lambda_k - \lambda_{k+1}|$ can be choosen as **k**
  

![](file:///home/karahan/snap/marktext/9/.config/marktext/images/2022-12-31-16-02-48-image.png?msec=1672605311436)

#### Similarity Graphs

> - **Connects all points** whose **pairwise distances** are smaller than ε
>   
> - Usually considered as an **unweighted graph**
>   

![](file:///home/karahan/snap/marktext/9/.config/marktext/images/2022-12-31-15-48-22-image.png?msec=1672490902533?msec=1672605311436)

- **Choose ε** such that the **resulting graph is safely connected**
  - Smallest &epsilon;
    - Longest path $ε = min(s_ij)$
    - Length of the **longest edge**
      - **in a minimal spanning tree**
      - of the **fully connected graph**
      - on the data points.

##### Disadvantages

- If data contains **outliers**, this method choose large ε, so that they will connect as well
  
- Also if data contains **several tight clusters** which are very far
  apart from each other.
  

##### k-Nearest Neighbour Graphs

- Steps
  
  - Connect vertex i with vertex_j if vertex_j is among the k-nearnest neighbors of the vertex_i
    
  - Weight the edges by the similarity of their endpoints
    
- However, this definition leads to a **directed graph**, as the **neighborhood
  relationship is not symmetric**
  
- k is usually chosen as **log(n)** where **n = number of data points**
  
- the connectivity parameter (k) should be chosen such that the resulting graph is connected
  
- if the similarity graph contains more connected components than :
  
  - the number of clusters we ask the algorithm to detect,
    
  - then **spectral clustering** will trivially return **connected components as clusters**
    
- As the graph should represent the **local neighborhood relationships**, the similarity function itself **models local neighborhoods**
  
- Commonly used similarity function is the Gaussian similarity function where the
  parameter 𝜎 controls the width of the neighborhoods
  

##### Gaussian Similarity Function

- Function for Point Similarity
  
  - 0-1 range

$$
s_{ij} = e^{-\frac{||x_j - x_j||^2}{2\sigma^2}}
$$

- Where $||x_j - x_j||^2$ is the Euclidean Distance
  
- $\sigma$ is control parameter controls the **width of the neighbors**
  
  - Smaller the $\sigma$, less weight to distant neighbors

![](file:///home/karahan/snap/marktext/9/.config/marktext/images/2022-12-31-16-31-57-image.png?msec=1672605311436)

- How to choose $\sigma$?
  
  - in the **order of the mean distance of a point to its kth nearest neighbor**,
    where k is chosen similarly as above (e.g., k ∼ log(n) + 1)
    
  - Another way is to determine ε by the minimal spanning tree heuristic, and then
    choose σ = ε
    

#### Algorithm

- Inputs:
  
  - Similarity Matrix: $S \in \mathbb{R}^{n \times n}$
    
  - Number of clusters: k
    
- Outputs:
  
  - Clustered Sets $A_1, ... ,A_k$

1. Construct a similarity graph G with similarity matrix S
2. Calculate Laplacian Matrix L of G
3. Compute the first k eigenvectors of u_1, ... u_k of the Laplacian Matrix
4. Let $U \in \mathbb{R}^{n \times k}$ be the matrix containing eigenvectors as columns
5. Let $y_i$ be the vector corresponding to the $i^{th}$ row of $U.y_i \in \mathbb{R}^{1 \times k}$
6. **Cluster** the points $y_1, ... y_n$ with the k-means algorithm into clusters $C_1, ... ,C_k$
7. Return cluster sets $A_1, ... ,A_k$ with $A_i = \{ j | y_j \in C_i\}$

## Decision Trees

![Screenshot from 20221231 163635png](file:///home/karahan/Documents/Masters Courses/CMPE481 /STUDY/img/Screenshot%20from%202022-12-31%2016-36-35.png?msec=1672605311437)

- Discrete vs Continouos

- Each decision &rarr; attribute (feature)
  
- Each branch → attribute (feature) **value**
  
- Each leaf → classification
  

**Univariate Decision Tree**

- Decision nodes are split the data along one axis (from [x_0, ..., x_n], only x_n is important )
  
- Splits are orthogonal(dik) to the x/y axis
  

##### Example

![](file:///home/karahan/snap/marktext/9/.config/marktext/images/2023-01-01-11-20-47-image.png?msec=1672605311437)

- $$
  B_1 = - ((4/8)\log_2 (4/8) + (4/8)\log_2 (4/8)) = -(1/2)\frac{2}{3} + -\frac{2}{3}  \\
  B_2 = - (\log_2 (1/8) + \log_2 (7/8)) \\
  B_3 = - (\log_2 (1)) = 0\\
  $$
  
  2log2(1) = 0
  
- log2(2) = 1
  
- log2(7) = 1
  
- log2(7) = 2.807355
  
- log2(3) = 1 ->
  

> **Information Gain (IG)**: is the metric used to determine the split point
> 
> - **the entropy reduction after a split**
>   
>   $$
>   IG = Entropy(Data) - \sum_{i=left,right}\frac{|S_i|}{|Data|}Entropy(S_i)
>   $$
>   - where
>     
>     - i -> all splits

#### Categorical Features

##### Algorithm

1. For each feature, calculate and get highest IG `abs(E_0 - E_1)`
  
  1. ```
    root <- argmax(IG(base, feats))
    ```
    
2. **RECURSIVELY**, For each value of a categorical feature
  
  1. Find a feature with highest IG and place feats as brach
    
  2. If Selected Subset has E(f) = 0 -> Classify
    

#### Gini Index

- Another commonly used measure to form DTs is the Gini Index

> Measures expected err.
> 
> - Expected rate of misclassification
> 
> $$
> Gini = \sum_{i=1}^k p_i(1-p_i)= \sum_{i=1}^k p_i - \sum_{i=1}^k p_i^2 = 1 -\sum_{i=1}^k p_i^2
> $$

###### Difference between Gini and Entropy

|     | Entropy | Gini |
| --- | --- | --- |
| Impure | 1   | 0.5 |
| Pure | 0   | 0   |
| Performance |     | Faster |

#### Regression with Decision Trees

> Regression is the est. of continouos values such as
> 
> - rent
>   
> - age
>   
> - price...
>   
> 
> It is a **function fitting problem**
> 
> ![Screenshot from 20230101 125015png](file:///home/karahan/Documents/Masters Courses/CMPE481 /STUDY/img/Screenshot%20from%202023-01-01%2012-50-15.png?msec=1672605311437)

- Various functions to fit

![](file:///home/karahan/snap/marktext/9/.config/marktext/images/2023-01-01-12-50-30-image.png?msec=1672605311437)

> **Regression tree**
> 
> - Fit piece-wise linear functions to the points
>   
> - Almost same manner as clf. tree
>   
> - But impurity measure is appr. to regression
>   
> - **MEAN SQUARE ERROR**
>   
>   $$
>   E_{MSE} = \frac{1}{N} \sum_{t=1}^N (y_i - g)^2
>   $$

##### Algorithm

```
1. find mean vector (linear horizontal line) of training data
2. MSE(mean, data)
3. (FOR EACH ATTRIBUTE) If MSE is not acceptable (?) 
    1. Data split node is further splitted into argmin(MSE)
    2. If MSE < Threshold -> Leaf == MEAN(SAMPLE) 
        
```

##### Overfitting

- DT Perfectly Classifies training also **learns noise**
  
- Train Acc high but generalization low
  
  - Test Acc low

##### Bias-Variance Trade-Off

> **Bias**: Generalization error due to **wrong model assumptions**:
> 
> - linear model when quadratic problem
>   
> - high bias -> **underfit**
>   

> **Variance**: extreme model sensitivity to small variation in training
> 
> - **complex model**
>   
> - high varinace -> **overfit**
>   

> **Error from Noise**: problematic data

- &uarr; complexity := ↑ variance, ↓ bias
  
- &darr; complexity :=&darr; variance,&uarr;bias
  

> $$
> Error = Variation + Bias^2
> $$

#### Avoid Overfitting (Pre-pruning)

> Pre-pruning: A node is not split further if the number of training instances reaching a node is smaller than a certain percentage of the training set
> 
> - **Stopping tree construction early** is called pre-pruning the tree
>   
> - For example, 5%
>   

#### Avoid Overfitting (Post-pruning)

- In post-pruning, we grow the tree full
  
  - We then find subtrees that cause overfitting and we prune (delete) them

```js
train, val <- split(dataset) // val == pruning set


while Err(Train) > Err(Val):
    R
```

- **Post-pruning** generates simpler trees, which in practice **works better than pre-pruning**

#### Advantages of DT

- Simple to understand and interpret
  
- Can be visualized
  
- Require little data prep
  
  - Other req normalization
- Cost of using tree is `O(log(n))`
  
- Both used in
  
  - numerical
    
  - categorical
    

#### Disadvantages of DT

- Can create over-complex trees
  
  - Does not generalize well (overfit)
    
  - Solutions:
    
    - Pruning
      
    - Min sample req for node
      
    - Max depth param
      
- Can be unstable
  
  - small variation -> very diff tree
    
  - mitigated by ensemble trees
    
- Cannot guarantee global optimum
  
  - Also ensemble and random sampling
- Can be biased trees if some classes are dominated
  
  - Balance prior
- If single training point marked * were instead slightly lower marked +,
  
  - the resulting tree and decision regions would differ significantly

### Random Forests

> **Random Forest**: Many different decision trees and outputs a decision based on the combination of these decision tree outputs

#### Algorithm

```js
samples <- split(M, dataset) // with replacement

k = features
for each samples:
    // Subspace sampling (REDUCES TRAINING TIME)
    d <- select(k) // with replacement |d| ~= \sqrt(|k|) 

    train(sample, d) // without pruning

final_pred <-vote(train(M))
```

##### Hyperparameters

- M : number of decision trees
  
  - set as large as possible
- d: number of selected features
  
  - $|d| \approx \sqrt{|k|}$

#### Bagging

- Random Forest &rarr; Bagging-based ensemble construction algorithm
  
- Ensemble clf (or reg.) are collection of indv. clfs
  
- Bagging is a popular way of ensemble conclf.
  
  - Named after `BOOTSTRAP AGGREGATION`
    
  - In statistics, resampling with replacement is called `BOOTSTRAPPING`
    
- Aggregation can be done via
  
  - classification: majority voting/ soft voting
    
  - regression: averaging / median
    
- Rather single clf., ensemble has
  
  - Similar bias
    
  - Low variance
    

##### Why DT in Bagging

- DT are suited for bagging
  
- DT is very sensitive to data change
  
  - Root selection can result in ripple effect

#### Advantages of Random Forest

- Reduce varianve
  
- Variance output can be considered as uncertainty
  
- Can output `out-of-bag error` which is an estimate of test error
  
  - out-of-bag error is computed by **the samples not selected during the training** of an individual classifier
- Don't need val set
  
- on the average, 36.79% of the training samples are in the out-of-bag set
  
  - for large training sets
    
  - $$
    p(!select) = (1 -\frac{1}{n})^n
    $$
- Used to compute <mark>**relative feature importance**</mark>
  
  - Nodes using the feature can **<mark>how much reduce impurity on average</mark>**

## Dimensionality Reduction

- Many problems have large features
  
  - Learning phase extremely slow
    
  - Harder to find good solution
    
- `curse of dimensionality`
  
- Possible to reduce
  
- Might cause info los
  
  - Image Compression
    
  - Speed up training but performance slightly worse
    
  - But also **filter out noise**
    
- Also useful for visualization
  
- Red. down to two (or three)
  
  - makes possible to plot
    
  - important insight
    
- High dim data -> `risk of sparsity`
  
  - most inst. are likely to be far away
    
  - makes new pred less reliable
    
- In general
  
  - more dim. -> more risk of overfit
    
  - solution: increase train data size
    
  - in practive <mark>**# of data req is exponential to # of dims**</mark>
    

> **Two Main Approches**
> 
> 1. **Projection-based Approaches**
>   
> 2. **Manifold Learning Approaches**
>   

### Projection Approaches

- Instances are not spread out uniformly across all dimensions
  
  - Many are constant, while others are highly correlated
- Res: all instances **lie within (or close to) a much lower-dimensional subspace of the high-dimensional space**
  

### Manifold Learning

- Project not always best
  
- Might req. subspace to twist and turn
  
- Ex: Swiss roll dataset
  
- Many dimensionality reduction algorithms work by modeling the manifold on which the training instances lie
  

### Principal Component Analysis (PCA)

- **Principal Component Analysis (PCA)** is a popular dimensionality reduction algorithm
  
- First it identifies the **hyperplane that lies closest to the data**, and then it projects the data onto it:
  

- Preserving the Variance
  
- Before projecting -> first choose right plane
  
  - Ex: Simple 2D dataset
    
  - Projejct onto solid line **preserves max. variance**
    
    - since most likely to lose less info than others
      
    - Another justification: minimizes MSD between original data and its projection
      
- Two different objectives are equivalent
  
  - Maximizing the `VARIANCE` == Minimizing the `RECONSTRUCTION LOSS`
  - Visual intuition

- $||x||^2 = r^2 + d^2$
  
- $||x||^2$ is constant for the dataset:
  
  - Rotating the projection axis does not change it (vector len)
    
  - So `max(d^2)` and `min(r^2)`
    
- PCA finds the axis that accounts for the **largest amount of variance** in the data set.
  
- $c_1$ = first component
  
- $c_2$ = second component
  
- $c_1 \perp c_2 $
  

#### Derivation

- Center by take the mean from all points

$$
x_i = x_i - \mu_X
$$

- Project x to the axis (by product with w<sup>T</sup>)

$$
z_i = w_1^T x_i
$$

- Variance is equal to w

$$
Var(z_1) = w_1^T \Sigma w_1
$$

- We seek $w_1$ such that $Var(z_1) = w_1^T \Sigma w_1$ is maximized subject to the constraint
  
  - For a unique solution $w_1$ should be **unit vector**
- **Lagrange problem**
  
  $$
  w_1^T \Sigma w_1 - \alpha (w_1^Tw_1 - 1)
  $$
  
  - Take derivate with respect to $w_1$ and setting it equal to 0:
  
  $$
  2\Sigma w_1 - 2\alpha w_1 = 0 = \Sigma w_1 = \alpha w_1
  $$
- which holds IFF
  
  - $w_1$ is an eigenvector of $\Sigma$
    
  - $\alpha$ is eigenvalue of $\Sigma$
    

#### Projection

- We def. $z= W^T (x-m)$ where W are the leading eigenvectors of the covariance matrix $\Sigma$
  
  - Subtract mean *m* before project to the center of data on the origin
    
  - After linear transformation we get k dimensional space whose
    
    - dims are the eigenvectors
      
    - variances are of them are the eigenvalues
      
  - PCA centers the sample
    
  - PCA rotates the axes to line up with highest variance direction
    

## Eigenvalues and Eigenvectors:

- Remembering
  
  - (fact to matrices are linear transformation)
    
  - (eigenvecs are the points where span is not changed) (??)
    
  - Given a matrix `A` :
    
    - $A\vec{x} = \lambda \vec{x}$
      
    - $\vec{x}$: eigenvector
      
    - $\lambda$ : eigenvalue
      
- Examples
  
  $$
  \begin{bmatrix}  
  6.88 & 4.37 \\  
  4.37 & 5.57 \\  
  \end{bmatrix} - \begin{bmatrix}  
  \lambda & 0 \\  
  0 & \lambda \\  
  \end{bmatrix} = \begin{bmatrix}  
  6.88-\lambda & 4.37-0 \\  
  4.37-0 & 5.57-\lambda \\  
  \end{bmatrix} = 0
  $$

$$
DET(A-\lambda I) = (6.88-\lambda) (5.57-\lambda) - (4.37)(4.37) \\
= (38.3216) - 12.45\lambda + \lambda^2 - 19.0969 \\
= \lambda^2 - 12.45\lambda + 19.2247
$$

### Visualization of High Dim. Data

- Same stats, different visualization

- Second Method
  
  - Manifold Based data
- Swiss roll ->
  
  - Geodesic Distance > Euclidean distance
    
  - Still not reliable
    

#### Similarity Between Points

- Similar as the Spectral Clustering
  
- Gaussian Kernel
  

$$
p_{ij} = \frac{exp(\frac{- || x_i - x_j ||^2}{2\sigma^2})}{\sum_i \sum_{j\neq } exp(\frac{- || x_i - x_j ||^2}{2\sigma^2})   }
$$

- Norm by pairwise sum
  
- local neighborhood -> gaussian penalizes longer distance exp.
  
- But too costly
  

$$
denom = \sum_{k \in perp} exp(\frac{- || x_i - x_k ||^2}{2\sigma^2_i})
$$

- How to choose perplexity??
  
  - Choose diff perp. for each point according to density
- $p_{ij} \neq p_{ji} $
  
  - since $\sigma_i$ and $\sigma_j$
    
  - symmetrize by
    
    $$
    \frac{p_{ij}+p_{ji}}{2N}
    $$

#### Low Dimensional Space

- USE `STUDENT T-DISTRIBUTION`

$$
q_{ij} = \frac{(1 + ||y_i - y_j||^2)^{-1}}{\sum_k\sum_{l\neq k}(1 + ||y_k - y_l||^2)^{-1}}
$$

- Why not Gaussian kernel
  
- Sİnce student t distribution has heavy tail
  
- Aim:
  
  - Dissimilar points in the original space should be modeled as too far apart in the low dimensional space

- High dim similarities of p_ij should be close to the low-dim similarities q_ij

#### Objective Function

- Diff between two dist. can be calculated as with Kullback-Liebler Divergence

$$
KL(P||Q) = \sum_i \sum_{j \neq i} p_{ij} \log\frac{p_{ij}}{q_{ij}}
$$

- KL penalizes more if
  
  - high sim in high dim -> low sim in low dim
    
  - high p_ij and low q_ij -> bigger score -> big penalty
    
  - low p_ij and high q_ij -> lower score -> small penalty
    
- t-SNE tries to preserve local similarity
  
- Minimize with Gradient descent
  

$$
\frac{\delta C}{\delta y_i} = 4 \sum_{j\neq i} (p_{ij} - q_{ij})(1 + ||y_i -y_j||^2)^{-1} (y_i-y_j)     
$$

- Gradient descent with t-dist can be slow
  
- Barnes-Hut Approximation
  
  - Compute mean of M remote points and its force f to point i
    
  - M x f (Combined force)
    

###

### Algorithm

```js
data <- X
cost_fn(perp)
optims <- T // number of it 
          eta // learning rate
          alpha(t) // momentum


Res: low dim y(T)
    Begi n  
```

#### Notes

- Run t-SNE multiple times and select lowest KL Div (random init)
  
- The performance of t-SNE is robust under different settings of the perplexity
  
  - Once mapped, must be rerun if new point is added
    
    - Iterativee
