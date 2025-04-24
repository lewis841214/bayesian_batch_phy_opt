# Bayesian Optimization for Physical Design Parameter Tuning

## Overview
Industrial physical design (PD) flows present significant optimization challenges due to their complex parameterization. Modern EDA tools contain hundreds of tunable parameters affecting power, performance, and area (PPA) metrics, with complex interdependencies that are difficult to model. Practical optimization in this domain faces several critical constraints:

1. **Large Parameter Space**: Industrial PD flows typically involve a greate amount of  parameters across various stages, creating an enormous design space impossible to explore exhaustively
2. **Limited Evaluation Budget**: Each parameter configuration evaluation requires extensive compute resources and time (hours to days), severely restricting the number of possible trials (typically 10-30)
3. **Concurrent Evaluation Capability**: While multiple parameter configurations can be evaluated in parallel, most organizations have limited compute resources, allowing only small batch sizes (3-10 concurrent runs)
4. **Mixed Parameter Types**: Parameters include continuous (timing margins), integer (effort levels), and categorical (algorithm choices) variables, complicating optimization
5. **Multi-objective Nature**: Optimization must simultaneously consider power, performance, area, and other metrics with complex trade-offs

These constraints make traditional optimization approaches impractical, driving the need for sample-efficient methods that can effectively navigate complex parameter spaces with minimal evaluations while leveraging available parallelism. This document examines various approaches to this challenging parameter tuning problem, with special attention to methods capable of discovering high-quality solutions with extremely limited evaluation budgets.

## Related Works

The optimization of physical design parameters has been approached through various methodologies, each with different trade-offs in terms of data requirements, computational complexity, and effectiveness.

### Reinforcement Learning Approaches

Recent research has demonstrated that Reinforcement Learning (RL) methods represent the state-of-the-art in parameter tuning for physical design. For example, Thomas et al. in ["ML-Based Physical Design Parameter Optimization for 3D ICs"](https://gtcad.gatech.edu/www/papers/dac24-thomas.pdf) (DAC 2024) presented a comprehensive machine learning framework specifically tailored for 3D IC parameter optimization. Their approach achieved significant improvements in Power, Performance, and Area (PPA) metrics through RL-based parameter exploration.

However, RL methods face fundamental limitations in industrial settings with large designs:

1. **Data Requirements**: RL models typically require thousands of training samples (1,000-10,000) to converge effectively
2. **Scalability Issues**: For real in-house designs with 10-100 million cells and nets, generating sufficient training data becomes prohibitively expensive
3. **Runtime Constraints**: Each design iteration can take hours or days, making data collection for RL impractical within project timelines

### Model-Free and Lightweight Approaches

Given these constraints, model-free or lightweight model approaches often demonstrate superior practical performance:

1. **Bayesian Optimization**: Works like "PPATuner: Pareto-Driven Tool Parameter Auto-Tuning in Physical Design" (ACM DAC 2022) demonstrate BO's effectiveness with limited samples. This approach models the objective function with uncertainty quantification, enabling efficient exploration-exploitation trade-offs with fewer evaluations.

2. **Evolutionary Algorithms**: Genetic Algorithms (GA) and Ant Colony Optimization (ACO) provide robust parameter tuning with minimal assumptions about the objective function landscape.

3. **Lightweight ML Models**: Decision trees, XGBoost, and regression-based approaches offer advantages when data is limited due to their lower parameter count and sample efficiency.

Chu and Wang demonstrated in their PPATuner work that Bayesian optimization approaches can efficiently identify Pareto-optimal parameter configurations with substantially fewer samples than required by RL methods, making them more suitable for industrial physical design workflows.

### Leveraging Intermediate Physical Representations

A key limitation of existing approaches is their focus solely on final PPA metrics, ignoring valuable intermediate physical design data:

1. **Rich Intermediate Data**: Physical design tools generate numerous intermediate outputs including 2D power maps, congestion distributions, and wirelength histograms
2. **Causal Mechanisms**: These intermediate results provide insights into the causal mechanisms by which parameters affect final outcomes
3. **Information Density**: Spatial distributions contain significantly more information than scalar PPA metrics

This gap presents an opportunity for approaches that explicitly incorporate physical design knowledge and intermediate representations.

### Our Contribution: Physics-Aware Bayesian Optimization

In this work, we propose a novel parameter optimization method called Physics-Aware Bayesian Optimization (PABO). This framework:

1. Explicitly models the relationship between design parameters and intermediate physical representations
2. Leverages these physical manifestations to more accurately predict final PPA metrics
3. Maintains the sample efficiency of Bayesian optimization while incorporating the expressivity of deep learning
4. Requires significantly fewer samples than RL approaches while achieving comparable or better results
5. Addresses the challenge of discrete parameters through transformer attention mechanisms, eliminating the need to define explicit distance metrics in one-hot encoded spaces that traditional BO methods rely on

By understanding how parameters affect the physical characteristics of the design, PABO provides more interpretable and effective parameter recommendations than methods that rely solely on final PPA values.

## Batch Parallel Parameter Optimization

### Problem Formulation

Given a process $h$ that maps parameters $x \in \mathcal{X}$ to outputs $y = h(x) \in \mathcal{Y}$, where:
- $\mathcal{X}$ is the parameter space (e.g., EDA tool parameters)
- $\mathcal{Y}$ is the output space (e.g., PPA metrics)
- $h$ is an expensive black-box function (e.g., EDA flow execution)

We aim to find the Pareto frontier of $h(x)$ where:
- We can evaluate $b$ parameter configurations in parallel per iteration
- Each evaluation provides one point in the output space

### Methods for Batch Parallel Pareto Optimization

#### 1. Bayesian Optimization with Batch Acquisition

**Mathematical Formulation:**
Let $\mathcal{D}_t = \{(x_i, y_i)\}_{i=1}^t$ be the observations collected up to iteration $t$. The posterior distribution over possible functions $h$ is:

$$p(h | \mathcal{D}_t) \propto p(\mathcal{D}_t | h) p(h)$$

The batch acquisition function selects $b$ points $\{x_{t+1}^{(1)}, \ldots, x_{t+1}^{(b)}\}$ by optimizing:

$$\{x_{t+1}^{(1)}, \ldots, x_{t+1}^{(b)}\} = \arg\max_{x^{(1)}, \ldots, x^{(b)} \in \mathcal{X}} \alpha_t(x^{(1)}, \ldots, x^{(b)} | \mathcal{D}_t)$$

#### Deriving Mean and Variance from Bayes' Rule

The posterior distribution formula $p(h | \mathcal{D}_t) \propto p(\mathcal{D}_t | h) p(h)$ translates into practical computational formulas for Gaussian Processes. Here's how the commonly used mean and variance expressions are derived from Bayes' rule:

1. **Start with Prior**: 
   - Let $p(h)$ be a Gaussian Process with mean function $m(x)$ (often set to 0) and kernel $k(x, x')$
   - This prior defines a distribution over functions $h \sim \mathcal{GP}(m(x), k(x,x'))$

2. **Define probability model and likelihood**:
   - We model the observations $y_i$ through the true function $h$ plus Gaussian noise: $y_i = h(x_i) + \epsilon_i$ where $\epsilon_i \sim \mathcal{N}(0, \sigma_n^2)$
   - The likelihood becomes $p(\mathcal{D}_t | h) = \prod_{i=1}^t \mathcal{N}(y_i; h(x_i), \sigma_n^2)$

3. **Posterior Computation**:
   - Our prior assumes $h \sim \mathcal{GP}(m(x), k(x,x'))$, which means for any point $x_*$ in the domain, $h(x_*)$ is a random variable with distribution $\mathcal{N}(m(x_*), k(x_*,x'))$. After observing data $\mathcal{D}_t = \{(x_i, y_i)\}_{i=1}^t$, we update our beliefs about the entire function $h$, which remains a GP but with updated mean and covariance functions. Below, we compute the posterior distribution of $h(x_*)$ at any new test point $x_*$.

   - When applying Bayes' rule with Gaussian processes, we can derive closed-form expressions
   - The joint distribution of observed targets $\mathbf{y} = [y_1,...,y_t]^T$ and function value $h(x_*)$ at a new test point $x_*$ is:
     $$\begin{bmatrix} \mathbf{y} \\ h(x_*) \end{bmatrix} \sim \mathcal{N}\left(\begin{bmatrix} \mathbf{m} \\ m(x_*) \end{bmatrix}, \begin{bmatrix} K + \sigma_n^2I & \mathbf{k}_* \\ \mathbf{k}_*^T & k(x_*, x') \end{bmatrix}\right)$$
   - Where $\mathbf{m} = [m(x_1),...,m(x_t)]^T$, $K$ is the kernel matrix with $K_{ij} = k(x_i, x_j)$, and $\mathbf{k}_* = [k(x_1, x'),...,k(x_t, x')]^T$

4. **Conditional Distribution**:
   - Using properties of multivariate Gaussians, the posterior $p(h(x_*) | \mathcal{D}_t)$ is:
     $$h(x_*) | \mathcal{D}_t \sim \mathcal{N}(\mu_t(x_*), \sigma_t^2(x_*))$$
   - Where:
     $$\mu_t(x_*) = m(x_*) + \mathbf{k}_*^T (K + \sigma_n^2I)^{-1} (\mathbf{y} - \mathbf{m})$$
     $$\sigma_t^2(x_*) = k(x_*, x_*) - \mathbf{k}_*^T (K + \sigma_n^2I)^{-1} \mathbf{k}_*$$

5. **Simplified Form**:
   - With the common choice of zero mean prior $m(x) = 0$, the expressions simplify to:
     $$\mu_t(x_*) = \mathbf{k}_*^T (K + \sigma_n^2I)^{-1} \mathbf{y}$$
     $$\sigma_t^2(x_*) = k(x_*, x') - \mathbf{k}_*^T (K + \sigma_n^2I)^{-1} \mathbf{k}_*$$

6. **Interpretation of GP Posterior Terms**:
   - Each component in the posterior formulas has an intuitive meaning:
   
   - For the posterior mean $\mu_t(x_*) = \mathbf{k}_*^T (K + \sigma_n^2I)^{-1} \mathbf{y}$:
     - $\mathbf{k}_*$: Vector of similarities between the test point $x_*$ and all training points. The closer $x_*$ with $x_t$, the larger the weight.
     - $(K + \sigma_n^2I)^{-1}$: "Information matrix" that accounts for data correlations and noise. The high correlation poitns $i, j$ will be reduced in weight to avoid "double-counting" information.
     - $(K + \sigma_n^2I)^{-1} \mathbf{k}_*$: Optimal prediction weights for each training point
     - The complete expression: A weighted combination of observed values, with weights determined by both similarity and information content
   
   - For the posterior variance $\sigma_t^2(x_*) = k(x_*, x_*) - \mathbf{k}_*^T (K + \sigma_n^2I)^{-1} \mathbf{k}_*$:
     - $k(x_*, x_*)$: Prior uncertainty at the test point
     - $\mathbf{k}_*^T (K + \sigma_n^2I)^{-1} \mathbf{k}_*$: Uncertainty reduction from observations
     - The complete expression: Remaining uncertainty after incorporating observed data
   
   - **Example: Impact of Correlations on Prediction Weights**
   
     Consider predicting at a point $x_*$ identical to the first of two training points. The matrix $(K + \sigma_n^2I)^{-1}$ handles redundancies, as demonstrated by comparing:
     
     **Case 1: Uncorrelated points** (points far apart):
     ```
     K = [1  0]
         [0  1]
     
     (K + σ²I)⁻¹ ≈ [0.91  0   ]  (assuming σ² = 0.1)
                  [0     0.91]
     ```
     
     **Case 2: Highly correlated points** (points very close together):
     ```
     K = [1   0.9]
         [0.9  1 ]
     
     (K + σ²I)⁻¹ ≈ [ 2.75  -2.25]
                  [-2.25   2.75]
     ```
     
     With $\mathbf{k}_* = [1, 0.9]^T$ (test point similar to both training points), the prediction weights are:
     
     **Uncorrelated case:**
     ```
     (K + σ²I)⁻¹\mathbf{k}_* = [0.91]  (weight for y₁)
                              [0.82]  (weight for y₂)
     ```
     
     **Correlated case:**
     ```
     (K + σ²I)⁻¹\mathbf{k}_* = [0.73]  (weight for y₁)
                              [0.23]  (weight for y₂)
     ```
     
     The weight for similar observations decreases in the presence of correlation, demonstrating how GP automatically prevents "double-counting" information from redundant observations.

These closed-form expressions directly implement the abstract Bayesian posterior $p(h | \mathcal{D}_t)$ for GPs. They are used to:
- Predict function values at new points (using $\mu_t(x)$)
- Quantify uncertainty (using $\sigma_t^2(x)$)
- Define acquisition functions (combining $\mu_t(x)$ and $\sigma_t^2(x)$)
- Select next evaluation point(s) (maximizing acquisition function)

**Key Batch Acquisition Strategies:**

1. **q-ParEGO (q-Pareto Efficient Global Optimization)**:
   - Scalarizes multiple objectives using random weights
   - Optimizes batch of points using q-EI (q-Expected Improvement)
   - Mathematical expression:
     $$\alpha_{\text{q-ParEGO}}(x^{(1)}, \ldots, x^{(b)}) = \mathbb{E}\left[\max_{i=1}^b \max(0, s(y^*) - s(f(x^{(i)})))\right]$$
     where $s$ is a scalarization function with random weights

2. **q-EHVI (q-Expected Hypervolume Improvement)**:
   - Directly maximizes expected improvement in Pareto hypervolume
   - Accounts for interactions between batch points
   - Mathematical expression:
     $$\alpha_{\text{q-EHVI}}(x^{(1)}, \ldots, x^{(b)}) = \mathbb{E}[\text{HV}(\mathcal{P}_t \cup \{f(x^{(1)}), \ldots, f(x^{(b)})\}) - \text{HV}(\mathcal{P}_t)]$$
     where $\mathcal{P}_t$ is the current Pareto front and $\text{HV}$ is the hypervolume indicator

3. **BUCB (Batch Upper Confidence Bound)**:
   - Originally proposed by Desautels et al. (2014) in ["Parallelizing Exploration-Exploitation Tradeoffs in Gaussian Process Bandit Optimization"](https://arxiv.org/pdf/1304.5350)
   - Extends the UCB acquisition function to the batch setting while accounting for the information gain from pending evaluations
   - Uses a sequential greedy selection approach with GP updates using hallucinated observations
   - Mathematical expression for selecting the batch sequentially:
     $$x_{t,j} = \arg\max_{x \in \mathcal{X}} \mu_{t,j-1}(x) + \beta_t^{1/2}\sigma_{t,j-1}(x)$$
     where $\mu_{t,j-1}$ and $\sigma_{t,j-1}$ are the posterior mean and standard deviation after adding $j-1$ fantasy points to the GP model
   - The key insight is updating the GP model with "hallucinated" observations (e.g., setting $y_{t,j} = \mu_{t}(x_{t,j})$) between batch selections, which naturally encourages diversity

### Acquisition Function Tradeoffs for Different Optimization Phases

When selecting acquisition functions for batch parallel optimization, the stage of optimization and available budget significantly impact performance:

1. **Early-stage vs. Late-stage Optimization**:
   - **BUCB (Batch Upper Confidence Bound)** typically performs better in early stages of optimization where exploration is critical. The UCB approach inherently balances exploration and exploitation through its formula $\mu(x) + \beta^{1/2}\sigma(x)$, but with sufficient tuning of $\beta$, it can emphasize exploration of uncertain regions.
   
   - **Expected Improvement (EI)** and its batch variants like **q-EI** tend to perform better in later stages of optimization, as they focus on areas with high probability of improving over the current best solution. This exploitation-focused behavior becomes valuable when refining already-promising solutions.

2. **Budget-constrained Scenarios**:
   - For extremely limited budgets (e.g., only three batch evaluations), **Expected Improvement** methods often outperform UCB approaches. This is because EI's focus on regions likely to immediately yield improvements provides better short-term returns.
   
   - The batch size $b$ also affects this tradeoff. With larger batch sizes, diversity-promoting methods like BUCB become more important to prevent redundant evaluations.

3. **Hybrid Approaches**:
   - For practical applications, adaptive acquisition strategies that transition from exploration-focused (UCB) to exploitation-focused (EI) as optimization progresses can yield superior performance.
   
   - Techniques like GP-Hedge automatically balance multiple acquisition functions by treating acquisition function selection as a multi-armed bandit problem.

This contextual selection of acquisition functions should be considered alongside the problem characteristics and computational constraints when designing batch parallel Pareto optimization strategies for physical design applications.

#### 2. Evolutionary Algorithms for Batch Pareto Optimization

**NSGA-III with Batch Parallelism:**
- Natural parallelism in population-based methods
- Employs reference points in objective space to maintain diversity
- Batch evaluation is inherent as entire population is evaluated per generation
- Mathematical model:
  - Selection: Dominance ranking and reference-point-based niching
  - Reference points on a normalized hyperplane defined by:
    $$\{z \in \mathbb{R}^m | \sum_{i=1}^m z_i = 1, z_i \geq 0 \text{ for } i = 1, \ldots, m\}$$

**MOEA/D (Multi-Objective Evolutionary Algorithm based on Decomposition):**
- Decomposes problem into subproblems solved simultaneously
- Well-suited for batch processing as each subproblem can be evaluated independently
- Mathematical decomposition using the Tchebycheff approach:
  $$g^{te}(x|\lambda, z^*) = \max_{1 \leq i \leq m} \{\lambda_i|f_i(x) - z_i^*|\}$$
  where $\lambda$ is a weight vector and $z^*$ is the reference point

#### 3. Ant Colony Optimization for Batch Pareto Frontier

**Multi-Objective Ant Colony Optimization (MOACO) with Batch Processing:**
- Maintains multiple pheromone matrices, one per objective
- Each ant constructs a solution based on weighted pheromone information
- Batch parallel version dispatches $b$ ants simultaneously
- Pheromone update rule:
  $$\tau_{ij}^k \leftarrow (1-\rho)\tau_{ij}^k + \sum_{a=1}^b \Delta\tau_{ij}^{k,a}$$
  where $\Delta\tau_{ij}^{k,a}$ is the pheromone deposit by ant $a$ for objective $k$

### Surrogate model for better mean estimation

Neural networks can replace traditional kernel-based Gaussian Processes in Bayesian optimization by directly modeling the posterior distribution:

#### Surrogate Model Formulation

The standard GP posterior for a test point $x_*$ provides:
- Mean: $\mu_t(x_*) = \mathbf{k}_*^T (K + \sigma_n^2I)^{-1} \mathbf{y}$
- Variance: $\sigma_t^2(x_*) = k(x_*, x_*) - \mathbf{k}_*^T (K + \sigma_n^2I)^{-1} \mathbf{k}_*$

Neural network surrogate models replace these kernel-based formulations by:

1. **Mean Estimation**: 
   - Direct prediction: $\hat{h}(x_*) = f_\theta(x_*)$ where $f_\theta$ is a neural network
   - The network parameters $\theta$ are trained to minimize $\mathcal{L}(\theta) = \sum_{i=1}^n (y_i - f_\theta(x_i))^2$

2. **Variance Estimation**:
   - Traditional GP formula: $\sigma_t^2(x_*) = k(x_*, x_*) - \mathbf{k}_*^T (K + \sigma_n^2I)^{-1} \mathbf{k}_*$


#### Rationale for Hybrid Mean-Variance Approach

This hybrid approach combines neural network mean estimation with traditional GP-based variance calculation for several key advantages:

1. **Mean Estimation Benefits**:
   - NNs excel at capturing complex, non-stationary patterns in the objective landscape
   - Higher expressivity than kernel methods for modeling intricate physical design relationships
   - Better scalability with increasing dimensionality and data volume

2. **Traditional Variance Benefits**:
   - Maintains the principled distance-based uncertainty quantification of GPs
   - Preserves the critical property that uncertainty increases with distance from observed data
   - Provides theoretical guarantees for exploration in Bayesian optimization

3. **Implementation Considerations**:
   - Kernel matrix $K$ can be computed using simpler kernels since complexity is handled by NN mean
   - Use sparse approximations of $K$ to maintain computational efficiency
   - May use the neural features as inputs to the kernel function: $k(f_\theta(x), f_\theta(x'))$

This approach offers computational advantages for high-dimensional spaces and better captures complex non-stationary functions while maintaining principled uncertainty quantification.


#### Hierarchical Prediction Through Intermediate Results

Traditional approaches directly map parameters to final PPA metrics, but this misses the opportunity to leverage intermediate products of the physical design flow. A more effective approach uses a hierarchical prediction model:

1. **Multi-stage Prediction Pipeline**:
   - Parameters $(x) \rightarrow$ Intermediate products $(z) \rightarrow$ Final PPA metrics $(y)$
   - Formally: $\hat{h}(x) = g_{\phi}(f_{\theta}(x))$ where $f_{\theta}: \mathcal{X} \rightarrow \mathcal{Z}$ and $g_{\phi}: \mathcal{Z} \rightarrow \mathcal{Y}$

2. **Intermediate Products to Model**:
   - 2D power distribution maps across the chip area
   - Congestion heatmaps showing routing resource utilization
   - Wirelength distributions by net type
   - Placement density visualizations
   - Timing path distributions

3. **Benefits**:
   - Better captures complex inter-parameter relationships through physical manifestations
   - Provides interpretable intermediate representations for design insight
   - Enables knowledge transfer across different designs or technology nodes
   - Creates models that respect physical causality in the design process

4. **Implementation**:
   - Use convolutional layers for spatial intermediate products (heatmaps, distributions)
   - Apply attention mechanisms to focus on critical design regions
   - Train with multi-task learning to predict both intermediates and final metrics
   - Potential for self-supervised pretraining on abundant intermediate data

This hierarchical approach significantly improves prediction accuracy for complex physical design problems where parameter interactions manifest through physical intermediate results.

#### Transformer-Based Implementation

Transformer architectures are particularly well-suited for implementing the neural network components in hierarchical prediction models:

1. **Parameter Relationship Modeling**:
   - Self-attention mechanisms excel at capturing complex dependencies between different design parameters
   - Transformers can model both local and global relationships without distance bias
   - Mathematical formulation: $\text{Attention}(Q, K, V) = \text{softmax}(\frac{QK^T}{\sqrt{d_k}})V$

2. **Handling Mixed Data Modalities**:
   - Encode heterogeneous inputs (parameters, constraints, spatial distributions) as tokens
   - Process parameter vectors alongside spatial representations using cross-attention
   - Combine 1D parameter sequences with 2D spatial features through appropriate positional encodings

3. **Advantages for Physical Design Problems**:
   - Captures long-range dependencies between physically distant elements on the chip
   - Naturally models the permutation invariance of certain design parameters
   - Excels at transfer learning from pretrained models on related designs

4. **Specific Applications**:
   - **Parameter → Intermediate**: Transformer encoder predicts spatial distributions from parameter vectors
   - **Intermediate → PPA**: Vision transformer processes spatial heatmaps to predict final metrics
   - **End-to-End**: Single transformer with appropriate encodings handles full hierarchy

5. **Implementation Considerations**:
   - Use efficient transformer variants (e.g., Linformer, Performer) for large spatial representations
   - Employ cross-attention to condition spatial predictions on parameter values
   - Consider hierarchical transformers matching the physical design hierarchy

Transformers' ability to model complex relationships across heterogeneous data makes them ideal for capturing the intricate parameter interactions in physical design optimization.

#### Detailed Transformer Architecture Design

For the specific task of mapping parameters P = [p₁, p₂, ..., pₖ] to intermediate n×n 2D grids and final PPA metrics, we propose a specialized transformer architecture:

```
                                 TRANSFORMER ARCHITECTURE
┌───────────────┐     ┌────────────────────┐     ┌───────────────────┐     ┌──────────────┐
│   Parameters  │     │     Parameter      │     │  Intermediate     │     │    Final     │
│               │     │      Encoder       │     │       Grid        │     │     PPA      │
│ p₁, p₂, ..., pₖ│ → │ TransformerEncoder │ → │ Vision Transformer │ → │   Metrics    │
│               │     │                    │     │      (n×n)        │     │              │
└───────────────┘     └────────────────────┘     └───────────────────┘     └──────────────┘
                              │                            │                        ↑
                              │                            │                        │
                              └────────────────────────────┼────────────────────────┘
                                                           │
                                                  Cross-modal fusion
```

1. **Parameter Encoding Stage**:
   - **Input Embedding**: Each parameter pᵢ is embedded into a d-dimensional space
     ```
     E(pᵢ) = W_e · pᵢ + P_e
     ```
     where W_e ∈ ℝ^(d×1) and P_e ∈ ℝ^d is a learnable position embedding
   
   - **Parameter Transformer**: Standard transformer encoder processes parameter tokens
     ```
     Z = TransformerEncoder([E(p₁), E(p₂), ..., E(pₖ)])
     ```
     
   - **Architecture Details**:
     - Embedding dimension: d = 256
     - Number of heads: h = 8
     - Number of layers: L = 4
     - Add a special [GRID] token to represent the 2D grid prediction task

2. **2D Grid Generation Stage**:
   - **Grid Initialization**: Initialize an n×n grid with learned tokens
     ```
     G₀ ∈ ℝ^(n×n×d)
     ```
   
   - **Cross-Attention**: Apply cross-attention between parameter encodings and grid tokens
     ```
     Attention(Q=G_l, K=Z, V=Z)
     ```
     where G_l is the grid representation at layer l
   
   - **Grid Transformer**: Process with additional self-attention layers
     ```
     G_{l+1} = SelfAttention(G_l) + CrossAttention(G_l, Z)
     ```
   
   - **Architecture Details**:
     - Grid resolution: n = 32 (configurable)
     - Use 2D positional encodings for grid tokens
     - 2D local attention patterns to capture spatial locality
     - Use efficient attention for the n² grid tokens (e.g., axial attention)

3. **PPA Prediction Stage**:
   - **Grid Pooling**: Apply hierarchical pooling to the final grid representation
     ```
     G_pool = HierarchicalPool(G_L)
     ```
   
   - **Cross-Modal Fusion**: Combine parameter encodings with grid representations
     ```
     F = CrossAttention(Z, G_pool) + Z
     ```
     
     Cross-modal fusion refers to the technique of integrating information from two different representation types (or "modalities"):
     * Parameter encodings (Z): Direct representations of the input parameters
     * Grid representations (G_pool): Spatial patterns generated from those parameters
     
     The fusion happens through cross-attention, where:
     * Parameter encodings (Z) serve as the query
     * Pooled grid features (G_pool) serve as keys and values
     * This allows parameter features to "attend to" the most relevant parts of the spatial grid
     * The residual connection (+Z) preserves the original parameter information
     
     This fusion is critical because some PPA aspects might be directly influenced by parameters (e.g., clock frequency) while others emerge from spatial interactions captured in the grid (e.g., power hotspots).
   
   - **PPA Decoder**: Final transformer decoder to predict PPA metrics
     ```
     PPA = MLPHead(TransformerDecoder(F))
     ```
   
   - **Architecture Details**:
     - Multiple prediction heads for different PPA metrics
     - Optional auxiliary losses during training for better gradient flow
     - Uncertainty-aware prediction for variance estimation

4. **Training Strategy**:
   - **Multi-task Learning**: Joint optimization of intermediate grid and final PPA predictions
     ```
     L = λ₁L_grid + λ₂L_PPA
     ```
     where L_grid measures the error in grid prediction and L_PPA in final metrics
   
   - **Curriculum Learning**: Start training on simpler parameter configurations
   - **Transfer Learning**: Pretrain on related physical design datasets

5. **Implementation Efficiency**:
   - Use patch-based processing for the grid (similar to ViT)
   - Employ linear attention variants for the n² grid tokens
   - Optional distillation from larger to smaller models for deployment

This architecture effectively leverages transformers' ability to model complex parameter interactions while efficiently generating the intermediate spatial representations needed for accurate PPA prediction.

#### Rationale for Hybrid Mean-Variance Approach

This hybrid approach combines neural network mean estimation with traditional GP-based variance calculation for several key advantages:

1. **Mean Estimation Benefits**:
   - NNs excel at capturing complex, non-stationary patterns in the objective landscape
   - Higher expressivity than kernel methods for modeling intricate physical design relationships
   - Better scalability with increasing dimensionality and data volume

2. **Traditional Variance Benefits**:
   - Maintains the principled distance-based uncertainty quantification of GPs
   - Preserves the critical property that uncertainty increases with distance from observed data
   - Provides theoretical guarantees for exploration in Bayesian optimization

3. **Implementation Considerations**:
   - Kernel matrix $K$ can be computed using simpler kernels since complexity is handled by NN mean
   - Use sparse approximations of $K$ to maintain computational efficiency
   - May use the neural features as inputs to the kernel function: $k(f_\theta(x), f_\theta(x'))$

This approach offers computational advantages for high-dimensional spaces and better captures complex non-stationary functions while maintaining principled uncertainty quantification.


## Experimental Setup

To evaluate the effectiveness of our Physics-Aware Bayesian Optimization (PABO) approach, we conduct a comprehensive comparison against established parameter optimization methods for physical design.

### Optimization Methods Compared

We compare the following methods:

1. **Ant Colony Optimization (ACO)**: A population-based metaheuristic that models the behavior of ants finding optimal paths through pheromone trails
2. **Standard Bayesian Optimization (BO)**: Classical BO approach using Gaussian Processes with Matérn 5/2 kernel and Expected Improvement acquisition function
3. **FIST** (Optional): Fast Iterative Slicing Technique, a gradient-free optimization method for parameter tuning
4. **PABO (Ours)**: Our proposed Physics-Aware Bayesian Optimization using transformer-based surrogate models with hierarchical prediction

### Experimental Protocol

For each optimization method, we employ the following protocol:

- **Number of Trials**: 3 independent trials with different random seeds to account for the stochastic nature of the algorithms
- **Runs per Trial**: Each trial consists of 10 sequential runs, where one run represents a single parameter configuration evaluation
- **Total Evaluations**: 30 parameter configuration evaluations per method (3 trials × 10 runs)
- **Initialization**: Each method begins with the same 3 randomly selected parameter configurations to establish a baseline
- **Sequential Evaluation**: Subsequent parameter configurations are determined by each method's selection strategy

### Designs Under Test

We evaluate on three representative designs with varying complexity:

1. **Small Design**: ~100K cells, 5 layers, representing a typical block
2. **Medium Design**: ~1M cells, 10 layers, representing a moderate subsystem
3. **Large Design**: ~10M cells, 14 layers, representing a full industrial chip

### Parameter Space

The parameter space consists of:

- **Continuous Parameters**: 8 numerical parameters (e.g., clock uncertainty, target utilization)
- **Integer Parameters**: 5 discrete numerical parameters (e.g., optimization effort level, number of threads)
- **Categorical Parameters**: 7 strategy selection parameters (e.g., placement algorithm, routing strategy)

### Evaluation Metrics

We assess the methods using the following metrics:

1. **Pareto Frontier Quality**: Hypervolume dominated by the discovered Pareto front relative to a reference point
2. **Convergence Rate**: Number of evaluations required to reach within 5% of the best-known solution
3. **Robustness**: Standard deviation of results across trials
4. **Prediction Accuracy**: For PABO, additional evaluation of intermediate prediction quality

### Implementation Details

- **Hardware**: All experiments are conducted on identical servers with 32-core CPUs, 128GB RAM, and NVIDIA A100 GPUs
- **Software**: EDA tool version [specific version] with consistent technology libraries
- **Runtime**: Physical design runs are limited to 8 hours per configuration

The experimental results are presented in the following section, analyzing both the quality of solutions and the efficiency of the optimization process.


## Implementation Plan

Our implementation strategy follows a systematic approach to ensure robust development and evaluation of the proposed Physics-Aware Bayesian Optimization framework.

### 1. Optimization Method Survey

- **Literature Review**: Comprehensive analysis of recent advancements in parameter optimization for EDA flows
- **Method Selection Criteria**: Evaluate methods based on sample efficiency, parallelization capability, and handling of mixed parameter types
- **Baseline Selection**: Identify the most promising existing approaches (BO, ACO, FIST, XGBoost) to serve as benchmarks
- **Performance Metrics**: Define appropriate evaluation metrics for comparing optimization methods in the physical design context

### 2. Implementation

#### 2.1. Data Preprocessing and Specification

- **Data Format Definition**: Establish standardized formats for:
  - Input parameters (continuous, discrete, categorical)
  - Intermediate physical representations (2D grids)
  - Final PPA metrics
- **Grid Size Determination**: Analyze appropriate resolution for intermediate 2D representations (power maps, congestion maps)
  - Balance between information preservation and computational efficiency
  - Determine optimal grid size parameter "n" (likely 32×32 or 64×64)
- **Feature Extraction**: Define procedures for extracting relevant features from EDA tool outputs

#### 2.2. Benchmark Method Implementation

- **Bayesian Optimization**:
  - Implement GP-based surrogate model with appropriate kernels for mixed parameter types
  - Implement batch acquisition functions (q-EI, q-UCB)
  - Integrate with GPyTorch/BoTorch frameworks

- **Ant Colony Optimization**:
  - Implement multi-objective ACO variant
  - Define appropriate pheromone update rules for the PD parameter space
  - Develop parallelized evaluation mechanism

- **FIST**:
  - Implement Fast Iterative Slicing Technique
  - Adapt to handle mixed parameter types
  - Optimize for parallel evaluation

- **XGBoost/Gradient Boosting**:
  - Implement regression models for PPA prediction
  - Develop uncertainty quantification approaches
  - Create acquisition function based on model predictions

#### 2.3. Synthetic Test Environment

- **Surrogate Function Creation**:
  - Develop synthetic test functions that mimic real PD parameter-to-PPA relationships
  - Incorporate realistic parameter interactions and constraints
  - Generate mock intermediate representations (2D grids) with appropriate characteristics

- **Verification Framework**:
  - Create automated testing pipeline for algorithm verification
  - Implement metrics for comparing algorithm performance on synthetic functions
  - Design test cases of varying difficulty to thoroughly evaluate method robustness

- **Parallelism Simulation**:
  - Develop framework to simulate concurrent evaluations
  - Implement mechanisms to test batch selection strategies
  - Verify scaling behavior with different batch sizes

### 3. PABO Implementation

- **Transformer Architecture**:
  - Implement parameter encoder module
  - Develop 2D grid generation network
  - Create PPA prediction decoder
  - Integrate cross-modal fusion mechanism

- **Training Pipeline**:
  - Implement multi-task learning with appropriate loss functions
  - Develop curriculum learning strategy
  - Create efficient batch generation procedures

- **Acquisition Function**:
  - Implement batch acquisition mechanisms compatible with the transformer model
  - Develop exploration-exploitation balance strategies
  - Optimize for parallel evaluation

- **Deployment Integration**:
  - Create interfaces to EDA tools
  - Develop automated parameter suggestion system
  - Implement feedback mechanism for continuous improvement

This implementation plan will be executed in phases, with regular validation against both synthetic and real physical design benchmarks to ensure the developed methods meet practical requirements.

## Conclusion

Batch parallel parameter optimization provides an efficient framework for exploring Pareto frontiers in physical design optimization problems. By leveraging parallelism, these methods can significantly reduce the time required to find optimal parameter configurations. The choice between Bayesian, evolutionary, or ant colony approaches depends on problem characteristics, with Bayesian methods typically preferred for expensive evaluations with small batch sizes, while population-based methods excel with larger batch sizes. 