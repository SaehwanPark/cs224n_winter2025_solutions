---
title: "Solution to Assignment 4"
author: "Sae-Hwan Park"
date: 2025-12-31
problem_set_link: https://web.stanford.edu/class/cs224n/assignments_w25/a4.pdf 
---

## Q1. Attention Exploration

### Q1-1. Copying in attention

One advantage of attention is that it's particularly easy to "copy" a value vector to the output $c$.

i. The distribution $\alpha$ is typically relatively "diffuse"; the probability mass is spread out between many different $\alpha_i$. However, this is not always the case. Describe (in one sentence) under what conditions the categorical distribution $\alpha$ puts almost all of its weight on some $\alpha_j$, where $j \in \{1, \ldots, n\}$ (i.e. $\alpha_j \gg \sum_{i \neq j} \alpha_i$). What must be true about the query $q$ and/or the keys $\{k_1, \ldots, k_n\}$?

ii.Under the conditions you gave in (i), describe the output $c$.


My answer:

The distribution $\alpha$ puts almost all its weight on a specific $\alpha_j$ when the dot product $k_j^\top q$ is significantly larger than all other dot products $k_i^\top q$ for $i \neq j$.

In this case, $c \approx v_j$. The output becomes a "copy" of the value vector associated with the winning key.

---

### Q1-2. An average of two

Consider the case where we instead want to incorporate information from two vectors $v_a$ and $v_b$, with corresponding key vectors $k_a$ and $k_b$. Assume that (1) all key vectors are orthogonal, so $k_i^\top k_j = 0$ for all $i \neq j$; and (2) all key vectors have norm 1. Find an expression for a query vector $q$ such that $c \approx \frac{1}{2}(v_a + v_b)$, and justify your answer.


My answer:

To get $c \approx \frac{1}{2}(v_a + v_b)$, we need the attention weights $\alpha_a$ and $\alpha_b$ to be approximately $0.5$, and all other $\alpha_i$ to be near $0$. Given the orthogonality and unit norm constraints, we can set:

$$q = \lambda(\mu_a + \mu_b)$$

For a large scalar $\lambda > 0$

The dot products will be $k_a^\top q = \lambda$, $k_b^\top q = \lambda$, and $k_i^\top q = 0$ for $i \notin \{a, b\}$.

The softmax results in $\alpha_a = \alpha_b = \frac{e^\lambda}{2e^\lambda + (n-2)e^0}$. As $\lambda \to \infty$, the constant terms in the denominator become negligible, $\alpha_a \approx \alpha_b \approx 0.5$, and $c \approx 0.5v_a + 0.5v_b$.

---

### Q1-3. Drawbacks of single-headed attention

Consider a set of key vectors $\{k_1, \ldots, k_n\}$ that are now randomly sampled, $k_i \sim \mathcal{N}(\mu_i, \Sigma_i)$, where the means $\mu_i \in \mathbb{R}^d$ are known to you, but the covariances $\Sigma_i$ are unknown (unless specified otherwise in the question). Further, assume that the means $\mu_i$ are all perpendicular; $\mu_i^\top \mu_j = 0$ if $i \neq j$, and unit norm, $\|\mu_i\| = 1$.

i. Assume that the covariance matrices are $\Sigma_i = \alpha I$, $\forall i \in \{1, 2, \ldots, n\}$, for vanishingly small $\alpha$. Design a query $q$ in terms of the $\mu_i$ such that as before, $c \approx \frac{1}{2}(v_a + v_b)$

ii. Let us consider a covariance for item $a$ as $\Sigma_a = \alpha I + \frac{1}{2}(\mu_a \mu_a^\top)$ for vanishingly small $\alpha$. This causes $k_a$ to point in roughly the same direction as $\mu_a$, but with large variances in magnitude. Further, let $\Sigma_i = \alpha I$ for all $i \neq a$.

When you sample $\{k_1, \ldots, k_n\}$ multiple times, and use the $q$ vector that you defined in part i., what do you expect the vector $c$ will look like qualitatively for different samples?


My answer:

i. Set $q = \lambda(\mu_a + \mu_b)$ for a large $\lambda$.

Since the variance $\alpha$ is vanishingly small, $k_i \approx \mu_i$.

The logic remains the same as earlier: the query projects equally onto the means of the keys we want to capture.

ii. In this scenario, $k_a$ has a large variance in magnitude. 

Because the dot product $k_a^\top q$ is sensitive to the length of $k_a$, the value of $\alpha_a$ will fluctuate wildly across samples.

When $\|k_a\|$ is large, $c \approx v_a$; when $\|k_a\|$ is small, $c \approx v_b$.

Qualitatively, $c$ will have high variance and will rarely actually be the average; it will "swing" between the two values depending on which key happens to have a larger magnitude in that sample.

---

### Q1-4. Benefits of multi-headed attention

i. Assume that the covariance matrices are $\Sigma_i = \alpha I$, for vanishingly small $\alpha$. Design $q_1$ and $q_2$ in terms of $\mu_i$ such that $c$ is approximately equal to $\frac{1}{2}(v_a + v_b)$.

ii. Assume that the covariance matrices are $\Sigma_a = \alpha I + \frac{1}{2}(\mu_a \mu_a^\top)$ for vanishingly small $\alpha$, and $\Sigma_i = \alpha I$ for all $i \neq a$. Take the query vectors $q_1$ and $q_2$ that you designed in part i. What, qualitatively, do you expect the output $c$ to look like across different samples of the key vectors?


My answer:

i. We can assign each head to target a specific vector: $q_1 = \lambda \mu_a$ and $q_2 = \lambda \mu_b$.

This way, $c_1 \approx v_a$ and $c_2 \approx v_b$.

ii. Even with the high variance of $\|k_a\|$, $c_1$ will almost always point to $v_a$ because $q_1$ only has a significant dot product with $k_a$ (since $q_1 \perp \mu_i$ for $i \neq a$). 

Similarly, $c_2$ will consistently be $v_b$. When we take the average $\frac{1}{2}(c_1 + c_2)$, the result is much more stable. 

The variance of the final $c$ is significantly reduced compared to the single-headed case.

---

### Q1-5. Summary of Exploration

Based on the above, briefly summarize how multi-headed attention overcomes the drawbacks of single-headed attention that you identified.


My answer:

Multi-headed attention allows the model to be robust to variations in key scales. By dedicating different heads to different features or positions, the model can "vote" or average the results at the end, rather than forcing a single attention head to maintain a precarious numerical balance between multiple keys.

---

## Q2. Position Embedding Exploration

Given an input tensor $X \in \mathbb{R}^{T \times d}$, where $T$ is the sequence length and $d$ is the hidden dimension, the self-attention layer computes the following:

$$Q = XW_Q, \quad K = XW_K, \quad V = XW_V$$

$$H = \text{softmax}\left(\frac{QK^\top}{\sqrt{d}}\right)V$$

where $W_Q, W_K, W_V \in \mathbb{R}^{d \times d}$ are weight matrices, and $H \in \mathbb{R}^{T \times d}$ is the output.

Next, the feed-forward layer applies the following transformation:

$$Z = \text{ReLU}(HW_1 + \mathbf{1} \cdot b_1)W_2 + \mathbf{1} \cdot b_2$$

where $W_1, W_2 \in \mathbb{R}^{d \times d}$ and $b_1, b_2 \in \mathbb{R}^{1 \times d}$ are weights and biases; $\mathbf{1} \in \mathbb{R}^{T \times 1}$ is a vector of ones; and $Z \in \mathbb{R}^{T \times d}$ is the final output.

### Q2-1. Permuting the input

i. Suppose we permute the input sequence $X$ such that the tokens are shuffled randomly. $P \in \mathbb{R}^{T \times T}$, i.e. $X_{\text{perm}} = PX$. 

Show that the output $Z_{\text{perm}}$ for the permuted input $X_{\text{perm}}$ will be $Z_{\text{perm}} = PZ$.

You are given that for any permutation matrix $P$ and any matrix $A$, the following hold:

- $\text{softmax}(PAP^\top) = P \text{softmax}(A) P^\top$
- $\text{ReLU}(PA) = P \text{ReLU}(A)$

ii. Think about the implications of the result you derived in part i. Explain why this property of the Transformer model could be problematic when processing text.


My answer:

i. For attention, let $Q_p = PXW_Q = PQ$.

Similarly $K_p = PK$ and $V_p = PV$.

The attention scores $A_p = \text{softmax}(\frac{(PQ)(PK)^\top}{\sqrt{d}}) = \text{softmax}(\frac{PQK^\top P^\top}{\sqrt{d}})$.

Using the identity $\text{softmax}(PAP^\top) = P \text{softmax}(A) P^\top$, we get $H_p = (P \text{softmax}(\frac{QK^\top}{\sqrt{d}}) P^\top)(PV)$.

Since $P^\top P = I$ (permutation matrices are orthogonal), $H_p = PH$

For feed-Forward, $Z_p = \text{ReLU}(PHW_1 + \mathbf{1}b_1)W_2 + \mathbf{1}b_2$.

Since $P\mathbf{1} = \mathbf{1}$ (shuffling rows of a vector of ones doesn't change it), we can factor out $P$: $Z_p = P(\text{ReLU}(HW_1 + \mathbf{1}b_1)W_2 + \mathbf{1}b_2) = PZ$.

ii. This shows that Transformers are permutation equivariant. Without positional information, the model treats the input as a "bag of words." In text, word order is critical for meaning (e.g., "The dog bit the man" vs. "The man bit the dog"). A model that is equivariant cannot distinguish between these two sequences.

---

### Q2-2. Fixed positional embeddings

i. Do you think the position embeddings will help the issue you identified above?

ii. Can the position embeddings for two different tokens in the input sequence be the same?


My answer:

i. Yes, this solves the issue. 

By adding $\Phi$, the input $X_{\text{pos}}$ for the same word at different positions becomes different ($x_t + \phi_t \neq x_k + \phi_k$).

The permutation $P$ would now change the values of the vectors associated with specific words, breaking the symmetry and allowing the model to "know" where a word is located.

ii. No, the embeddings cannot be the same for different $t$.

The function uses a range of frequencies (determined by $i$).

For any two positions $t_1, t_2$, there will be at least one dimension $i$ where the combination of sine and cosine at that specific frequency uniquely identifies the position, provided the sequence length doesn't exceed the period of the lowest frequency wave.
