---
title: "Solution to Assignment 2"
author: "Sae-Hwan Park"
date: 2025-12-24
problem_set_link: https://web.stanford.edu/class/cs224n/assignments_w25/a2.pdf 
---

## Q1. Word2Vec

**Q1-a. Prove the naive-softmax loss is the same as the cross-entropy loss.**

My answer:

Consider cross entropy:
$$
\begin{split}
J 
&=-\sum_{w\in V}{y_w \log{\widehat{y_w}}} \\
&= -\log{\widehat{y_o}}\\
\end{split}
$$

Note $y_w$ is 1 when $w=o$. Otherwise 0.

Replacing $\widehat{y_o}$ with probability prediction term, we get naive softmax loss.

$$
J = -\log{
  \frac{\exp{u_o^\text{T}v_c}}{\sum_{w\in V}\exp{u_w^\text{T}v_c}}
}
$$

---

**Q1-b.(i) Compute the partial derivative of naive-softmax loss $J_\text{naive-softmax}(v_c, o, U)$ with respect to $v_c$; (ii) When is the gradient equal to zero?; (iii) The gradient is the difference between the two terms. Provide an interpretation**

My answer:

Rewrite $J$

$$
J = -u_o^\text{T}v_c + \log\left({\sum_{w\in V}\exp{u_w^\text{T}v_c}}\right)
$$

Then take a derivative

$$
\begin{split}
\frac{\partial J}{\partial v_c} &= -u_o +
\sum_{w\in V}\frac{\exp{u_w^\text{T}v_c}}{\sum_{x\in V}\exp{u_x^\text{T}v_c}}u_w\\
\end{split}
$$

Note that this is essentially:

$$
\frac{\partial J}{\partial v_c} = -u_o + \sum_{w\in V} \Pr[w|c]\text{ }u_w = U(\hat{y} -y)
$$


This forms equalibrium (zero gradient) when $\Pr[o|c]=1$ and $\Pr[w|x]=0$ wherever $w\neq o$ (i.e., predicted probability distribution is identical to the ground-truth)


Thinking this as a update process in gradient descent (assume $\alpha$ is a learning rate),

$$
v_c^\text{new} = v_c^\text{old} + \alpha u_o - \alpha\Pr[w|c]u_w 
$$

This shows 2 processes: attraction and repulsion:

- Attraction: the update pulls $v_c$ toward $u_o$ in the vector space. By moving $v_c$ toward $u_o$, the dot product $u_o^\text{T}v_c$ increases. Since the probability $\Pr[o|c]$ is proportional to the exponent of this dot product, this increases the likelihood that the model will predict the correct word $o$ when it sees the center word $c$ in the future.
- Repulsion: the update pushes $v_c$ away from all words $u_w$ but the strength of the push is proportional to how much the model currently thinks $w$ belongs in that context $\Pr[w|c]$. This prevents the model from cheating by just making $v_c$ large in all directions. It penalizes the probability of all words. However, because it is a weighted average, it pushes hardest against the words that the model is currently incorrectly confident about.

We may also think this dynamics as iterations betweeen reinforcement and normalization.

- Reinforcement: the specific word $o$ that actually appeared gets a strong, direct pull
- Normalization: the rest of the vocabulary (including $o$ itself) exerts a sollective push away to ensure the total probability distribution sums to 1.

---

**Q1-c. In many downstream applications using word embeddings, L2 normalized vectors are used instead of their raw forms. When would L2 normalization take away useful information for the downstream task? When would it not?**

My answer:

Bottom line, when we normalize a vector, we effectively discard "magnitude" and keep only its "direction".

Let's assume our hypothetical task is to classify sentiment based on the sum of embeddings. The impact from L2 normalization would depend on whether the discarded magnitude contained signal or noise.

In many embedding spaces, words with stronger sentiment or more specific meanings tend to have larger magnitude.

For example, let's consider the words "decent" and "extraordinary". Both are positive. But the latter is much more intense. This may results in larger norm for "extraordinary" than "decent". The sum natually gives "extraordinary" more weight in the final classification.

Normalizing them forces both words to have a magnitude of 1, treating "my grade is decent" and "my grade is extraordinary" as having equal sentiment weight.

Also In some models, word frequency affects the norm. If rare, highly descriptive words (which are often the most useful for sentiment) have larger norms than common, generic words, the raw sum will be dominated by those descriptive words. Normalizing would "demote" these informative rare words and "promote" common filler words.

On the other hand, normalization may be beneficial (or harmless) when vector magnitude can be noise.

In models like Skip-gram, high-frequency words (like "the," "and," "is") can sometimes develop very large magnitudes simply because they were updated so many times during training. If "the" has a massive norm, the sum of "The movie was bad" might be dominated by the vector for "the," potentially leading to a misclassification if "the" happens to have a slight directional bias. Normalizing ensures every word gets an "equal vote" in the sum, preventing high-frequency "stop words" from drowning out the actual sentiment-carrying words.

Also, if the sentiment of a phrase is strictly about the average direction of the words rather than their cumulative intensity, normalization is ideal.

---

**Q1-d/e. Compute the partial derivatives of $J_\text{naive-softmax}(v_c, o, U)$ with respect to $U$**

My answer:

Consider $U = [u_1 .. u_{|V|}]$ (collection of column vectors)

Let's also consider two cases: $w=o$ vs $w\neq o$

Case 1 ($w=o$): Actual target word

$$
\frac{\partial J}{\partial u_o} = (\Pr[o|c] -1) v_c
$$

Case 2 ($w\neq o$): Any other word

$$
\frac{\partial J}{\partial u_w} = \Pr[w|c]\text{ } v_c
$$

Combining both cases and that $\Pr[w|c] = \widehat{y_w}$:

$$
\frac{\partial J}{\partial u_w} = (\widehat{y_w} - y_w) v_c
$$

This means:

$$
\frac{\partial J}{\partial U} = v_c (\hat y - y)^\text{T}
$$

This and what we computed in `Q1-b` illustrate how W2V learns holistically, forming a coordinated dance between the center word and the entire library of outside words.

1. The perspective of the center word ($v_c$): The center word vector moves to become more like the context it is currently in and less like the average context the model predicts. This pushes $v_c$ into a specific neighborhood of the vector space associated with its meaning.

2. The perspective of the outside words ($U$): (a) For the true context words $\widehat{y_o}-1$ is negative. This pulls the target word $u_o$ toward the center word.; (b) For all other words, since $\widehat{y_w} - 0$ is positive, the update pushes every other words in the dictionary away from the center word.

The whole learning process acts like a system:

1. Mutual attraction: the center word and the target outside word pulls each other together, creating a cluster of related terms.
2. Global repulsion: Every time a pair of words is pulled together, the center word simultaneously pushes every other word in the vocabulary away, ensuring that the vector space doesn't collapse into a single point and that words remain distinct.
3. Semantic clustering: words that appear in similar environments (e.g., Paris and London) may never appear in the same window together. But, because they are both frequently pulled toward the same context words, they eveually end up near each other.

This reminds me of E-M algorithm. That's likely because W2V model is also essentially performing latent factor analysis. Because each word acts as both a "center" and a "context," updating $v$ depends on the current values of $U$, and updating $U$ depends on the current values of $v$. This "chicken-and-egg" dependency is the hallmark of problems solved by EM.

---

## Q2. Neural Network

### Q2-1. Adam Optimizer

**Q2-1-i. Adam uses a trick called momentum by keeping track of $m$, a rolling average of gradients. How does using $m$ stop the updates from varying as much? Why may this low variance be helpful to learning overall?**

$$m_{t+1} \leftarrow \beta_1 m_t + (1 - \beta_1) \nabla_{\theta_t} J_{\text{minibatch}}(\theta_t)$$
$$\theta_{t+1} \leftarrow \theta_t - \alpha m_{t+1}$$

My answer: 

Using the rolling average $m$ acts as a low-pass filter on the gradient updates. By incorporating a fraction of the previous gradients, the update vector dampens high-frequency noise and oscillations that occur when minibatches are non-representative or the loss landscape is jagged.

This reduction in variance is helpful because it allows the optimizer to maintain a consistent direction toward the local minimum. Instead of *zig-zagging* across the walls of a narrow ravine, momentum accumulates speed in the direction of the consistent downward slope, leading to faster and more stable convergence.

---

**Q2-1-ii. Adam also extends the idea of momentum with the trick of adaptive learning rates by keeping track of $v$, a rolling average of the magnitudes of the gradients. Explain how this will affect.**

$$m_{t+1} \leftarrow \beta_1 m_t + (1 - \beta_1) \nabla_{\theta_t} J_{\text{minibatch}}(\theta_t)$$
$$v_{t+1} \leftarrow \beta_2 v_t + (1 - \beta_2)(\nabla_{\theta_t} J_{\text{minibatch}}(\theta_t) \odot \nabla_{\theta_t} J_{\text{minibatch}}(\theta_t))$$
$$\theta_{t+1} \leftarrow \theta_t - \alpha m_{t+1} / \sqrt{v_{t+1}}$$

My answer:

In adam, dividing by the square root of $v$ (uncentered variance of the gradients) scales the updates for each individual parameter.

Parameters that have infrequent or small gradients (i.e., $v$ is small) will receive relatively larger updates. Conversely, parameters with large volatile gradients (where $v$ is large) will have their effective larning erate significantly reduced.

This is especially helpful for sparse data or complex architectures where some features appear rarely. By magnifying updates for parameters that otherwise move slowly, Adam ensures that all weights--even those with weak signals--can contribute to learning. It effectively levels the playing field, preventing "loud" gradients from dominating the training process while "quiet" ones are ignored.

In practice, Adam also includes a bias correction step for $m$ and $v$ to account for their initialization at zero, which is especially important during the first few timesteps of training.

---

### Q2-2. Dropout

$$h_{\text{drop}} = \gamma d \odot h$$

where $d \in \{0, 1\}^{D_h}$ ($D_h$ is the size of $h$) is a mask vector where each entry is 0 with probability $p_{\text{drop}}$ and 1 with probability $(1 - p_{\text{drop}})$. $\gamma$ is chosen such that the expected value of $h_{\text{drop}}$ is $h$:

$$\mathbb{E}_{p_{\text{drop}}}[h_{\text{drop}}]_i = h_i$$

for all $i \in \{1, \ldots, D_h\}$.

**Q2-2-i. What must $\gamma$ equal in terms of $p_\text{drop}$?**

My answer:

Use the expected value of $h_\text{drop}$ and $h_i , \gamma$ being constant,

$$
\begin{align*}
\mathbb{E}[h_{\text{drop}, i}]
&= \mathbb{E}[\gamma d_i h_i] \\
&= \gamma h_i\mathbb{E}[d_i] \\
&= \gamma h_i (1-p_\text{drop}) \\
h_i & = \gamma h_i (1-p_\text{drop})
\end{align*}
$$

Thus, we can get $$\gamma = \frac{1}{1-p_\text{drop}}$$

Note that the scaling ensures that the total magnitude (of the activations) remains consistent between training (i.e., units are missing) and testing (i.e., all units are present).

---

**Q2-2-ii. Why should dropout be applied during training? Why should dropout NOT be applied during evaluation?**

My answer:

Dropout forces the network to learn redundant representations and prevents *co-adaptation*, where neurons rely on the specific presence of other neurons to correct their errors. By randomly removing units, the model is forced to learn more robust features that are useful across an exponential number of different internal architectures, effectively acting as a form of ensemble learning.

However, during evaluation, we want the model to use its full capacity and the complete consensus of the learned features to make the most accurate prediction possible. Because we scaled the weights by $\gamma$ during training, the expected output of the full network at the test time is already calibrated to match the training distribution, providing a deterministic and lower-variance result.

---

## Q3. Neural Dependency Parsing

### Q3-1. Transition sequence trace

Go through the sequence of transitions needed for parsing the sentence "I presented my findings at the NLP conference".

My answer:

| Stack | Buffer | New dependency | Transition |
| --- | --- | --- | --- |
| `[ROOT]` | `[I, presented, my, findings, at, the, NLP, conference]` | - | Initial |
| `[ROOT, I]` | `[presented, my, findings, at, the, NLP, conference]` | - | SHIFT |
| `[ROOT, I, presented]` | `[my, findings, at, the, NLP, conference]` | - | SHIFT |
| `[ROOT, presented]` | `[my, findings, at, the, NLP, conference]` | `presented → I` | LEFT-ARC |
| `[ROOT, presented, my]` | `[findings, at, the, NLP, conference]` | - | SHIFT |
| `[ROOT, presented, my, findings]` | `[at, the, NLP, conference]` | - | SHIFT |
| `[ROOT, presented, findings]` | `[at, the, NLP, conference]` | `findings → my` | LEFT-ARC |
| `[ROOT, presented]` | `[at, the, NLP, conference]` | `presented → findings` | RIGHT-ARC |
| `[ROOT, presented, at]` | `[the, NLP, conference]` | - | SHIFT |
| `[ROOT, presented, at, the]` | `[NLP, conference]` | - | SHIFT |
| `[ROOT, presented, at, the, NLP]` | `[conference]` | - | SHIFT |
| `[ROOT, presented, at, the, NLP, conference]` | `[]` | - | SHIFT |
| `[ROOT, presented, at, the, conference]` | `[]` | `conference → NLP` | LEFT-ARC |
| `[ROOT, presented, at, conference]` | `[]` | `conference → the` | LEFT-ARC |
| `[ROOT, presented, conference]` | `[]` | `conference → at` | LEFT-ARC |
| `[ROOT, presented]` | `[]` | `presented → conference` | RIGHT-ARC |
| `[ROOT]` | `[]` | `ROOT → presented` | RIGHT-ARC |

---

### Q3-2. Parsing complexity

A sentence containing $n$ words will be parsed in how many steps (in terms of $n$)? Briefly explain in 1–2 sentences why.

My answer:

A sentence with  words will be parsed in exactly $2n$ steps. Each of the words must be shifted onto the stack exactly once (SHIFT operations). To empty the stack back down to just the ROOT, each of those  words must also be involved in exactly one ARC operation (either LEFT or RIGHT) to be removed from the stack.

---

### Q3-3. NN gradients

i. Compute the derivative of $h = \text{ReLU}(xW + b_1)$ with respect to $x$.

ii. Compute the partial derivative of $J(\theta)$ with respect to the $i$-th entry of $l$, which is denoted as $l_i$.

My answer:

$$h_i = \max(0, \sum_k x_k W_{ki} + b_{1,i})$$

So,
$$\frac{\partial h_i}{\partial x_j} = \begin{cases} W_{ji} & \text{if } \sum_k x_k W_{ki} + b_{1,i} > 0 \\ 0 & \text{otherwise} \end{cases}$$


And the gradient with respect to the $i$-th logit $l_i$ simplifies beautifully to:

$$\frac{\partial J}{\partial l_i} = \hat{y}_i - y_i$$

This means if $i$ is the correct class, the gradient is $\hat{y}_c - 1$; otherwise, it is simply the predicted probability $\hat{y}_i$.

---

### Q3-4. Dependency parsing error analysis

We'd like to look at example dependency parses and understand where parsers might be wrong. Here are four types of parsing error:

- **Prepositional Phrase Attachment Error:** A prepositional phrase is attached to the wrong head word.
- **Verb Phrase Attachment Error:** A verb phrase is attached to the wrong head word.
- **Modifier Attachment Error:** A modifier is attached to the wrong head word.
- **Coordination Attachment Error:** The second conjunct is attached to the wrong head word.

For each of the following sentences with dependency parses obtained from a parser, state the type of error, the incorrect dependency, and the correct dependency.

i. The university blocked the acquisition , citing concerns about the risks involved .

ii. Brian has been one of the most crucial elements to the success of Mozilla software .

iii. Investment Canada declined to comment on the reasons for the goverment decision .

iv. People benefit from a separate move that affects three US car plants and one in Quebec

My answer:

i.

- Error type: PP attachment error
- Incorrect: `involved -> about`
- Correct: `concerns -> about`

ii.

- Error type: Modifier attachment error
- Incorrect: `elements -> success`
- Correct: `crucial -> success`

iii.

- Error type: Coordination attachment error
- Incorrect: `Canada -> Investment`
- Correct: `declined -> Investment Canada` (note: as a compound object)

iv.

- Error type: coordination attachment error
- Incorrect: `plants -> one`
- Correct: `move -> one`

---

### Q3-5. Benefit of POS Tags

Recall in part Q3-3, the parser uses features which includes words and their part-of-speech (POS) tags. Explain the benefit of using part-of-speech tags as features in the parser.

My answer:

POS tags act as a form of generalization. While a word embedding captures the specific meaning of "apple," its POS tag (`NN`) tells the parser it functions as a noun. This helps the model learn universal syntactic rules (e.g., "Adjectives usually modify the following Noun") that apply even if the specific word is rare or unseen in the training data.
