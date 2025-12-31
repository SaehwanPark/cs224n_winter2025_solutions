---
title: "Solution to Assignment 3"
author: "Sae-Hwan Park"
date: 2025-12-30
problem_set_link: https://web.stanford.edu/class/cs224n/assignments_w25/a3.pdf 
---

## Q2. NMT Analysis

### Q2-1. 1D Conv Layers for mandarin Chinese

How could adding a 1D Convolutional layer after the embedding layer and before passing the embeddings into the bidirectional encoder help our NMT system?


My answer:

Note that, in Mandarin Chinese, meaning is often constructed by combining individual characters into compound words (e.g., 电 + 脑 = 电脑/Computer).

A 1D Convolutional layer acts as a local feature extractor. By using a sliding window (kernel) over the character embeddings, the CNN can capture the local dependencies and compositional semantics of adjacent characters before the sequence is processed by the Bi-LSTM.

This allows the model to form better representations of multi-character words/phrases, effectively "grouping" characters into meaningful units, which reduces the burden on the RNN to learn these local patterns from scratch.

---

### Q2-2. Error Analysis

Here we present a series of errors we found in the outputs of our NMT model. For each example of a reference (i.e., 'gold') English translation, and NMT (i.e., 'model') English translation, please:

- Identify the error in the NMT translation.
- Provide possible reason(s) why the model may have made the error (either due to a specific linguistic construct or a specific model limitation).
- Describe one possible way we might alter the NMT system to fix the observed error.

i. (2 points) Source Sentence: 罪犯们其后被警方拘捕及被判处盗窃罪名成立。
- Reference Translation: the culprits were subsequently arrested and convicted.
- NMT Translation: the culprit was subsequently arrested and sentenced to theft.

ii. (2 points) Source Sentence: 几乎已经没有地方容纳这些人, 资源已经用尽。
- Reference Translation: there is almost no space to accommodate these people, and resources have run out.
- NMT Translation: the resources have been exhausted and resources have been exhausted.

iii. (2 points) Source Sentence: 当局已经宣布今天是国殇日。
- Reference Translation: authorities have announced a national mourning today.
- NMT Translation: the administration has announced today's day.

iv. (2 points) Source Sentence: 俗语有云:"唔做唔错"。
- Reference Translation: " act not, err not ", so a saying goes.
- NMT Translation: as the saying goes, " it's not wrong. "


My Answer:

i.

- Error: "culprit" (singular) vs "culprits" (plural); "sentenced to theft" is semantically incorrect.
- Reason: The model failed to track the plural number from the source ("们") and misinterpreted the relationship between "theft" and "convicted."
- Fix: Use a larger training corpus or subword tokenization (BPE) to better link morphology (plurality) to meaning.

ii.

- Error: The phrase "resources have been exhausted" is repeated twice.
- Reason: This is a classic RNN "looping" failure where the decoder state fails to update significantly, causing it to attend to the same source tokens repeatedly.
- Fix: Implement Coverage Modeling or a Length Penalty during decoding to discourage repeating the same attended regions.

iii.

- Error: "today's day" loses the core meaning of "national mourning" (国殇).
- Reason: The phrase "national mourning" might be an Out-of-Vocabulary (OOV) or rare term that the model mapped to a generic "day" token.
- Fix: Incorporate Pre-trained Embeddings (like BERT/RoBERTa) or use a Copy Mechanism to handle rare entities and specific nouns.

iv.

- Error: "it's not wrong" misses the idiomatic nuance of "if you do nothing, you make no mistakes."
- Reason: Idioms and proverbs (俗语) are non-compositional. The model tried to translate "唔做唔错" literally/locally rather than as a cultural unit.
- Fix: Use Back-translation to augment the training data with more idiomatic expressions or use a larger Transformer-based model.

---

### Q2-3. BLEU Score Analysis

Suppose we have a source sentence $s$, a set of $k$ reference translations $r_1, \ldots, r_k$, and a candidate translation $c$. To compute the BLEU score of $c$, we first compute the modified $n$-gram precision $p_n$ of $c$, for each of $n = 1, 2, 3, 4$:

$$p_n = \frac{\sum_{n\text{-gram} \in c} \min(\max_{i=1,\ldots,k} \text{Count}_{r_i}(n\text{-gram}), \text{Count}_c(n\text{-gram}))}{\sum_{n\text{-gram} \in c} \text{Count}_c(n\text{-gram})}$$

$$BP = \begin{cases}
1 & \text{if } \text{len}(c) \geq \text{len}(r) \\
\exp(1 - \text{len}(r) / \text{len}(c)) & \text{otherwise}
\end{cases}$$

$$\text{BLEU} = BP \times \exp\left(\sum_{n=1}^{4} \lambda_n \log p_n\right)$$

i. Consider this example:

- Source Sentence $s$: 需要有充足和可预测的资源。
- Reference Translation $r_1$: resources have to be sufficient and they have to be predictable
- Reference Translation $r_2$: adequate and predictable resources are required
- NMT Translation $c_1$: there is a need for adequate and predictable resources
- NMT Translation $c_2$: resources be sufficient and predictable to

Compute the BLEU scores for $c_1$ and $c_2$. Let $\lambda_i = 0.5$ for $i \in \{1, 2\}$ and $\lambda_i = 0$ for $i \in \{3, 4\}$ (ignore 3-grams and 4-grams)

ii. We lost Reference Translation $r_1$. Please recompute BLEU scores for $c_1$ and $c_2$, this time with respect to $r_2$ only. Which of the two NMT translations now receives the higher BLEU score?

iii. NMT systems are often evaluated with respect to only a single reference translation. Explain why this may be problematic.

iv. List two advantages and two disadvantages of BLEU, compared to human evaluation, as an evaluation metric for Machine Translation.


My Answer:

i. BLEU with $r_1$ and $r_2$

Candidate $c_1$: there is a need for adequate and predictable resources (Length: 9)

- $p_1$: (adequate, predictable, resources) match. Total matches: 5 (need, for, adequate, and, predictable, resources). Check: $c_1$ has 9 words. $r_2$ contains "adequate", "and", "predictable", "resources". $r_1$ contains "there" (in "they"). 
- Matched 1-grams: {there, is, a, need, for, adequate, and, predictable, resources}. In $r_1/r_2$, we find "adequate", "and", "predictable", "resources". Total matches = 4.
- $p_1 = 4/9 \approx 0.444$
- $p_2$: Pairs like "adequate and", "and predictable", "predictable resources" match $r_2$. Total matches = 3.
- $p_2 = 3/8 = 0.375$$BP$: $\text{len}(c_1)=9, \text{len}(r_2)=7$ (closest).
- $BP = 1$ (since $9 \geq 7$)
- BLEU $c_1$ $= 1 \times \exp(0.5 \ln 0.444 + 0.5 \ln 0.375) \approx \mathbf{0.408}

$Candidate $c_2$: resources be sufficient and predictable to (Length: 7)

- $p_1$: {resources, sufficient, and, predictable} match.
- $p_1 = 4/7 \approx 0.571$
- $p_2$: {sufficient and, and predictable} match.
- $p_2 = 2/6 = 0.333$$BP$: $\text{len}(c_2)=7, \text{len}(r_2)=7$.
- $BP = 1$.
- BLEU $c_2$ $= 1 \times \exp(0.5 \ln 0.571 + 0.5 \ln 0.333) \approx \mathbf{0.436}$

$c_2$ has a higher BLEU. I disagree that $c_2$ is better; $c_1$ is a grammatically correct English sentence ("there is a need for..."), whereas $c_2$ ends in a dangling preposition ("to") and uses incorrect "be" syntax.

ii. BLEU with $r_2$ only

Without $r_1$, the precision for $c_1$ remains the same if matches were primarily in $r_2$. However, if $c_2$ relied on "sufficient" from $r_1$, its score drops significantly because $r_2$ uses "adequate."

$c_1$ likely receives the higher score now because its vocabulary ("adequate") aligns better with $r_2$.

iii. Single Reference Problem

Using a single reference is problematic because translation is a one-to-many task. There are multiple "correct" ways to translate a sentence (different synonyms, active vs. passive voice).

BLEU behavior: With multiple references, BLEU takes the maximum count across all references for the numerator, allowing for stylistic diversity. With one reference, a perfectly valid translation might get a 0 score simply because it used a synonym (e.g., "adequate" vs "sufficient").

iv. BLEU vs. Human Eval

Pros
- Fast and inexpensive for iterative development.
- Consistent/Reproducible (unlike humans who vary in fatigue/bias).

Cons
- Insensitive to meaning; a sentence with "not" removed might have a high BLEU but opposite meaning.
- Does not account for grammatical correctness or fluid syntax (as seen in part i).


