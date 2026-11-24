# AISTATS 2027 review

## Meta Review
The authors propose a regression adjustment method for estimating treatment effects in randomized experiments with network interference: a prediction model (for example a graph neural network) is fitted first, and its output is then calibrated to reduce the variance of the effect estimator rather than the prediction error.

The initial reviews are positive, with all reviewers recommending acceptance. The reviewers find the paper well organized and the distinction between prediction accuracy and estimation efficiency well motivated.

Missing related work: Lu, Wang and Zhang (2024), "Adjusting Auxiliary Variables under Approximate Neighborhood Interference", studies design-based regression adjustment under network interference with the same objective; a detailed comparison is needed (Reviewer XK3s). Reviewer kJ1B also asks how the method relates to existing doubly robust and regression-based estimators under interference and what it adds beyond them.

Can you clarify how models and random seeds were selected in the experiments? Appendix H.3 states that one variant "requires careful initialization to achieve competitive performance" and reports results for a specific seed, and Reviewer XK3s asks for evidence of stability across seeds. If hyperparameters or seeds are chosen on the realized data to minimize confidence interval width, and interval width is then the metric used to show efficiency gains, the evaluation is circular. The authors should state exactly what was tuned on what, and report results over seeds.

The assumptions require exposure probabilities bounded below at a polylogarithmic rate, which excludes networks with high-degree nodes; under Bernoulli assignment the probability that no neighbour is treated decays exponentially in the degree (Reviewer XK3s).

### Author response (draft)

> Placeholders: `[TODO-EXP]` = experiment to run, `[TODO-THEORY]` = result to write up, `[CONFIRM]` = fact to verify before posting. Limit: 5,000 characters per response.

We thank the AC for the careful summary. We respond to the three points raised.

**1. Lu, Wang and Zhang (2024) and doubly robust / regression-based estimators.** We agree this work should be discussed and will add it to Section 1.3 and Appendix A. Both papers study design-based adjustment under network interference. The main differences, as we understand them, are: (i) GAUGER works with a general multi-level exposure mapping and an AIPW form with exposure-specific features, and (ii) the coefficient is chosen by graph-weighted estimating equations that target the design variance (the $\Lambda$-weighted objective), rather than a least-squares fit. Theorem 4.2 then gives a no-harm guarantee within the feature span: asymptotically, GAUGER is no worse than any fixed linear adjustment using the same features, which includes least-squares-type coefficients. GAUGER is also designed to sit on top of learned (e.g., GNN) features. [CONFIRM: precise estimator, assumptions and variance estimator in Lu, Wang and Zhang (2024), so the comparison is stated accurately.] [TODO-EXP: add their estimator as a baseline in Tables 1, 2 and 6 using the same features.] The DR estimators of Reviewer kJ1B's [1, 2] target observational settings where propensities must be modeled; in our randomized setting propensities are known, so the open question is efficiency, which is what GER addresses. Details are in our response to Reviewer kJ1B.

**2. Model and seed selection.** We appreciate this question and want to be fully transparent. The current protocol is:
- Synthetic studies: GAT was tuned for prediction $R^2$ on a validation split; PNA was tuned on [CONFIRM: separate randomization draws from the same fixed synthetic population, using RMSE over B=20 draws] that are disjoint from the evaluation draws.
- Kenya data: for the wage-bill outcome, the PNA seed was chosen [CONFIRM: by comparing realized CI widths across seeds].

We agree that selecting a seed or configuration on the realized data by CI width, then reporting CI width as the efficiency metric, is circular, and that the resulting interval is not guaranteed to have nominal coverage. We will fix this in three ways:
- (a) Fix all hyperparameters in advance, by a rule that does not use the realized outcomes of the analysis sample (a default configuration, or tuning on an independent replicate population for the simulations).
- (b) Remove seed selection: average the predictions over S seeds, which gives a fixed feature map, and report results for every individual seed. [TODO-EXP: Kenya revenue and wage bill, S = 50 seeds: median and IQR of $\hat\tau$ and CI width, and the share of seeds with a narrower CI than Hájek; same for the seed-averaged model.]
- (c) For the simulations, re-tune on a replicate population generated with different DGP seeds and re-evaluate. [TODO-EXP: Tables 1 and 6 under the re-tuned protocol.]

Appendix G and H.3 will state exactly what was tuned on what.

**3. Positivity with high-degree nodes.** This is a fair point, and it is a limitation of the estimand rather than of GAUGER specifically: when $\pi_i^{(d)}$ is exponentially small, the HT, Hájek and AIPW estimators of that contrast all have exploding variance. In practice there are three remedies:
- choosing exposure mappings based on the fraction of treated neighbors, whose probabilities do not vanish with degree (the mappings we use in Sections 6 and H);
- restricting the estimand to units with adequate positivity (Aronow and Samii, 2017);
- using designs such as graph cluster randomization.

We will add a remark on this. [TODO-EXP: a heavy-tailed-degree simulation reporting coverage and variance reduction as the maximum degree grows.]

## Reviewer kJ1B
Summary:
This paper studies causal effect estimation in design-based inference under interference. This paper first identifies a mismatch between accuracy and efficiency. To address this mismatch, this paper proposes a two-step method. It first leverages classic prediction method, and then apply a calibration scheme called Graph-weighted Exposure-level Residualization (GER) to reduce variance. The paper also establishes corresponding theoretical guarantees for the method. The experimental results show the effectiveness of the proposed method.

Strengths:
The paper considers an important issues, causal effect estimation under network interference.
The proposed framework is flexible and can be combined with different prediction models.
This paper is well written and easy to follow.
Weaknesses:
The exposure mapping is assumed to be known in the main theoretical results. Is it possible to extend to unknown interference mechanism? Additionally, is the proposed method robust to the exposure misspecification?

Could the authors further clarify the relationship between the proposed variance calibration and existing doubly robust or efficient estimation [1, 2, 3]? What additional efficiency gains does GER provide beyond these existing method?

It could be better to include a direct ablation study comparing uncalibrated AIPW, standard regression adjustment, and GER under the same backbone, while clearly explaining the differences among the different variants. Furthermore, directly illustrating the mismatch between prediction MSE and the variance of the causal estimator would help to support the core motivation of the paper.

It would be better to include datasets with denser networks, e.g., Flickr and BC, which are commonly used in network effect estimation.

[1] Doubly robust estimation in observational studies with partial interference. Stat

[2] Doubly robust causal effect estimation under networked interference via targeted learning. ICML

[3] Causal inference in network experiments: Regression-based analysis and design-based properties. JoE

Questions:
please see weakness

Clarity - Presentation: 3: Good (Generally clear; minor issues with organization, notation, or exposition that do not hinder understanding)
Soundness - Correctness: 3: Minor gaps (Generally sound; minor gaps or unstated benign assumptions)
Contribution - Significance: 3: Good (Solid contribution with clear value to at least one sub-area; some novelty, with limited scope or impact)
Rating: 3: Borderline Accept (Technically solid; merits outweigh minor limitations)
Confidence: 3: Fairly confident (Understood the core contribution; math/details not checked thoroughly)

### Author response (draft)

Thank you for the positive assessment and the concrete suggestions.

**Unknown or misspecified exposure mapping.** The main theory assumes a known mapping, as is standard in design-based work (Aronow and Samii, 2017). Two extensions are already in the appendix:
- Appendix D.6.2 follows Leung (2022): under misspecification, the estimator remains well defined and targets the contrast of expected outcomes $\mu(d)$ at each exposure value. The design-based unbiasedness of AIPW and the calibration argument do not rely on the mapping being correct, since the randomization is known.
- Appendix D.6.3 sketches a GNN-based procedure for learning the mapping.


**Relation to DR / efficient estimation [1–3].** [1] and [2] are for observational data: data are draw from a super population, and DR there protects against misspecification of the outcome model or the propensity model. In our randomization setup, potential outcomes are treated as fixed quantities following Rubin (2005). Also, in randomization, propensities are known exactly, so every AIPW estimator is unbiased for any outcome model (Theorem 3.1), and the open question is efficiency. Here GER adds something the DR estimators do not: it chooses the adjustment to minimize the design variance, which, under interference, is the $\Lambda$-weighted quadratic form, not the prediction MSE that [1, 2] effectively target. [3] (Gao and Ding) is design-based and uses (weighted) least-squares adjustment. Theorem 4.2 implies that, for the same features, GER is asymptotically no worse than any fixed coefficient, including a least-squares one such as that in [3]. The reason is that a least-squares fit treats units as uncorrelated, whereas under interference the design variance contains cross-unit covariance terms (the off-diagonal entries of $\Lambda$); GER is built to target exactly these terms. Empirically, our "linear-raw" baseline is a standard least-squares adjustment (per-exposure OLS plugged into AIPW), and GER on the same covariates ("linear-linear") reduces its SD from 0.202 to 0.170 (Table 1) and from 0.459 to 0.238 (Table 6).

**Ablation under the same backbone.** We agree the row labels were not self-explanatory. Tables 1 and 6 already contain the requested ablation, and we will add a legend. For the GNN backbones (GAT, PNA):
- "raw" is the uncalibrated AIPW that plugs in the MSE-trained predictions;
- "pred" is GER applied to those same predictions (Paradigm 1);
- "linear" is GER applied to the learned representations (Paradigm 2).

For the linear backbone, "raw" is the standard regression adjustment (per-exposure OLS plugged into AIPW), and "linear" and "pred" are GER on the same covariates (with an OLS-type and a LASSO-penalized solve, respectively). Uncalibrated AIPW, standard regression adjustment and GER are therefore compared on identical inputs.

**Mismatch between prediction MSE and variance.** The raw-versus-pred comparison isolates this mismatch directly. Both use the same predictor; "raw" keeps the MSE-trained output as is, while "pred" only re-weights it through GER. Any drop in SD from raw to pred is therefore variance that the MSE criterion leaves on the table. The drop is consistent:
- Table 1: GAT SD 0.168 → 0.140; linear 0.202 → 0.170; PNA 0.127 → 0.124.
- Table 6 (denser network): GAT 0.382 → 0.221; PNA 0.241 → 0.159.

The gap is small when the predictor is already close to variance-optimal (PNA in Table 1) and large when it is not, which is exactly the behaviour Theorem 3.1 predicts. We will make this reading explicit in Section 6.

**Denser networks (Flickr, BC).** Thank you for the suggestion. These graphs have heavy-tailed degrees, which also helps examine the positivity issue raised by other reviewers. [TODO-EXP: semi-synthetic Flickr/BC studies with fraction-based exposure mappings, comparing HT, Hájek, uncalibrated AIPW, and GER on GAT/PNA.]

## Reviewer AnSZ
Summary:
The paper proposes GAUGER for design-based causal inference under network interference. It combines flexible outcome prediction with graph-weighted exposure residualization, calibrating adjustments toward variance reduction rather than predictive accuracy alone. The authors provide asymptotic guarantees and evaluate GNN-based implementations using synthetic data, real-network simulations, and a randomized field experiment.

Strengths:
The paper is well organized and easy to follow.

The distinction between prediction accuracy and estimation efficiency is clearly motivated, and the proposed calibration directly targets the relevant variance structure.

3.. The framework accommodates different predictive models and combines theoretical analysis with empirical evaluation of estimation accuracy and uncertainty quantification.

Weaknesses:
The connection between the fixed-feature theoretical results and representations learned from the same outcomes needs clearer justification.

Finite-sample stability requires more attention: the GER weighting matrix can be indefinite, and singular estimating systems may not admit an exact solution. Practical safeguards and their impact on inference are insufficiently characterized.

Questions:
1, How are singular or ill-conditioned GER systems handled in practice, and how robust are variance reduction and coverage when exposure probabilities are small or networks contain high-degree nodes?

Clarity - Presentation: 4: Excellent (Very clear and well organized; contributions, notation, and assumptions are precisely stated and well motivated)
Soundness - Correctness: 3: Minor gaps (Generally sound; minor gaps or unstated benign assumptions)
Contribution - Significance: 3: Good (Solid contribution with clear value to at least one sub-area; some novelty, with limited scope or impact)
Rating: 4: Accept (Technically solid, clear contribution to AI & Stats; convincing evaluation)
Confidence: 3: Fairly confident (Understood the core contribution; math/details not checked thoroughly)

### Author response (draft)

We thank the reviewer for the supportive evaluation and the precise questions.

**Learned features vs. fixed-feature theory.** Our theorems condition on features that do not depend on the realized assignment. We take two routes to connect this to the GNN pipeline.
- (i) Cross-fitting (Appendix D.5). The buffer construction ensures that the features used for fold-1 units are trained only on units whose exposures are independent of the fold-1 exposures. Conditional on the other fold, the features are therefore fixed, and Theorems 3.1, 4.1 and 4.2 apply fold by fold. [TODO-THEORY: state this as a corollary, including how fold-wise estimates are combined and how the buffer affects $\Lambda$.] [TODO-EXP: report the cross-fitted GAUGER results alongside Tables 1 and 2.]
- (ii) Without cross-fitting, the empirical coverage of the calibrated estimators in Tables 1 and 6 is 0.92–0.98, which suggests that the dependence induced by learning is mild in these settings. We will state this distinction explicitly in Section 5.

**Indefinite $\Delta$ and singular systems.** We agree this deserved more space.
- First, we never minimize $Q$. We solve the linear system $\hat A\beta=\hat b$, so indefiniteness of $\Delta$ matters only through the conditioning of $\hat A$.
- Second, $\mathbb{E}[\hat A]=F^\top\Lambda F$ is positive semidefinite, and the proof of Theorem 4.1 shows $(n\kappa_n)^{-1}(\hat A - F^\top\Lambda F)\to 0$. Under Assumption C.1(e), $\hat A$ is therefore well conditioned on the relevant rank-$r$ subspace with probability tending to one.
- Third, in practice we use a truncated pseudoinverse that discards eigendirections whose singular values fall below 1% of the largest. This is exactly the rank-$r$ estimator analyzed in Appendix F.7, so the guarantees cover the implemented procedure. With Paradigm 1, the system has only 2 dimensions per exposure level and is well conditioned in all our runs. [CONFIRM: rcond=1e-2 is the setting used for all reported tables.]

[TODO-EXP: report the distribution of condition numbers of $\hat A$, and the sensitivity of RMSE and coverage to the truncation level (1e-1 to 1e-4) and to a ridge alternative.]

**Small exposure probabilities and high-degree nodes.** When the $\pi_i^{(d)}$ are tiny, every IPW-based design estimator suffers, and GER cannot remove that. It does, however, reduce the residual magnitude that those weights multiply. [TODO-EXP: simulation with a growing maximum degree and power-law graphs, reporting variance reduction relative to Hájek and coverage, for fraction-based and "any-neighbor-treated" mappings.]

## Reviewer
Summary:
This paper proposes a regression adjustment framework for improving estimation efficiency under network interference. An important observation show by the authors is that minimizing prediction error does not necessarily minimize the variance of causal effect estimators. To address this issue, they develop a graph-weighted calibration method with theoretical guarantees on convergence, asymptotic normality, and variance reduction. The proposed method is evaluated through synthetic, semi-synthetic, and real-world experiments.

Strengths:
The paper addresses a meaningful and practically relevant problem in design-based causal inference under network interference. The proposed graph-weighted calibration framework is interesting and provides a principled approach to improving estimation efficiency beyond conventional prediction-based regression adjustment.

The theoretical analysis is relatively comprehensive.

The paper provides extensive empirical evaluations, demonstrating the potential benefits of the proposed approach.

Weaknesses:
The paper studies a important problem and provides a good framework for improving estimation efficiency under network interference. However, I have several concerns about the generality of the theoretical guarantees and the connection between the theory and practical implementation:

Assumption C.1(d) require the first-order exposure probabilities and positive second-order inclusion probabilities to be bounded below by 1/eta_n
, where eta_n = O(log n ^ gamma_1)
. This excludes even simple star shaped networks under simple neighborhood-based exposure mappings, where the dependency degree is Theta(n)
. Therefore, the theoretical guarantees in Theorems 4.1 and 4.2 have limited applicability to networks with highly connected nodes.

Assumption C.1 (a)–(b) require pi_i^(d) and pi_{ij}^{d,d'} > 1/eta_n, where \eta_n = O(log n ^\gamma_1). 
. This can be restrictive in some settings. For example, under independent Bernoulli treatment assignment with probability p
, the probability that none of a node's 
 neighbors is treated is pi_i^{(0)} = (1-p)^{d_i}
, which decreases exponentially with 
 and may violate the required lower bound.

The theoretical guarantee in Theorems 3.1 and 4.2 are established for given adjustment features, while the GNN representations in Section 5 are learned from observed data and may depend on the treatment assignments and outcomes. Although Appendix D.5 discusses graph-based cross-fitting, the paper does not explicitly establish that the complete GNN-assisted GAUGER pipeline satisfies the proposed efficiency guarantees.

The paper does not discuss an important reference: Adjusting Auxiliary Variables under Approximate Neighborhood Interference (Lu, Wang, and Zhang, 2024), which presents an elegant framework for design-based regression adjustment under network interference. Although this paper adopts a (relatively) different graph-weighted calibration approach, the two works share similar objectives. Given the relevance of this work, I believe a more thorough comparison is important for understanding the novelty and additional contributions of GAUGER.

Overall, I believe the above limitations raise concerns about the generality and practical applicability of GAUGER and I currently consider the paper to be borderline between acceptance and rejection. I am open to further discussion and would appreciate any clarification or correction from the authors if I have misunderstood any part of the paper.

Questions:
Could the authors provide a notation list summarizing the main symbols used throughout the paper? This would help improve readability.
The theoretical guarantees in Section 4.2 are established under Bernoulli randomization, whereas the Kenya field experiment in Section 6 uses a two-stage saturation design. Could the authors clarify whether the proposed consistency and efficiency guarantees extend to this randomization scheme?
In Tables 1 and 2, the differences between PNA-raw and PNA-based GER are relatively small. Could the authors further discuss under what conditions GER provides significant additional benefits beyond strong GNN predictions?
Appendix H.3 mentions that PNA-no-harm requires careful initialization and reports results using a particular random seed. Could the authors provide additional evidence on the stability of the results across different random seeds?

Clarity - Presentation: 3: Good (Generally clear; minor issues with organization, notation, or exposition that do not hinder understanding)
Soundness - Correctness: 2: Questionable (Significant unaddressed gaps, overly strong unverified assumptions, or weak empirical support)
Contribution - Significance: 2: Limited (Incremental over existing work, or empirical gains are small / lack key baselines or statistical support)
Rating: 3: Borderline Accept (Technically solid; merits outweigh minor limitations)
Confidence: 4: Confident (Understand the paper well; minor chance of missing subtle details)

### Author response (draft)

We thank the reviewer for the detailed, constructive review and for being open to discussion.

**Assumptions C.1(a), (b), (d) and high-degree nodes.** We agree the conditions exclude star-like graphs under neighborhood mappings, and we will say so explicitly. Two remarks:
- The bounded dependency degree (C.1(d)) is what normal approximation via dependency graphs (Chen and Shao, 2004) requires. Comparable sparsity conditions appear in Aronow and Samii (2017), Leung (2022) and Gao and Ding (2025). It is not specific to GAUGER.
- Your Bernoulli example is exactly right for the mapping "no neighbor treated": positivity then forces the degree to be $O(\log\log n)$. This is a property of that estimand: the HT and Hájek estimators of the same contrast also degenerate. For fraction-based mappings, like the ones used in our experiments, the exposure probabilities do not vanish as the degree grows.

We will add this discussion and point to design-side remedies: cluster randomization, trimming the estimand, or capping the degree, as in our Kenya network. [TODO-THEORY: whether a vanishing fraction of hub nodes can be accommodated by separating their contribution.] [TODO-EXP: heavy-tailed degree simulation.]

**Learned GNN features.** Please see our response to Reviewer AnSZ. In short, the guarantees apply fold-wise under the graph cross-fitting of Appendix D.5. We will state this as a formal corollary and report cross-fitted results. [TODO-THEORY/EXP as above.]

**Lu, Wang and Zhang (2024).** Thank you for pointing us to this paper. We will add a detailed comparison; please see the response to the AC for the main differences. [CONFIRM: details of their estimator.] [TODO-EXP: include it as a baseline.]

**Notation list.** We will add a notation table in the appendix.

**Two-stage saturation design (Kenya).** The proofs use the design only through two ingredients: known (Monte Carlo computable) inclusion probabilities, and a dependency graph $\Delta$ of bounded degree. Bernoulli assignment is a convenient sufficient case. Under the two-stage design, $\Delta$ must also connect villages in the same sublocation, which have dependent saturation draws. Because sublocations are small, the degree of $\Delta$ stays bounded and the same arguments go through. [CONFIRM: that the $\Delta$ used in the Kenya analysis includes sublocation co-membership, and the maximum degree of $\Delta$; if not, rerun with the corrected $\Delta$.] We will state the extension as a remark.

**When does GER help beyond a strong GNN?** GER is guaranteed not to hurt asymptotically, and its gain is large when the predictions are miscalibrated across exposure levels or their residuals are concentrated on units with large $\Lambda$ weights. When the backbone already lies near the variance-optimal direction, the gain is small, as in Table 1. In the denser setting of Table 6, GER reduces the PNA RMSE from 0.242 to 0.163 (−33%) and the GAT RMSE from 0.382 to 0.248. Note that all PNA rows in Table 2 are GER-based; the labels refer to the feature choice. We will clarify this. [TODO-EXP: add an uncalibrated PNA row to Table 2.]

**Seed stability.** We agree, and we will report all seeds, together with a seed-averaged model that involves no selection; see the AC response. [TODO-EXP: S = 50 seeds for the Kenya analysis and the synthetic studies.]