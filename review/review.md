# AISTATS 2027 review

## Meta Review
The authors propose a regression adjustment method for estimating treatment effects in randomized experiments with network interference: a prediction model (for example a graph neural network) is fitted first, and its output is then calibrated to reduce the variance of the effect estimator rather than the prediction error.

The initial reviews are positive, with all reviewers recommending acceptance. The reviewers find the paper well organized and the distinction between prediction accuracy and estimation efficiency well motivated.

Missing related work: Lu, Wang and Zhang (2024), "Adjusting Auxiliary Variables under Approximate Neighborhood Interference", studies design-based regression adjustment under network interference with the same objective; a detailed comparison is needed (Reviewer XK3s). Reviewer kJ1B also asks how the method relates to existing doubly robust and regression-based estimators under interference and what it adds beyond them.

Can you clarify how models and random seeds were selected in the experiments? Appendix H.3 states that one variant "requires careful initialization to achieve competitive performance" and reports results for a specific seed, and Reviewer XK3s asks for evidence of stability across seeds. If hyperparameters or seeds are chosen on the realized data to minimize confidence interval width, and interval width is then the metric used to show efficiency gains, the evaluation is circular. The authors should state exactly what was tuned on what, and report results over seeds.

The assumptions require exposure probabilities bounded below at a polylogarithmic rate, which excludes networks with high-degree nodes; under Bernoulli assignment the probability that no neighbour is treated decays exponentially in the degree (Reviewer XK3s).

### Author response (draft)

> Limit: 5,000 characters per response. All numbers below come from `code/rebuttal/results/summary.txt` (experiments 1–7). Remaining placeholders: `[TODO-THEORY]` = result to write up.

We thank the AC for the careful summary.

**1. Lu, Wang and Zhang (2024) and DR / regression estimators.** We will add LWZ to Section 1.3 and Appendix A. LWZ develop HT- and Hájek-based regression adjustment estimators, with adjustment coefficients obtained by minimizing a network HAC variance estimator over pairs within a prescribed bandwidth ($b_n=3$ in their simulation, i.e. our two-hop $\Delta$). Their framework accommodates assignment-dependent auxiliary variables, with specific constructions motivated by linear-in-means and nonlinear contagion models. They introduce a normalization $\phi_0$ that makes the asymptotic bias of the HAC variance estimator independent of the adjustment coefficients. GAUGER differs in (i) multi-level exposures with exposure-specific features in AIPW form, (ii) a design-based variance with exact inclusion probabilities ($\Lambda$) instead of HAC, and (iii) learned (GNN) features instead of model-motivated ones. We implemented LWZ on the designs of Tables 1 and 6 and compare the best LWZ variant in each setting with GAUGER-PNA on the same 50 assignments (RMSE / coverage):
- Table 1 setting: GAUGER-PNA 0.129 / 0.98 vs. LWZ 0.172 / 0.90 (ND-$\phi_0(G_2)$).
- Table 6 setting (denser): GAUGER-PNA 0.162 / 0.98 vs. LWZ 0.283 / 0.90 (ND-$\phi_0(G_1)$).

LWZ's HAC intervals under-cover in our settings, and their network auxiliary variables help less on our outcome model, which is not linear-in-means. GAUGER learns features and uses design-based inference, with better accuracy and coverage.


*Doubly robust estimators.* DR estimators under interference (kJ1B's [1, 2]) are developed for observational data in a super-population framework: both propensities and outcomes are modeled, and DR protects against misspecification of either. We work instead in the design-based framework with fixed potential outcomes (Rubin, 2005), where the only randomness is the known assignment. Propensities are then known exactly, so every AIPW estimator is design-unbiased for any outcome model (Theorem 3.1), and double robustness is automatic. What remains open is efficiency: DR methods fit the outcome model by prediction loss, whereas GAUGER chooses the AIPW adjustment to minimize the design variance, the $\Lambda$-weighted form that includes the cross-unit covariances from interference. Theorem 4.2 makes it asymptotically no worse than any fixed adjustment in the same feature span, including the MSE-fitted one (DR with known propensities) and least-squares adjustment (Gao and Ding, kJ1B's [3]), which treats units as uncorrelated. In our tables, "raw" is AIPW with known propensities and the MSE-trained predictor, and "pred" applies GER to the same predictor: the SD drops from 0.202 to 0.160 (linear, Table 1), 0.382 to 0.221 (GAT, Table 6) and 0.241 to 0.159 (PNA, Table 6).

**2. Model and seed selection.** Our model selection protocol was: prespecify a grid of parameters, then choose the configuration that minimize held-out metrics such as MSE (same as maximizing $R^2$) or held-out variance estimates (same as minimizing interval length) on randomly held-out nodes. We apologize for the confusion by our wording that PNA-based estimator "requires careful initialization" - it does not mean that we were trying to do a circular evaluation, but says we need to do a careful grid search instead of just using the PNA repo defaults like four aggregators. As we mentioned in the paper, we found simpler models with fewer aggregators and moderate regularization yield shorter intervals. 

Nevertheless, we do agree that a better evaluation should average results over seeds for the Kenya application. We conduct more experiments on Kenya data over 50 seeds. Conclusions are stable: revenue Direct($H=1$) is significant for 100% of seeds; wage-bill Direct($H=0$) and Spillover($A=0$) for 98%. The efficiency claims are also stable: the average CI width for PNA/Hajek/Linear is 3,119/3,214/3,321 for revenue and 520/521/562 for wage bill, which maintains the papers' conclusion that GER should lead to similar or more efficiency over existing baselines. 

We will replace Tables 3, 8, 9, 10 by the seed-averaged model with the all-seed distributions, and revise the text: on the Kenya data GAUGER matches the statistical conclusion from Hájek and linear adjustment, while providing similar or more efficiency. We will also state what was tuned on what more clearly in Appendices G and H.3.

**3. Positivity with high-degree nodes.** We fully agree with this. Our current framework targets mostly a sparser degree setting, as required by our Assumption C.1(d), which is usually needed for positivity as well. We ran two new studies to check how this matters in practice:
- Heavy-tailed degrees (Chung–Lu graphs, $n=2{,}000$, mean degree 3, maximum degree capped at 10–300, 200 draws): the minimum $\pi$ falls to $10^{-20}$, so C.1(a)–(b) fail, yet GER still lowers the SD by 49–72% relative to Hájek at every cap, with coverage 0.96–1.00 (Hájek 0.80–0.90).
- Uniformly dense graphs (BlogCatalog and Flickr, mean degree 63–66): exposure probabilities stay above 0.27, but the two-hop $\Delta$ covers 48–58% of all pairs and is strongly indefinite, and GER becomes unstable (biased toward zero by 30–49% of $\tau$). A PSD weight, the diagonal matrix of two-hop degrees, removes the bias but gives no gain over uncalibrated AIPW with PNA, which is already about 3× more accurate than Hájek.

We will state this scope explicitly (GER targets sparse dependence) and add these experiments on denser networks with more discussions. 

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


**Relation to DR / efficient estimation [1–3].** [1] and [2] are for observational data: data are drawn from a super population, and DR there protects against misspecification of the outcome model or the propensity model. In our randomization setup, potential outcomes are treated as fixed quantities following Rubin (2005). Also, in randomization, propensities are known exactly, so every AIPW estimator is unbiased for any outcome model (Theorem 3.1), and the open question is efficiency. Here GER adds something the DR estimators do not: it chooses the adjustment to minimize the design variance, which, under interference, is the $\Lambda$-weighted quadratic form, not the prediction MSE that [1, 2] effectively target. [3] (Gao and Ding) is design-based and uses (weighted) least-squares adjustment. Theorem 4.2 implies that, for the same features, GER is asymptotically no worse than any fixed coefficient, including a least-squares one such as that in [3]. The reason is that a least-squares fit treats units as uncorrelated, whereas under interference the design variance contains cross-unit covariance terms (the off-diagonal entries of $\Lambda$); GER is built to target exactly these terms. Empirically, GER on the same covariates lowers the RMSE of least-squares adjustment (linear-raw) from 0.203 to 0.173 (Table 1) and from 0.467 to 0.245 (Table 6).

**Ablation under the same backbone.** Tables 1 and 6 already contain this ablation; we apologize for not explaining the variants clearly. For every backbone (linear, GAT, PNA):
- "raw" is the uncalibrated AIPW that plugs in the MSE-trained predictions (for the linear backbone: per-exposure OLS, i.e. standard regression adjustment); Table 6 RMSE 0.467 / 0.382 / 0.242 for linear / GAT / PNA (Hájek 0.368);
- "pred" is GER applied to those same predictions (Paradigm 1); RMSE 0.270 / 0.250 / 0.164;
- "linear" is GER applied to the covariates or learned representations (Paradigm 2); RMSE 0.245 / 0.248 / 0.163.

**Mismatch between prediction MSE and variance.** The raw-versus-pred comparison highlights this mismatch directly. Both use the same predictor; "raw" keeps the MSE-trained output as is, while "pred" only re-weights it through GER. Any drop from raw to pred is therefore variance that the MSE criterion leaves on the table: the SD falls from 0.202 to 0.160 (linear, Table 1), from 0.382 to 0.221 (GAT, Table 6) and from 0.241 to 0.159 (PNA, Table 6).

The gap is small when the predictor is already close to variance-optimal (PNA in Table 1) and large when it is not (PNA in Table 6), as Theorem 3.1 predicts. We will make this reading explicit in Section 6.

**Dense networks (Flickr, BC).** We ran this. We built semi-synthetic studies on BlogCatalog ($n=5{,}196$, mean degree 66) and Flickr ($n=7{,}575$, mean degree 63) with the same outcome model as our semisynthetic study, Bernoulli(1/3) assignment and the fraction mapping ($k=3$); 30 draws (GAT: 20). Three findings, which we will report in full:
- Positivity: we estimate the contrast $d=1$ vs. $0$, which has good support on both graphs: every unit has $\pi_i^{(0)},\pi_i^{(1)}\ge 0.27$ (minimum 0.32 / 0.42 on BlogCatalog, 0.27 / 0.27 on Flickr). Hájek has RMSE 0.014 / 0.016.
- Learned features help a lot: uncalibrated AIPW with PNA has RMSE 0.0049 / 0.0044, about 3× smaller than Hájek, with coverage 1.00; GAT reaches 0.0088 / 0.0166.
- GER as specified in the paper is not a strong candidate here: the two-hop $\Delta$ is very dense, covering 48–58% of all pairs, and is strongly indefinite (smallest eigenvalue $-330$ / $-891$), and the calibrated estimates are biased toward zero by 30–49% of $\tau$ for PNA and GAT. Replacing $\Delta$ by the positive semidefinite bound $\mathrm{diag}(\Delta\mathbf 1)\succeq\Delta$ removes the bias (−1% to −3% of $\tau$, coverage 0.93–1.00) but gives no gain over uncalibrated AIPW (PNA RMSE 0.010 / 0.023).

We will stress this more in the paper: GER's efficiency guarantees and gains concern sparse dependence (bounded $\Delta$-degree, Assumption C.1(d)); when two-hop neighborhoods cover a constant fraction of the graph, practitioners should use a PSD weight or uncalibrated AIPW with learned features.

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
- (i) Cross-fitting (Appendix D.5). Cross-fitting is a standard device for using machine-learned adjustments in design-based inference (e.g., Lu, Shi, Liu and Ding, 2025, "Conditional cross-fitting for unbiased machine-learning-assisted covariate adjustment in randomized experiments"). In our graph version, the model used for one fold is trained only on buffered units whose exposures are independent of that fold's exposures, so, conditional on the training fold, the features are fixed and the design of the evaluation fold is unchanged. Adapting the argument of Lu et al. (2025), Theorems 3.1, 4.1 and 4.2 extend to the final cross-fitted estimator that combines the folds; we will add this as a corollary with proof. Empirically, cross-fitting matches full fit: on the student network (Table 2), cross-fitted PNA is close, with RMSE 0.0110 vs. 0.0106 (Hájek 0.0488) and coverage 0.98–1.00. 
- (ii) Without cross-fitting, the empirical coverage of the calibrated estimators is 0.92–1.00 in Tables 1 and 6, which suggests that the dependence induced by learning is mild there. 
We will add the cross-fitting experiments in next iteration.

**Indefinite $\Delta$ and singular systems.** We agree this deserved more space.
- We solve the linear system $\hat A\beta=\hat b$; since $\mathbb{E}[\hat A]=F^\top\Lambda F\succeq 0$ and $(n\kappa_n)^{-1}(\hat A - F^\top\Lambda F)\to 0$ (proof of Theorem 4.1), $\hat A$ is well conditioned on the relevant rank-$r$ subspace with probability tending to one under Assumption C.1(e).
- In practice we use a truncated pseudoinverse (relative threshold $10^{-2}$, used for all reported tables), which is the rank-$r$ estimator analyzed in Appendix F.7.
- New diagnostics (50 draws, Tables 1 and 6 settings): the condition number of $\hat A$ has median 12–24 (max 47) for GER on predictions (Paradigm 1) and median 72–143 (max 391) for GER on covariates. For GER on PNA representations, $\hat A$ is numerically singular (the representation dimensions are not linearly independent), so the truncation matters.
- Sensitivity: for linear GER, RMSE and coverage are identical for thresholds $10^{-3}$ to $10^{-8}$ and for a ridge alternative; a coarse threshold $10^{-1}$ hurts (RMSE 0.188 to 0.244, 50-draw rerun on Table 1). For GER on PNA representations, thresholds $10^{-2}$–$10^{-4}$ and ridge agree (0.123–0.129), while $10^{-8}$ increases the RMSE to 0.142 (Table 1) and 0.220 (Table 6).

We will therefore recommend either Paradigm 1 of Section 5 (GER on the predictions), which is low-dimensional and well conditioned, or a regularized solve (truncated pseudoinverse with a moderate threshold, or ridge) for representation features. In finite samples the indefiniteness of $\Delta$ can matter: on very dense graphs (BlogCatalog/Flickr, where two-hop pairs cover about half of all pairs) the $\Delta$-weighted system is badly conditioned and GER is biased; the PSD bound $\mathrm{diag}(\Delta\mathbf 1)$ fixes this (see our response to AC and kJ1B).

**Small exposure probabilities and high-degree nodes.** We agree that our framework mainly targets sparse dependence, as required by Assumption C.1(d), which in practice also keeps exposure probabilities away from zero. We ran two new studies to check how this matters in practice:
- Heavy-tailed degrees (Chung–Lu graphs, $n=2{,}000$, mean degree 3, maximum degree capped at 10, 30, 100 and 300; Bernoulli(1/3); fraction-based and "any neighbor treated" mappings; 200 draws): the minimum exposure probability falls to $10^{-20}$ and $10^{-32}$, so C.1(a)–(b) fail. Yet across all caps and both mappings GER lowers the SD by 49–72% relative to Hájek, with coverage 0.96–1.00 (Hájek 0.80–0.90; uncalibrated AIPW has 8–79% larger SD than Hájek). GER cannot remove the inverse weights, but it shrinks the residuals they multiply.
- Uniformly dense graphs (BlogCatalog and Flickr, mean degree 63–66): exposure probabilities stay above 0.27, but the two-hop $\Delta$ covers 48–58% of all pairs and is strongly indefinite, and GER becomes unstable (biased toward zero by 30–49% of $\tau$). The PSD weight above removes the bias but gives no gain over uncalibrated AIPW with PNA, which is already about 3× more accurate than Hájek.

We will state this scope explicitly (GER targets sparse dependence) and add both studies to the paper.

## Reviewer XK3s
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
, which decreases exponentially with d_i
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
- The bounded dependency degree (C.1(d)) is what normal approximation via dependency graphs (Chen and Shao, 2004) requires. Comparable sparsity conditions appear in Aronow and Samii (2017), Leung (2022), Gao and Ding (2025), Lu, Wang, and Zhang (2024), which all assume uniform marginal positivity where exposure propensity is strictly bounded by constants. It is not specific to GAUGER.
- The reviewer's Bernoulli example is correct: for the exposure mapping corresponding to no treated neighbors, the exposure probability decreases exponentially with node degree. This is a fundamental overlap issue arising from the combination of the exposure mapping, the target estimand, and the randomization design, rather than a limitation specific to our estimator. Other weighting estimators targeting the same contrast, including HT and Hájek, can similarly suffer from instability. In such settings, more appropriate alternatives may involve imposing additional structure on the exposure-response relationship (e.g., Harshaw et al., 2023) or adopting experimental designs that improve exposure overlap (e.g., graph cluster randomization in Ugander et al., 2013).

To check how much this matters in practice, we ran a new simulation with same synthetic outcome model on heavy-tailed graphs (Chung–Lu graph, $n=2{,}000$, mean degree 3) with the maximum degree capped at 10, 30, 100 and 300, Bernoulli(1/3) assignment and 200 draws. As the cap grows, the smallest exposure probability falls to $10^{-20}$, so C.1(a)–(b) fail. Even so, GER reduces the SD by 49–72% relative to Hájek at every cap, and its intervals cover at 0.96–1.00, while Hájek's cover at only 0.80–0.90. Our guarantees do not cover these graphs, but the estimator remains well behaved. We will add this simulation and state the limitation explicitly, together with the remedies above and degree capping (as in our Kenya network).

**Learned GNN features.** Cross-fitting is standard for machine-learned adjustments in design-based inference (Lu, Shi, Liu and Ding, 2025). With our graph cross-fitting (Appendix D.5), the features for each fold are trained on buffered units whose exposures are independent of that fold's exposures, so they are fixed given the training fold. Adapting Lu et al. (2025), the guarantees of Theorems 3.1, 4.1 and 4.2 extend to the final cross-fitted estimator; we will state this as a corollary with proof. Empirically, cross-fitted PNA on the student network matches the full-sample fit (RMSE 0.0110 vs. 0.0106, Hájek 0.0488; coverage 0.98–1.00); see our response to AnSZ.

**Lu, Wang and Zhang (2024).** Thank you for pointing us to this paper; we implemented it. LWZ choose regression coefficients by minimizing a HAC variance estimate (bandwidth 3, i.e. the same two-hop pairs as our $\Delta$) with auxiliary variables motivated by linear-in-means working models and a normalization $\phi_0$. Comparing the best LWZ variant in each setting with GAUGER-PNA on the same 50 assignments (RMSE / coverage): 0.172 / 0.90 vs. 0.129 / 0.98 on the Table 1 design, and 0.283 / 0.90 vs. 0.162 / 0.98 on the denser Table 6 design. We will add this comparison and a discussion to the paper.

**Notation list.** We will add a notation table in the appendix.

**Two-stage saturation design (Kenya).** Our proofs use the design only through known (Monte Carlo computable) inclusion probabilities, positivity, and a dependency graph $\Delta$ of bounded degree; Bernoulli assignment is one design that satisfies these. In the Kenya design, conditional on the stage-1 sublocation saturations, villages are independent draws with probability 2/3 or 1/3, and since the exposure $(A_i,S_i)$ depends only on village $i$ and its neighbours, the two-hop graph is a valid $\Delta$. Unconditionally, villages in the same sublocation share the saturation draw and are dependent; our $\Delta$ already includes these same-sublocation pairs, so it is valid for the full two-stage design. Sublocations are small (at most 17 villages), so $\Delta$ keeps bounded degree (mean 24, max 41), and our consistency and efficiency results apply. We will add this as a remark.

**When does GER help beyond a strong GNN?** GER is guaranteed not to hurt asymptotically, and its gain is large when the predictions are miscalibrated across exposure levels or their residuals are concentrated on units with large $\Lambda$ weights. When the backbone already lies near the variance-optimal direction, the gain is small, as in Table 1. In the denser setting of Table 6, GER reduces the PNA RMSE from 0.242 to 0.163 (−33%) and the GAT RMSE from 0.382 to 0.248. We will add an uncalibrated PNA row to Table 2: in our rerun (50 draws), PNA-raw and PNA with GER on predictions have RMSE 0.0107 and 0.0106 (Hájek 0.0488). GER also has limits that we will state: on very dense graphs (BlogCatalog, Flickr) the $\Delta$-weighted calibration is biased and should use a PSD weight (see the kJ1B response).

**Seed stability.** See the AC response for details. For Kenya data, 50 seeds: the significant effects are reproduced for 98–100% of seeds, and the efficiency claims are also stable: the average CI width for PNA/Hajek/Linear is 3,119/3,214/3,321 for revenue and 520/521/562 for wage bill, which maintains the papers' conclusion. 



## Final version

### AC

We thank AC for the careful summary.

1. LWZ (2024) and DR. LWZ develop regression adjustment by minimizing a network HAC variance estimator. They use auxiliary variables motivated by working models. GAUGER differs in (i) multi-level exposures with AIPW form, (ii) variance with exact inclusion probabilities not HAC, and (iii) learned features not working models. We test LWZ and compare its best variant with GAUGER-PNA (RMSE/coverage) in Table 1: PNA 0.129/0.98 vs LWZ 0.172/0.90. LWZ undercovers and the auxiliary variables help less here. 

DR (kJ1B's [1, 2]) are for observational data: both propensities and outcomes are modeled, and DR protects against misspecification of either. We work instead in design-based setup, where propensities are known exactly, so AIPW estimator is design-unbiased for any outcome model. What remains open is efficiency: DR fit the outcome model by MSE, yet GER minimizes the design variance. It is no worse than any fixed adjustment in the same feature span, including the MSE-fits and LS (Gao-Ding, kJ1B's [3]). In our tables, using usual DR vs GER, the SD drops from 0.202 to 0.160 (linear) and 0.241 to 0.159 (PNA).

2. Model/seed selection. Our model selection protocol was: specify a parameter grid and choose the configuration that minimize validation metrics such as MSE (max $R^2$) or variance (min CI length) on held-out nodes. Sorry for our vague wording: we did not mean to do circular evaluation, but try to propose a careful grid search instead of using the PNA defaults like four aggregators. As we said in paper, fewer aggregators and moderate regularization yield shorter intervals. 

We further report 50 seeds for Kenya study. Results are stable: revenue(H=1) is significant for 100% seeds; wage-bill(H=0) and (A=0) for 98%. The efficiency claims also hold: the mean CI width for PNA/Hajek/Linear is 3119/3214/3321 for revenue and 520/521/562 for wage bill.

3. Positivity with high-deg nodes. We fully agree. Our work targets a sparse setting. We run two new studies:
- Heavy-tailed degrees (Chung–Lu graph, mean deg 3, max deg capped at 10–300): the min $\pi$ falls to $10^{-20}$, yet GER still lowers the SD by 49–72% vs Hajek, with coverage 0.96–1.00 (Hajek 0.80–0.90).
- Uniformly dense graphs (BC and Flickr, mean deg 63–66): the two-hop $\Delta$ is dense and strongly indefinite, and GER becomes unstable (biased toward zero by 30–49%). Uncalibrated AIPW with PNA is already about 3x more accurate than Hajek, no need for calibration.


### Reviewer kJ1B
Thank you for the positive assessment and the concrete suggestions.

**Unknown or misspecified exposure mapping.** The main theory assumes a known mapping, as is standard in design-based work (Aronow and Samii, 2017). Two extensions are already in the appendix:
- Appendix D.6.2 follows Leung (2022): under misspecification, the estimator remains well defined and targets the contrast of expected outcomes $\mu(d)$ at each exposure value. The design-based unbiasedness of AIPW and the calibration argument do not rely on the mapping being correct, since the randomization is known.
- Appendix D.6.3 sketches a GNN-based procedure for learning the mapping.


**Relation to DR / efficient estimation [1–3].** [1] and [2] are for observational data: data are drawn from a super population, and DR there protects against misspecification of the outcome model or the propensity model. In our randomization setup, potential outcomes are treated as fixed quantities following Rubin (2005). Also, in randomization, propensities are known exactly, so every AIPW estimator is unbiased for any outcome model (Theorem 3.1), and the open question is efficiency. Here GER adds something the DR estimators do not: it chooses the adjustment to minimize the design variance, which, under interference, is the $\Lambda$-weighted quadratic form, not the prediction MSE that [1, 2] effectively target. [3] (Gao and Ding) is design-based and uses (weighted) least-squares adjustment. Theorem 4.2 implies that, for the same features, GER is asymptotically no worse than any fixed coefficient, including a least-squares one such as that in [3]. The reason is that a least-squares fit treats units as uncorrelated, whereas under interference the design variance contains cross-unit covariance terms (the off-diagonal entries of $\Lambda$); GER is built to target exactly these terms. Empirically, GER on the same covariates lowers the RMSE of least-squares adjustment (linear-raw) from 0.203 to 0.173 (Table 1) and from 0.467 to 0.245 (Table 6).

**Ablation under the same backbone.** Tables 1 and 6 already contain this ablation; we apologize for not explaining the variants clearly. For every backbone (linear, GAT, PNA):
- "raw" is the uncalibrated AIPW that plugs in the MSE-trained predictions (for the linear backbone: per-exposure OLS, i.e. standard regression adjustment); Table 6 RMSE 0.467 / 0.382 / 0.242 for linear / GAT / PNA (Hájek 0.368);
- "pred" is GER applied to those same predictions (Paradigm 1); RMSE 0.270 / 0.250 / 0.164;
- "linear" is GER applied to the covariates or learned representations (Paradigm 2); RMSE 0.245 / 0.248 / 0.163.

**Mismatch between prediction MSE and variance.** The raw-versus-pred comparison highlights this mismatch directly. Both use the same predictor; "raw" keeps the MSE-trained output as is, while "pred" only re-weights it through GER. Any drop from raw to pred is therefore variance that the MSE criterion leaves on the table: the SD falls from 0.202 to 0.160 (linear, Table 1), from 0.382 to 0.221 (GAT, Table 6) and from 0.241 to 0.159 (PNA, Table 6).

The gap is small when the predictor is already close to variance-optimal (PNA in Table 1) and large when it is not (PNA in Table 6), as Theorem 3.1 predicts. We will make this reading explicit in Section 6.

**Dense networks (Flickr, BC).** We ran this. We built semi-synthetic studies on BlogCatalog ($n=5{,}196$, mean degree 66) and Flickr ($n=7{,}575$, mean degree 63) with the same outcome model as our semisynthetic study, Bernoulli(1/3) assignment and the fraction mapping ($k=3$); 30 draws (GAT: 20). Three findings, which we will report in full:
- Positivity: we estimate the contrast $d=1$ vs. $0$, which has good support on both graphs: every unit has $\pi_i^{(0)},\pi_i^{(1)}\ge 0.27$ (minimum 0.32 / 0.42 on BlogCatalog, 0.27 / 0.27 on Flickr). Hájek has RMSE 0.014 / 0.016.
- Learned features help a lot: uncalibrated AIPW with PNA has RMSE 0.0049 / 0.0044, about 3× smaller than Hájek, with coverage 1.00; GAT reaches 0.0088 / 0.0166.
- GER as specified in the paper is not a strong candidate here: the two-hop $\Delta$ is very dense, covering 48–58% of all pairs, and is strongly indefinite (smallest eigenvalue $-330$ / $-891$), and the calibrated estimates are biased toward zero by 30–49% of $\tau$ for PNA and GAT. Replacing $\Delta$ by the positive semidefinite bound $\mathrm{diag}(\Delta\mathbf 1)\succeq\Delta$ removes the bias (−1% to −3% of $\tau$, coverage 0.93–1.00) but gives no gain over uncalibrated AIPW (PNA RMSE 0.010 / 0.023).

We will stress this more in the paper: GER's efficiency guarantees and gains concern sparse dependence (bounded $\Delta$-degree, Assumption C.1(d)); when two-hop neighborhoods cover a constant fraction of the graph, practitioners should use a PSD weight or uncalibrated AIPW with learned features.

### Reviewer AnSZ
We thank the reviewer for the supportive evaluation and the precise questions.

**Learned features vs. fixed-feature theory.** Our theorems condition on features that do not depend on the realized assignment. We take two routes to connect this to the GNN pipeline.
- (i) Cross-fitting (Appendix D.5). Cross-fitting is a standard device for using machine-learned adjustments in design-based inference (e.g., Lu, Shi, Liu and Ding, 2025, "Conditional cross-fitting for unbiased machine-learning-assisted covariate adjustment in randomized experiments"). In our graph version, the model used for one fold is trained only on buffered units whose exposures are independent of that fold's exposures, so, conditional on the training fold, the features are fixed and the design of the evaluation fold is unchanged. Adapting the argument of Lu et al. (2025), Theorems 3.1, 4.1 and 4.2 extend to the final cross-fitted estimator that combines the folds; we will add this as a corollary with proof. Empirically, cross-fitting matches full fit: on the student network (Table 2), cross-fitted PNA is close, with RMSE 0.0110 vs. 0.0106 (Hájek 0.0488) and coverage 0.98–1.00. 
- (ii) Without cross-fitting, the empirical coverage of the calibrated estimators is 0.92–1.00 in Tables 1 and 6, which suggests that the dependence induced by learning is mild there. 
We will add the cross-fitting experiments in next iteration.

**Indefinite $\Delta$ and singular systems.** We agree this deserved more space.
- We solve the linear system $\hat A\beta=\hat b$; since $\mathbb{E}[\hat A]=F^\top\Lambda F\succeq 0$ and $(n\kappa_n)^{-1}(\hat A - F^\top\Lambda F)\to 0$ (proof of Theorem 4.1), $\hat A$ is well conditioned on the relevant rank-$r$ subspace with probability tending to one under Assumption C.1(e).
- In practice we use a truncated pseudoinverse (relative threshold $10^{-2}$, used for all reported tables), which is the rank-$r$ estimator analyzed in Appendix F.7.
- New diagnostics (50 draws, Tables 1 and 6 settings): the condition number of $\hat A$ has median 12–24 (max 47) for GER on predictions (Paradigm 1) and median 72–143 (max 391) for GER on covariates. For GER on PNA representations, $\hat A$ is numerically singular (the representation dimensions are not linearly independent), so the truncation matters.
- Sensitivity: for linear GER, RMSE and coverage are identical for thresholds $10^{-3}$ to $10^{-8}$ and for a ridge alternative; a coarse threshold $10^{-1}$ hurts (RMSE 0.188 to 0.244, 50-draw rerun on Table 1). For GER on PNA representations, thresholds $10^{-2}$–$10^{-4}$ and ridge agree (0.123–0.129), while $10^{-8}$ increases the RMSE to 0.142 (Table 1) and 0.220 (Table 6).

We will therefore recommend either Paradigm 1 of Section 5 (GER on the predictions), which is low-dimensional and well conditioned, or a regularized solve (truncated pseudoinverse with a moderate threshold, or ridge) for representation features. In finite samples the indefiniteness of $\Delta$ can matter: on very dense graphs (BlogCatalog/Flickr, where two-hop pairs cover about half of all pairs) the $\Delta$-weighted system is badly conditioned and GER is biased; the PSD bound $\mathrm{diag}(\Delta\mathbf 1)$ fixes this (see our response to AC and kJ1B).

**Small exposure probabilities and high-degree nodes.** We agree that our framework mainly targets sparse dependence, as required by Assumption C.1(d), which in practice also keeps exposure probabilities away from zero. We ran two new studies to check how this matters in practice:
- Heavy-tailed degrees (Chung–Lu graphs, $n=2{,}000$, mean degree 3, maximum degree capped at 10, 30, 100 and 300; Bernoulli(1/3); fraction-based and "any neighbor treated" mappings; 200 draws): the minimum exposure probability falls to $10^{-20}$ and $10^{-32}$, so C.1(a)–(b) fail. Yet across all caps and both mappings GER lowers the SD by 49–72% relative to Hájek, with coverage 0.96–1.00 (Hájek 0.80–0.90; uncalibrated AIPW has 8–79% larger SD than Hájek). GER cannot remove the inverse weights, but it shrinks the residuals they multiply.
- Uniformly dense graphs (BlogCatalog and Flickr, mean degree 63–66): exposure probabilities stay above 0.27, but the two-hop $\Delta$ covers 48–58% of all pairs and is strongly indefinite, and GER becomes unstable (biased toward zero by 30–49% of $\tau$). The PSD weight above removes the bias but gives no gain over uncalibrated AIPW with PNA, which is already about 3× more accurate than Hájek.

We will state this scope explicitly (GER targets sparse dependence) and add both studies to the paper.


### Reviewer XK3s
We thank the reviewer for the constructive review.

- Assumptions C.1(a), (b), (d) and high-degree nodes. We agree the conditions exclude star-like graphs under neighborhood mappings, and we will say so explicitly. Two remarks:
  1. The bounded dependency degree (C.1(d)) is what normal approximation via dependency graphs (Chen and Shao, 2004) requires. Similar conditions appear in Aronow and Samii (2017), Leung (2022), Gao and Ding (2025), Lu, Wang, and Zhang (2024), all assuming uniform marginal positivity where propensity is strictly bounded by constants. It is not specific to GAUGER.
  2. The Bernoulli example is correct: for all-zero exposure, the probability decreases exponentially with node degree. This is a fundamental overlap issue arising from exposure mapping, target estimand, and randomization design, rather than a limitation specific to our estimator. Other weighting estimators targeting the same contrast, including HT and Hájek, can similarly suffer from instability. In such settings, better alternatives are: imposing additional structure on the exposure-response relationship (e.g., Harshaw et al., 2023) or using designs that improve exposure overlap (e.g., graph cluster randomization in Ugander et al., 2013).

    To check how much this matters in practice, we ran a new simulation with same synthetic outcome model on heavy-tailed graphs (Chung–Lu graph, $n=2{,}000$, mean degree 3) with the maximum degree capped at 10, 30, 100 and 300, Bernoulli(1/3) assignment and 200 draws. As the cap grows, the smallest exposure probability falls to $10^{-20}$, so C.1(a)–(b) fail. Even so, GER reduces the SD by 49–72% relative to Hájek at every cap, and its intervals cover at 0.96–1.00, while Hájek's cover at only 0.80–0.90. Our guarantees do not cover these graphs, but the estimator remains well behaved. But if the network gets uniformly dense, the proposal can be instable (See response to AC and Reviewer kJ1B). We will add this simulation and state the limitation, together with the remedies above.

- Learned GNN features. Cross-fitting is standard for machine-learned adjustments in design-based inference (Lu, Shi, Liu and Ding, 2025). With our graph cross-fitting (Appendix D.5), the features for each fold are trained on buffered units whose exposures are independent of that fold's exposures, so they are fixed given the training fold. Adapting Lu et al. (2025), the guarantees of Theorems 3.1, 4.1 and 4.2 extend to the final cross-fitted estimator; we will state this as a corollary with proof. Empirically, cross-fitted PNA on the student network matches the full-sample fit (RMSE 0.0110 vs. 0.0106, Hájek 0.0488; coverage 0.98–1.00); see our response to AnSZ.

- Lu, Wang and Zhang (2024). Thank you for pointing us to this paper; we implemented it. LWZ choose regression coefficients by minimizing a HAC variance estimate with auxiliary variables motivated by linear-in-means working models and a normalization $\phi_0$. Comparing the best LWZ variant in each setting with GAUGER-PNA on the same 50 assignments (RMSE / coverage): 0.172 / 0.90 vs. 0.129 / 0.98 on the Table 1 design, and 0.283 / 0.90 vs. 0.162 / 0.98 on the denser Table 6 design. We will add this comparison and a discussion to the paper.

- Notation list. We will add a notation table in the appendix.

- Two-stage saturation design (Kenya). Our proofs use the design only through known (Monte Carlo computable) inclusion probabilities, positivity, and a dependency graph $\Delta$ of bounded degree; Bernoulli assignment is one design that satisfies these. In the Kenya design, conditional on the stage-1 sublocation saturations, villages are independent draws with probability 2/3 or 1/3, and since the exposure $(A_i,S_i)$ depends only on village $i$ and its neighbours, the two-hop graph is a valid $\Delta$. Unconditionally, villages in the same sublocation share the saturation draw and are dependent; our $\Delta$ already includes these same-sublocation pairs, so it is valid for the full two-stage design. Sublocations are small (at most 17 villages), so $\Delta$ keeps bounded degree (mean 24, max 41), and our consistency and efficiency results apply. We will add this as a remark.

- When does GER help beyond a strong GNN? GER is guaranteed not to hurt asymptotically, and its gain is large when the predictions are miscalibrated across exposure levels or their residuals are concentrated on units with large $\Lambda$ weights. When the backbone already lies near the variance-optimal direction, the gain is small, as in Table 1. In the denser setting of Table 6, GER reduces the PNA RMSE from 0.242 to 0.163 and the GAT RMSE from 0.382 to 0.248. 

- Seed stability. See the AC response for details. For Kenya data, 50 seeds: the significant effects are reproduced for 98–100% of seeds, and the efficiency claims are also stable: the average CI width for PNA/Hajek/Linear is 3119/3214/3321 for revenue and 520/521/562 for wage bill, which maintains the papers' conclusion.