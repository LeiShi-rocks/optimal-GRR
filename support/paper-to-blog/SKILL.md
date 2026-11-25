---
name: paper-to-blog
description: 'Generate a Distill-style technical blog from a research paper. Use when asked to explain a paper as a blog post, write a research explainer or distillation, or turn an academic paper into an accessible, visual technical article with figures and experimental results.'
---

# Technical Blog from a Paper

## Purpose

Use a paper as source material to write a standalone, Distill-style article about its central idea. The article is not a shortened paper, a section-by-section summary, or promotional copy. It is a distillation: it does the interpretive work so the reader does not have to. A reader should understand the problem, why the approach works, what the experiments show, and when the result matters, without opening the source.

Good distillation often transforms the idea rather than polishing the paper's presentation. Look for a better frame than the one the paper uses: a simpler model that exposes the mechanism, a clearer notation, a visual way of thinking, or a running example that makes each step concrete.

Hold the article to three standards:

- **Advance understanding.** Offer a perspective a reader would not easily get from the paper itself. A short, crisp article with a narrow but valuable scope is better than broad coverage.
- **Communicate outstandingly.** The structure is focused, the writing avoids needless jargon, and the figures serve as tools for thought. Readability is judged relative to the difficulty of the topic.
- **Maintain scientific integrity.** Claims match the evidence, experiments are presented fairly, conjecture is labeled, limitations are clear even to a non-expert, and results are easy to check or replicate.

## Defaults

- **Audience:** technically literate readers with basic ML or statistics knowledge, but no assumed familiarity with the paper's specialized framework.
- **Length:** no fixed word count. Use whatever length best serves the reader. Do not omit important mechanisms, experiments, or qualifications to meet a target, and do not ramble.
- **Scope:** one central idea, developed in depth, with the experiments needed to support and qualify it. Do not give every contribution equal space.
- **Format:** Markdown with a title, a one- or two-sentence subtitle, a hero figure near the top, LaTeX math, figures saved next to the article, footnotes for asides, and a references section. Use HTML or interactive figures when the destination supports them and interaction reveals something a static figure cannot.

Follow user-specified audience, length, emphasis, and platform requirements. Ask only when missing information would materially change the article; otherwise proceed with these defaults.

## Voice

Distill articles sound like knowledgeable researchers walking a colleague through an idea at a whiteboard.

- **Write as the researchers, never about the document.** When the user is an author of the paper (for example, they say "our paper" or "our experiments"), write in the authors' first-person plural: "We trained...", "We found...", "Our first attempt failed because...". Otherwise, use the inclusive "we" of writer and reader ("we can see that...") and name researchers only when attribution matters. Never use "the paper shows," "the authors' answer is," or "in this paper." Referring to the article itself ("In this article, we explore...") is fine.
- **Talk to the reader.** Address the reader directly when it helps ("If you have ever used integrated gradients, you know that..."). Ask the questions the reader is likely to have, then answer them.
- **Be conversational and precise.** Use short paragraphs and plain words. Vivid phrasing and light humor are welcome when they carry meaning ("Do neural networks hate bright colors?"). Avoid hype and generic praise.
- **Be candid about what is unknown.** Say plainly what was not tested or is not understood: "We have not investigated this," "It is unclear whether...". Label hypotheses as hypotheses: "One possibility is...". State each uncertainty once, where it matters; candor is not defensive hedging.
- **Settle terminology once.** When a concept has several names, pick one, say so briefly, and use it consistently.

## Writing Principles

- Make each paragraph's point clear early, and put intuition before formalism.
- State claims at the strength the evidence supports. Avoid vague evaluations such as "works well" unless the setting, measure, and comparison make the claim concrete.
- Keep theorem, experimental finding, illustration, interpretation, and conjecture distinguishable.
- Explain assumptions and limitations where they affect the meaning of a result.
- Compare related approaches neutrally and specifically: describe the relevant difference and why it matters.
- Cut filler, repeated summaries, inflated praise, and details that do not advance the explanation. Move tangents, proof details, and fine print into footnotes.

## Workflow

### 1. Read and Establish the Source

Read the paper's introduction, setup and method, central results, experiments, and limitations. Check the appendix when it supplies conditions, definitions, or experimental details needed to interpret a claim. Do not draft from the abstract alone. If access is incomplete, disclose the limitation and avoid claims that depend on unread sections.

Identify the title, authors, version, and stable paper link. Distinguish a preprint from a published version; do not infer acceptance from formatting. Note any public code, data, or notebooks.

Build brief internal notes containing:

- The scientific or technical question and the obstacle in existing practice.
- The central insight and the construction that implements it.
- The closest relevant alternative and the precise difference.
- Essential assumptions and the scope of the theoretical guarantees.
- Every experiment, the question it answers, its main finding, any failure or surprise, and its source location.
- Limitations that change how the reader should interpret or use the result.

Keep these notes separate from the reader-facing article. If no public paper URL is available, identify the supplied source accurately rather than inventing a link.

### 2. Find the Reader's Question and the Right Frame

State the central insight in one or two plain sentences, then choose the frame that makes it easiest to see. Distill articles often use one of these moves:

- **Show the phenomenon first.** Open with something the reader can see, such as a strange artifact, a misleading plot, or a surprising failure, then ask what causes it.
- **Break the familiar story.** Start from the standard intuition, acknowledge what it gets right, and show what it fails to explain.
- **Use the simplest faithful model.** Find a "hello world" case that is simple enough to understand completely but still shows the real phenomenon, such as two well-separated clusters, a convex quadratic, or one dimension before two. Develop the idea there, then generalize.
- **Use a running example.** Pick one concrete scenario and return to it at every step, so each new concept builds on something the reader has already seen.

Select the minimum prerequisite concepts and explain them inside the article. A link may support verification or further reading, but must not carry essential meaning.

### 3. Structure the Article as a Chain of Questions

- **Title and subtitle.** The title names the idea or the question. The subtitle states the takeaway in one or two sentences.
- **Opening.** Within the first screen, show the hero figure or puzzle and state the question. Follow with a short paragraph on what the article explores. Do not open with an abstract or a literature review.
- **Question chain.** Each section answers one question and ends by raising the next one. When an approach falls short, let that shortcoming motivate the next section, as a research story would.
- **Headings as ideas or claims.** Prefer headings like "Cluster sizes in a t-SNE plot mean nothing" or "What persists, exists" over "Method" or "Experiments." Number sections when the article is a sequence of lessons or experiments.
- **Bottom lines.** End a section with a one-sentence takeaway when it helps the reader keep track.
- **Conclusion.** Keep it short: restate the core insight, its broader implication, and the open questions. Do not recap numbers.

Integrate theory near the mechanism it explains. A conceptual or theoretical paper need not have an experimental section; an empirical paper need not be given an invented theoretical justification.

### 4. Explain the Technical Core

- Preserve the target, dependencies, and substantive assumptions while simplifying. Simplification must not silently change the claim.
- Ask "why would this make sense?" before showing a formula, then show it.
- Annotate key equations: label their parts with `\overbrace` or `\underbrace` (for example, "difference from baseline" and "accumulated gradients"). For a central formula, break it into numbered parts and show each part in a figure panel, using matching colors in equation and figure.
- Introduce notation only when it makes the explanation easier, and prefer notation that reveals structure.
- Explain a theorem's conclusion, essential conditions, and practical meaning. Sketch the proof mechanism when it illuminates the idea; put fuller detail in a footnote.
- Use analogies only when their correspondence to the technical objects is precise.
- Include short pseudocode or real code when it shows how the procedure works or how to use it. Run executable examples when possible; otherwise label them illustrative.

### 5. Tell the Experimental Story

Experiments are a core part of the article, not an appendix. Cover every experiment that materially supports, qualifies, or complicates the central explanation.

- **Frame each experiment as a question**, and explain why its setting, comparator, and metric answer that question. For a sequence of experiments, name them ("Experiment 1: Learning to grow") and let each result motivate the next.
- **Show the path, including failures.** Report instabilities, negative results, and surprises, then explain what they reveal. A failed first attempt that motivates the fix is often the most instructive part of the story.
- **Use controlled comparisons.** Vary one factor at a time and show the variants side by side, with each panel's caption stating what changed and what happened (for example, "Deconvolution in the last two layers: artifacts at frequencies 2 and 4" next to "Resize-convolution in all layers: no artifacts").
- **Include controls and sanity checks** such as random baselines, and explain what they rule out.
- **Teach the reader how to read a plot** before interpreting it ("the lower the curve, the better the method").
- **Give key setup inline**, briefly: sample sizes, iterations, the main hyperparameters. Put arbitrary choices and secondary details in footnotes, noting when a choice was not tuned.
- **Report magnitudes and uncertainty**, not just directions. Mention exceptions and mixed findings, and point out when baselines are weak or settings favor the method.
- **Keep evidence types distinct.** Simulation, real-data-calibrated simulation, observational analysis, and randomized or deployed evaluations support different conclusions. Empirical trends do not prove asymptotic guarantees, and prediction quality alone does not establish identification or causal validity.
- **Never invent measurements, error bars, or results.** Usually omit tuning grids and exhaustive baseline lists unless they matter to the question.

### 6. Design Figures as Tools for Thought

Figures carry the explanation, not decoration. Plan them alongside the outline.

- **Hero figure:** put a figure at the top that shows the phenomenon, core idea, or main result.
- **One job per figure:** each figure answers one question. Prefer small multiples to one dense figure.
- **Captions that teach:** start with a short takeaway sentence, then tell the reader what to notice ("Notice how large gradients accumulate at small values of α.").
- **Consistent visual language:** use the same color for the same concept across all figures and equations, such as the proposed method in one color and baselines in another.
- **Integrated with prose:** place each figure next to the text that uses it, and refer to it directly.
- **Generated, not screenshotted:** create figures with code, such as matplotlib or SVG, and keep the scripts with the article.
- **Honest data:** re-plot experimental results only from values reported in the source or from the authors' own result files, and note the source. Label conceptual diagrams and toy simulations as illustrations.
- **Interactive when it helps:** if the destination supports it, use sliders, toggles, or hover states when exploring a parameter reveals the mechanism, such as varying a hyperparameter across its range.

### 7. Sources, Asides, and Replicability

- Link the source paper near the beginning or end. Cite other work where a specific comparison or factual claim needs support, and describe prior approaches by what they did ("Earlier work clustered activations and optimized for each centroid").
- Use footnotes for asides, technical conditions, terminology notes, and proof details.
- Link code, data, and notebooks where available; a "Reproduce this figure" link next to key figures is ideal.
- Never mention local file paths, directory names, or repository-internal locations (e.g., `code/...`, `results/...`, `/mnt/...`, `scripts/make_figs.py`) in the article. Link only to public URLs; if none exist, describe the code or data in words or omit the reference.
- Never mention that the paper is under submission or review, or name a conference or journal it was submitted to. Cite it by title (and a public link if one exists); state a venue only if the paper has been published there.
- When the user is an author, optionally end with acknowledgments and author contributions, as Distill articles do.

### 8. Review Before Delivery

**Advancing understanding**

- Does the article offer a clear perspective or mental model, rather than restating the paper in simpler words?
- Is the scope focused, with one central idea developed well?

**Communication and style**

- Does it read as researchers explaining their work, not as commentary about a document? Search for "the paper," "this paper," "the authors," and "in this work," and rewrite them.
- Does the opening show a concrete phenomenon or question within the first screen?
- Does each section answer a question and lead to the next? Are headings ideas rather than generic labels?
- Can a reader understand the problem, central insight, essential terms, and experiments without opening any link?
- Does each figure have one job, a caption that states a takeaway, and colors consistent with the other figures and equations?
- Is every section necessary? Cut background, repeated summaries, and tangents that belong in footnotes.
- Is the article free of local paths and repository-internal file names, and of any mention of submission or review status or a target venue?

**Scientific integrity**

- Can every claim, number, comparison, and plotted value be traced to the source?
- Are the central experimental results present, interpretable, and fairly framed, including failures and exceptions?
- Are assumptions retained wherever omitting them would change the meaning of a result?
- Are uncertainties and untested questions stated candidly, once, where they matter?

## Exemplars

When web access is available, skim one or two exemplars close to the paper's type before drafting:

| Article | What to learn |
|---|---|
| [Deconvolution and Checkerboard Artifacts](https://distill.pub/2016/deconv-checkerboard/) | Opening with a visible phenomenon; controlled before/after comparisons; candid "we have not investigated this." |
| [How to Use t-SNE Effectively](https://distill.pub/2016/misread-tsne/) | Simple synthetic cases; headings stated as lessons; a bottom line per section. |
| [Why Momentum Really Works](https://distill.pub/2017/momentum/) | Breaking a familiar story; the simplest faithful model; vivid but precise prose. |
| [Growing Neural Cellular Automata](https://distill.pub/2020/growing-ca/) | Research told as a sequence of experiments, where each failure motivates the next; reproduce links. |
| [Visualizing the Impact of Feature Attribution Baselines](https://distill.pub/2020/attribution-baselines/) | Annotated equations decomposed into figure panels; honest discussion of evaluation difficulty. |
| [Exploring Bayesian Optimization](https://distill.pub/2020/bayesian-optimization/) | One running example (gold mining) carried through the whole article. |

## Deliverable

Save the article and its assets in one folder at the user's requested destination, unless they ask for inline output only:

- `index.md` (or `index.html` for an interactive article) with the complete article, references, and figure links.
- `figures/` with the rendered figures.
- `scripts/` with the code that generates each figure.

Do not modify the source paper. Report source-access limitations, unverified code, and other production notes to the user separately, outside the article. Do not include the internal notes or review checklist unless requested.
