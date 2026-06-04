# Research

> **June 2026 literature scan (added 2026-06-04, drove the pivot).** Independent
> verification, not the originating chat, established the current state of the
> art. The chat's claim that test-time residual experimentation is novel is
> stale. Verified primary sources (first authors checked against arXiv):
>
> - [T1] Nguyen, T. et al. (2026), *ATLAS: Adaptive Test-Time Latent Steering with
>   External Verifiers*, arXiv:2601.03093. Verifier-selected test-time latent
>   steering. The closest prior art to the original framing; confirms a trained
>   verifier is needed, and shows the mechanism is not new.
> - [T1] Lee, B. W. et al. (2025), *Programming Refusal with Conditional Activation
>   Steering* (CAST), ICLR; arXiv:2409.05907. Condition-gated steering = routing.
> - [T1] Bayat, R. et al. (2025), *Steering LLM Activations in Sparse Spaces*,
>   arXiv:2503.00177. SAE-basis steering claims finer control (counterpoint to the
>   old H2).
> - [T1] Khanmohammadi, R. et al. (2025), *Calibrating LLM Confidence by Probing
>   Perturbed Representation Stability* (CCPS), arXiv:2505.21772. Single-shot
>   perturbation-stability confidence, no control: the uncontrolled baseline the
>   new paper beats.
> - [T1] Kong, L. et al. (2026), *CausalGaze*, arXiv:2604.11087; Xu, H. et al.
>   (2026), *Illusions of Confidence? Neighbourhood Consistency*, arXiv:2601.05905.
>   Counterfactual / consistency confidence, single-shot.
> - [T2] Kadavath, S. et al. (2022), *Language Models (Mostly) Know What They Know*,
>   arXiv:2207.05221. Confidence is weak on confident errors; motivates the gap.
>
> Novelty check that survived: no work runs an *answer-agnostic, control-corrected*
> internal experiment as a label-free correctness signal. The control/placebo
> element is absent from the LLM-experiment literature. This is the paper's
> contribution. The original tiered findings below (steering canon) remain valid
> background.

Findings, tiered by source proximity. See docs/research-pipeline.md §2.
T1 primary · T2 authoritative secondary · T3 reference · T4 general web (leads only).
A claim that reaches the paper rests on a T1 or T2 source.

## The framing

The starting observation comes from the originating dialogue (`chat.md`): the
word "expert" in mixture-of-experts is a fossil. A modern MoE expert is a routed
feed-forward subnetwork selected for conditional compute, not a domain
specialist and certainly not something that runs experiments. The dialogue
develops the alternative — modules that probe, perturb, and test before
contributing — and then sharpens it to its mathematical core: at inference time,
fork or perturb the residual stream, evaluate the counterfactual downstream
effect, then select or aggregate the intervention. The dialogue's own conclusion
is sober: the pieces all exist, the synthesis is plausibly new, and it has seven
concrete failure modes. This paper turns those failure modes into preregistered
measurements.

## Findings

### What an MoE expert actually is (the fossil-word claim)

- [T1] Shazeer et al. (2017), *Outrageously Large Neural Networks: The
  Sparsely-Gated Mixture-of-Experts Layer*, arXiv:1701.06538. The expert is a
  routed feed-forward subnetwork; routing is for conditional compute, to scale
  parameters without proportional FLOPs. Supports §1's claim that the expert is
  a compute shard.
- [T1] Fedus, Zoph & Shazeer (2022), *Switch Transformers*, JMLR. Top-1 routing;
  again compute routing, not epistemic specialization. Supports §1.

### Runtime activation intervention already exists (so the novelty is narrow)

- [T1] Li et al. (2023), *Inference-Time Intervention* (ITI), NeurIPS. Shifts
  activations along learned truthful directions in selected heads at inference.
  The canonical "intervene on activations at runtime" result. Supports §2 (the
  intervention is the easy part) and the dense steering operator in §3.
- [T1] Turner et al. (2023), *Activation Addition / Steering Language Models
  Without Optimization* (ActAdd), arXiv:2308.10248. Adds a steering vector from
  contrastive prompt pairs at a layer. Source of the difference-of-means
  steering construction used in the simulation.
- [T1] Zou et al. (2023), *Representation Engineering: A Top-Down Approach to AI
  Transparency*, arXiv:2310.01405. Frames reading/controlling representations;
  reports high variance and occasional anti-steering. Supports the generalization
  caution in §7.

### Activation patching is the offline ancestor of "online experimentation"

- [T1] Meng et al. (2022), *Locating and Editing Factual Associations in GPT*
  (ROME / causal tracing), NeurIPS. Causal patching of activations to find
  mediating states. Supports the "experiment = intervention + measure downstream"
  primitive in §2.
- [T2] Zhang & Nanda (2024), *Towards Best Practices of Activation Patching*,
  ICLR. Results depend heavily on metric and setup. Supports the scorer caution
  (H1) and §7's interpretation caveats.

### The substrate question: sparse features

- [T1] Bricken et al. (2023), *Towards Monosemanticity* (transformer-circuits).
  SAEs extract interpretable features from the activation basis. Source of the
  SAE substrate in H2.
- [T1] Templeton et al. (2024), *Scaling Monosemanticity* (transformer-circuits).
  SAEs scale to a production model. Supports the substrate framing.
- [T1] Heap et al. (2025), *Sparse Autoencoders Can Interpret Randomly
  Initialized Transformers*, arXiv:2501.17727. The caution: SAE interpretability
  can be partly an artifact. Supports H2's "the gap is bounded by reconstruction
  quality" hedge and the FAIL-direction reading.

### Lenses and scoring

- [T2] Belrose et al. (2023), *Eliciting Latent Predictions with the Tuned Lens*,
  arXiv:2303.08112. Decoding intermediate states into predictions; the basis for
  entropy/margin readouts as label-free scorers (H1).
- [T3] nostalgebraist (2020), the logit lens (LessWrong). Reference for the
  same idea; navigate, do not cite as primary.

### Latent reasoning (the "experiment below language" lineage)

- [T1] Hao et al. (2024), *Training Large Language Models to Reason in a
  Continuous Latent Space* (Coconut), arXiv:2412.06769. Continuous-thought tokens
  as a non-language substrate. Cited in §6 (related work / substrates).
- [T1] Geiping et al. (2025), *Scaling up Test-Time Compute with Latent Reasoning:
  A Recurrent Depth Approach*, arXiv:2502.05171. Test-time compute via recurrence;
  the controller-state idea. Cited in discussion.

### Linear representation hypothesis (why directions work)

- [T1] Park, Choe & Veitch (2024), *The Linear Representation Hypothesis and the
  Geometry of Large Language Models*, ICML. Concepts as directions/subspaces;
  inner-product and counterfactual structure matter. Grounds the steering
  directions and the manifold geometry in §3 and H5.

### Outer-loop ancestors the paper distinguishes itself from

- [T1] Schick et al. (2023), *Toolformer*, NeurIPS. Action-mediated information
  gathering, but one model learning tool use, not internal experimenters.
- [T1] Yao et al. (2023), *Tree of Thoughts*, NeurIPS. Inference-time search in
  language, an orchestration pattern, not an in-forward-pass primitive.
- [T1] Wang et al. (2024), *Mixture-of-Agents*, arXiv:2406.04692. Layered LLM
  agents; "mixture of answerers", adjacent but text-level.

## What the simulation must show (mapping findings to hypotheses)

- H1 (scorer): ITI/ActAdd show the intervention is easy; Zhang & Nanda and Zou
  show scoring/metric choice dominates. Prediction: oracle helps, proxies
  Goodhart.
- H2 (substrate): Bricken/Templeton vs Heap. Prediction: sparse more surgical,
  bounded by FVU.
- H3 (invasive vs fork): the KV-contamination intuition from the dialogue.
  Prediction: invasive helps near token, hurts horizon.
- H4 (router): MoE routing literature + the dialogue's "route for epistemic
  value". Prediction: mixture pays only on mixed failure types; else collapse.
- H5 (off-manifold): the dialogue's deepest failure mode. Prediction: drift is
  real even under a correct scorer; a manifold penalty buys it back.
