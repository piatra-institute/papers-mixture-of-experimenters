# Sources

The frozen bibliography. Each entry in the form it will take in the paper's
`## References`, with a one-line provenance note. See docs/research-pipeline.md §4.
Every in-text `et al.` must map to a co-authored entry here.

## Bibliography

- Belrose, N., et al. (2023). Eliciting latent predictions from transformers with the tuned lens. arXiv:2303.08112. — tuned lens; grounds entropy/margin readouts as label-free scorers (H1).
- Bricken, T., et al. (2023). Towards monosemanticity: decomposing language models with dictionary learning. *Transformer Circuits Thread*. — SAE features; source of the H2 substrate.
- Fedus, W., Zoph, B., & Shazeer, N. (2022). Switch Transformers: scaling to trillion parameter models with simple and efficient sparsity. *Journal of Machine Learning Research*, 23. — top-1 routing; supports the "expert is a compute shard" claim (§1).
- Geiping, J., et al. (2025). Scaling up test-time compute with latent reasoning: a recurrent depth approach. arXiv:2502.05171. — controller/test-time-compute lineage; discussion.
- Hao, S., et al. (2024). Training large language models to reason in a continuous latent space (Coconut). arXiv:2412.06769. — non-language latent substrate; §6.
- Heap, T., Lawson, T., Farnik, L., & Aitchison, L. (2025). Sparse autoencoders can interpret randomly initialized transformers. arXiv:2501.17727. — the SAE caution; bounds the H2 claim.
- Li, K., et al. (2023). Inference-time intervention: eliciting truthful answers from a language model. *NeurIPS*. — canonical runtime activation intervention; the intervention is the easy part (§2).
- Meng, K., Bau, D., Andonian, A., & Belinkov, Y. (2022). Locating and editing factual associations in GPT (ROME). *NeurIPS*. — causal tracing; the experiment-as-intervention primitive (§2).
- Park, K., Choe, Y. J., & Veitch, V. (2024). The linear representation hypothesis and the geometry of large language models. *ICML*. — concepts as directions/subspaces; grounds steering and the manifold geometry (§3, H5).
- Schick, T., et al. (2023). Toolformer: language models can teach themselves to use tools. *NeurIPS*. — action-mediated information gathering; the outer-loop ancestor (§6).
- Shazeer, N., et al. (2017). Outrageously large neural networks: the sparsely-gated mixture-of-experts layer. arXiv:1701.06538. — the MoE expert as routed compute (§1).
- Templeton, A., et al. (2024). Scaling monosemanticity: extracting interpretable features from Claude 3 Sonnet. *Transformer Circuits Thread*. — SAEs at scale; supports the substrate framing (§3).
- Turner, A. M., et al. (2023). Activation addition: steering language models without optimization. arXiv:2308.10248. — difference-of-means steering; the dense operator (§3).
- Wang, J., et al. (2024). Mixture-of-agents enhances large language model capabilities. arXiv:2406.04692. — "mixture of answerers"; distinguished in §6.
- Yao, S., et al. (2023). Tree of thoughts: deliberate problem solving with large language models. *NeurIPS*. — inference-time search in language; orchestration, not in-pass primitive (§6).
- Zhang, F., & Nanda, N. (2024). Towards best practices of activation patching in language models: metrics and methods. *ICLR*. — patching results depend on metric; supports the scorer caution (H1, §7).
- Zou, A., et al. (2023). Representation engineering: a top-down approach to AI transparency. arXiv:2310.01405. — variance and anti-steering in representation control; the generalization caution (§7).
