<!-- SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers -->
<!-- SPDX-License-Identifier: CC-BY-4.0 -->

# Publications using Newton

A curated list of research using the [Newton physics engine](https://github.com/newton-physics/newton).
Entries must be research publications or preprints. Code and project websites
may support an entry, but do not qualify on their own.

## Inclusion criteria

Include work that uses Newton for simulation, training, evaluation, optimization,
or data generation; develops and uses a Newton component; or studies Newton itself.
Public API use qualifies, including through another framework when its Newton
backend was actually used.

A background mention or citation alone is insufficient. Papers underlying
algorithms later implemented in Newton do not qualify unless they used Newton.
Put papers using Newton solely as a comparison baseline in
[Newton used solely as a comparison baseline](#newton-used-solely-as-a-comparison-baseline).
Studies of Newton and work proposing improvements belong in the research section.

## Contributing

Add or update entries in this file using the format below. Open a PR explaining how the
work uses Newton and link material accessible to reviewers in the **PR description**.
For paywalled papers, provide an accessible author version, code, or supplementary
documentation demonstrating Newton use. These details do not need to appear in
the list. Maintainers review additions and may correct or remove entries.
Use [CITATION.cff](CITATION.cff) to cite Newton itself.

Keep one entry per work, combining preprint and published versions. Prefer the
published citation when available and retain an accessible preprint link.

## Format and sorting

- Group by the cited version's year, newest first; within each year, sort by first
  author surname, then given name, then subsequent authors, then title.
  Ignore capitalization and accents; preserve the published author order.
- Use a bullet with **title**, abbreviated authors, date, and a direct arXiv/DOI link.
  Verify metadata against the cited paper and preserve its title capitalization.

## Research using or developing Newton

### 2026

- **FlashNeRD: Performance-First Contact-Rich Neural Robot Dynamics**. *M. Ataei, M. A. Nabian, J. Xu, M. Macklin, Y. Narang, E. Heiden, O. Zhang, Y. Guo, R. D'Sa, O. Hennigh, G. State, S. Choudhry*. October 2026. [arXiv:2610.09130](https://arxiv.org/abs/2610.09130)

- **Same Action, Different Outcome: Variability in Dynamic Cloth Manipulation**. *M. Dadgostar, G. Alenyà, J. Borràs*. October 2026. [arXiv:2610.10801](https://arxiv.org/abs/2610.10801)

- **Mixed Material Point Methods for Stiff Elastoplasticity**. *G. Daviet*. ACM Transactions on Graphics 45(4), July 2026. [DOI:10.1145/3811345](https://doi.org/10.1145/3811345)

- **Backpropagating Through Simulation: Analytic Policy Gradients for Sample and Learning Efficient Differentiable Continuous Control**. *Y. Deng*. June 2026. [arXiv:2606.21525](https://arxiv.org/abs/2606.21525)

- **RigPI: Dynamic Parameter Identification of Rigid Body via VLM-Seeded Differentiable Simulation**. *X. He, R. Zhang, W. Jiang, W. Xu*. June 2026. [arXiv:2606.25212](https://arxiv.org/abs/2606.25212)

- **GM-Loco: Terrain-Adaptive Humanoid Locomotion on Granular Media**. *J. Kamohara, F. Wu, A. N. Zong, D. I. Goldman, Y. Nakka, S. Hutchinson, Y. Zhao*. September 2026. [arXiv:2609.10286](https://arxiv.org/abs/2609.10286)

- **LUMO: Designing Luminous Contact Morphology for Repeatable Whole-Finger Contact Observation**. *D. H. Kang, Y. Ko, L. Sentis*. September 2026. [arXiv:2609.18165](https://arxiv.org/abs/2609.18165)

- **Ostrich: Taking Large Strides Through Stiff Contact in Differentiable Dynamics**. *A. Kučera, K. Zimmermann*. September 2026. [arXiv:2609.08800](https://arxiv.org/abs/2609.08800)

- **PileBelief: Persistent Physical State for Interaction-Driven World Modeling**. *H. Lin, S. Zhang, H. Liu, Y. Liu, J. Zhao, X. Qu*. September 2026. [arXiv:2609.22858](https://arxiv.org/abs/2609.22858)

- **High-Fidelity 4D Cloth Capture Pipeline with a Two-Level Pattern**. *Z. Liu, A. Chen, S. Chen, Y. Yang, C. Yuksel, J. H. Lin*. ACM Transactions on Graphics 45(4), July 2026. [DOI:10.1145/3811305](https://doi.org/10.1145/3811305), [author version](https://graphics.cs.utah.edu/research/projects/two-level-cloth-capture-pattern/two-level-cloth-capture-pattern.pdf)

- **OrchardBench: A Physically-Grounded, GPU-Parallel Apple-Orchard Simulation Benchmark for Agricultural Robotics**. *H. Munn*. July 2026. [arXiv:2607.06337](https://arxiv.org/abs/2607.06337)

- **DetAug: Obstacle-Blind Trajectory Augmentation for Zero-shot Obstacle Avoidance**. *R. O'Mahoney, M. Zoellner, I. Havoutis*. September 2026. [arXiv:2609.18395](https://arxiv.org/abs/2609.18395)

- **PersistGS: Differentiable Physics for Object Permanence in 4D Gaussian Splatting**. *A. Ramlal, J. S. Zelek*. CVPR Workshops, 2026. [arXiv:2606.03479](https://arxiv.org/abs/2606.03479)

- **Informationally Decoupled Trajectory Design for Sim-to-Real System Identification**. *S. Shin, A. Anilkumar, R. Gao, J. P. Hanna*. October 2026. [arXiv:2610.10905](https://arxiv.org/abs/2610.10905)

- **A Splitting Architecture for Exact Reduced Coulomb Friction**. *H. Song, Y. Fan, U. M. Ascher, D. K. Pai*. Computer Graphics Forum 45(8), 2026. [DOI:10.1111/cgf.70576](https://doi.org/10.1111/cgf.70576), [arXiv:2607.19599](https://arxiv.org/abs/2607.19599)

- **Kamino: GPU-based Massively Parallel Simulation of Multi-Body Systems with Challenging Topologies**. *V. Tsounis, G. Maloisel, C. Schumacher, R. Grandia, A. Serifi, D. Müller, C. Amevor, T. Widmer, M. Bächer*. March 2026. [arXiv:2603.16536](https://arxiv.org/abs/2603.16536)

- **GAUGE: A Measurement-Grounded Benchmark for Physical Fidelity in Simulation Engines and Video World Models**. *S. Wang, Y. Feng, X. Jiang, S. Tian, N. Yan, X. Shen, C. Lyu, H. Wang, Y. Zhou, H. Wang, J. Pang, Y. Xiang, X. Gao, C. Shen, W. Zhang*. August 2026. [arXiv:2608.05948](https://arxiv.org/abs/2608.05948)

- **RoboFin3D: A Sim-to-Real Platform for Robotic Surface Finishing**. *H. Wen, S. Li, V. Sanjay, P. Huang, J. Li, C. Liu*. September 2026. [arXiv:2609.37560](https://arxiv.org/abs/2609.37560)

- **Learning Sim-Grounded Policies for Bimanual Rope Manipulation from Human Teleoperation Data**. *G. Wigginghaus, T. Missal, B. Guler, S. Manschitz, J. Peters*. May 2026. [arXiv:2605.16043](https://arxiv.org/abs/2605.16043)

- **Learning Expressive and Compositional Motion Representation via Spectral Skills**. *F. Wu, C. Gao, C. Yang, Y. Zhao, B. Dai, A. Wu*. September 2026. [arXiv:2609.37677](https://arxiv.org/abs/2609.37677)

- **RopeFormer: Cross-Trial Adaptation from Interaction History for Dynamic Rope Manipulation**. *M. Wu, K. Yao, S. Luan, M. Tomizuka, Y. Chen*. September 2026. [arXiv:2609.23432](https://arxiv.org/abs/2609.23432)

- **PointCast: One World Model for Rigid, Articulated, and Deformable Object Manipulation**. *H. Ye, R. Worobel, Z. Xie, M. Li, H. Yu, Y. Hong, C. Choi*. September 2026. [arXiv:2609.28393](https://arxiv.org/abs/2609.28393)

- **AlphaNIST: Emergent Assembly Skills from Binary Task Rewards**. *O. Zhang, I. Akinola, R. Scalise, M. Guaman Castro, P. Yin, T. Widmer, M. Macklin, C. Amevor, D. Turpin, D. Van Gelder, J. Carius, A. Handa, G. State, Y. Guo, Y. Narang*. October 2026. [arXiv:2610.13354](https://arxiv.org/abs/2610.13354)

- **A Balanced Data Diet: Addressing the Exploration Bottleneck in Mega-Scale RL for Robot Control**. *O. Zhang, M. Guaman Castro, P. Yin, I. Dagnino, A. Gupta, R. Scalise, B. Boots*. October 2026. [arXiv:2610.12465](https://arxiv.org/abs/2610.12465)

- **SIM1: Physics-Aligned Simulator as Zero-Shot Data Scaler in Deformable Worlds**. *Y. Zhou, H. Liu, X. Jiang, X. Shen, Y. Zhou, H. Wang, B. Fang, Y. Tian, M. Yu, Q. Yu, L. Ma, H. Li, H. Wang, J. Zeng, J. Pang*. April 2026. [arXiv:2604.08544](https://arxiv.org/abs/2604.08544)

- **SceneFactory-3D: Lifting 2D Traffic Scenes into 3D Physical Counterfactuals for Scalable Physically Grounded Safety Evaluation**. *Y. Zhu, L. Tian, T. Zhao, Y. Chen, F. Zuo, T. Li, Z. Bian*. October 2026. [arXiv:2610.02874](https://arxiv.org/abs/2610.02874)

## Newton used solely as a comparison baseline

### 2026

- **PhysLDM: Latent Diffusion for High-Fidelity Deformable Simulation**. *Y. Zhang, X. Xu, X. Pan*. October 2026. [arXiv:2610.07609](https://arxiv.org/abs/2610.07609)
