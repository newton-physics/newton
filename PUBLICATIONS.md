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

Open a PR adding a citation and BibTeX entry in the format below. Explain how the
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
- Use a bullet with **title**, abbreviated authors, date, and arXiv/DOI link.
  Keep full author names in the collapsible BibTeX block.
- Use `@article`, `@inproceedings`, or `@misc` as appropriate, with verified
  `title`, `author`, `year`, `url`, and available venue/DOI or arXiv fields.
  Use `Family, Given` names separated by `and`; protect title acronyms with braces.
- Use first-author surname and year as the key, such as `tsounis2026`, adding
  letters only for collisions (`tsounis2026a`, `tsounis2026b`). Preserve existing
  keys when adding or updating entries.
  Keep Markdown and BibTeX metadata consistent; move entries if the cited year changes.

## Research using or developing Newton

### 2026

- **Kamino: GPU-based Massively Parallel Simulation of Multi-Body Systems with Challenging Topologies**. *V. Tsounis, G. Maloisel, C. Schumacher, R. Grandia, A. Serifi, D. Müller, C. Amevor, T. Widmer, M. Bächer*. March 2026. [arXiv:2603.16536](https://arxiv.org/abs/2603.16536)

<details>
<summary>BibTeX</summary>

```bibtex
@misc{tsounis2026,
  title         = {{Kamino}: {GPU}-based Massively Parallel Simulation of Multi-Body Systems with Challenging Topologies},
  author        = {Tsounis, Vassilios and Maloisel, Guirec and Schumacher, Christian and Grandia, Ruben and Serifi, Agon and Müller, David and Amevor, Chris and Widmer, Tobias and Bächer, Moritz},
  year          = {2026},
  month         = mar,
  eprint        = {2603.16536},
  archivePrefix = {arXiv},
  primaryClass  = {cs.RO},
  url           = {https://arxiv.org/abs/2603.16536}
}
```

</details>

- **SIM1: Physics-Aligned Simulator as Zero-Shot Data Scaler in Deformable Worlds**. *Y. Zhou, H. Liu, X. Jiang, X. Shen, Y. Zhou, H. Wang, B. Fang, Y. Tian, M. Yu, Q. Yu, L. Ma, H. Li, H. Wang, J. Zeng, J. Pang*. April 2026. [arXiv:2604.08544](https://arxiv.org/abs/2604.08544)

<details>
<summary>BibTeX</summary>

```bibtex
@misc{zhou2026,
  title         = {{SIM1}: Physics-Aligned Simulator as Zero-Shot Data Scaler in Deformable Worlds},
  author        = {Zhou, Yunsong and Liu, Hangxu and Jiang, Xuekun and Shen, Xing and Zhou, Yuanzhen and Wang, Hui and Fang, Baole and Tian, Yang and Yu, Mulin and Yu, Qiaojun and Ma, Li and Li, Hengjie and Wang, Hanqing and Zeng, Jia and Pang, Jiangmiao},
  year          = {2026},
  month         = apr,
  eprint        = {2604.08544},
  archivePrefix = {arXiv},
  primaryClass  = {cs.RO},
  url           = {https://arxiv.org/abs/2604.08544}
}
```

</details>

## Newton used solely as a comparison baseline

No entries yet. Use the same format and sorting rules.
