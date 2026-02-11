# Overview

Newton is a project of the Linux Foundation and aims to be governed in a transparent, accessible way for the benefit of the community. All participation in this project is open and not bound to corporate affiliation. Participants are all bound to the Linux Foundation [Code of Conduct](https://lfprojects.org/policies/code-of-conduct/).

# General Guidelines and Legal

Please refer to [the contribution guidelines](https://github.com/newton-physics/newton-governance/blob/main/CONTRIBUTING.md) in the `newton-governance` repository for general information, project membership, and legal requirements for making contributions to Newton.

# Contributing to Newton

Newton welcomes contributions from the community. In order to avoid any surprises and to increase the chance of contributions being merged, we encourage contributors to communicate their plans proactively by opening a GitHub Issue or starting a Discussion in the corresponding repository.

Please also refer to the [development guide](https://newton-physics.github.io/newton/latest/guide/development.html).

There are several ways to participate in the Newton community:

## Questions, Discussions, Suggestions

* Help answer questions or contribute to technical discussion in [GitHub Discussions](https://github.com/newton-physics/newton/discussions) and Issues.
* If you have a question, suggestion or discussion topic, start a new [GitHub Discussion](https://github.com/newton-physics/newton/discussions) if there is no existing topic.
* Once somebody shares a satisfying answer to the question, click "Mark as answer".
* [GitHub Issues](https://github.com/newton-physics/newton/issues) should only be used for bugs and features. Specifically, issues that result in a code or documentation change. We may convert issues to discussions if these conditions are not met.

## Reporting a Bug

* Check in the [GitHub Issues](https://github.com/newton-physics/newton/issues) if a report for the bug already exists.
* If the bug has not been reported yet, open a new Issue.
* Use a short, descriptive title and write a clear description of the bug.
* Document the Git hash or release version where the bug is present, and the hardware and environment by including the output of `nvidia-smi`.
* Add executable code samples or test cases with instructions for reproducing the bug.

## Documentation Issues

* Create a new issue if there is no existing report, or
* directly submit a fix following the "Fixing a Bug" workflow below.

## Fixing a Bug

* Ensure that the bug report issue has no assignee yet. If the issue is assigned and there is no linked PR, you're welcome to ask about the current status by commenting on the issue.
* Write a fix and regression unit test for the bug following the [style guide](https://newton-physics.github.io/newton/latest/guide/development.html#style-guide).
* Open a new pull request for the fix and test.
* Write a description of the bug and the fix.
* Mention related issues in the description: E.g. if the patch fixes Issue \#33, write Fixes \#33.
* Have a signed CLA on file (see [Legal Requirements](https://github.com/newton-physics/newton-governance/blob/main/CONTRIBUTING.md#legal-requirements)).
* Have the pull request approved by a [Project Member](https://github.com/newton-physics/newton-governance/blob/main/CONTRIBUTING.md#project-members) and merged into the codebase.

## Improving Performance

* Write an optimization that improves an existing or new benchmark following the [style guide](https://newton-physics.github.io/newton/latest/guide/development.html#style-guide).
* Open a new pull request with the optimization, and the benchmark, if applicable.
* Write a description of the performance optimization.
* Mention related issues in the description: E.g. if the optimization addresses Issue \#42, write Addresses \#42.
* Have a signed CLA on file (see [Legal Requirements](https://github.com/newton-physics/newton-governance/blob/main/CONTRIBUTING.md#legal-requirements)).
* Have the pull request approved by a [Project Member](https://github.com/newton-physics/newton-governance/blob/main/CONTRIBUTING.md#project-members) and merged into the codebase.

## Adding a Feature or Solver

* Discuss your proposal ideally before starting with implementation. Open a GitHub Issue or Discussion to:
  * propose and motivate the new feature or solver;
  * detail technical specifications;
  * and list changes or additions to the Newton API.
* Wait for feedback from [Project Members](https://github.com/newton-physics/newton-governance/blob/main/CONTRIBUTING.md#project-members) before proceeding.
* Implement the feature or solver following the [style guide](https://newton-physics.github.io/newton/latest/guide/development.html#style-guide).
* Add comprehensive testing and benchmarking for the new feature or solver.
* Ensure all existing tests pass and that existing benchmarks do not regress.
* Update or add documentation for the new feature or solver.
* Have a signed CLA on file (see [Legal Requirements](https://github.com/newton-physics/newton-governance/blob/main/CONTRIBUTING.md#legal-requirements)).
* Have the pull request approved by a [Project Member](https://github.com/newton-physics/newton-governance/blob/main/CONTRIBUTING.md#project-members) and merged into the codebase.

## Adding Simulation Assets

* Before proposing to add any assets to the Newton project, make sure that the assets are properly licensed for use and distribution. If you are unsure about the license, open a new discussion.
* The Newton project hosts possibly large simulation assets such as models, textures, datasets, or pre-trained policies in the [newton-assets](https://github.com/newton-physics/newton-assets) repository to keep the main newton repository small.
* Therefore, along with a pull request in the main newton repository that relies on new assets, open a corresponding pull request in the [newton-assets](https://github.com/newton-physics/newton-assets) repository.
* Follow the instructions in the [README](https://github.com/newton-physics/newton-assets) of the newton-assets repository.
* Have a signed CLA on file (see [Legal Requirements](https://github.com/newton-physics/newton-governance/blob/main/CONTRIBUTING.md#legal-requirements)).
* Have the pull request approved by a [Project Member](https://github.com/newton-physics/newton-governance/blob/main/CONTRIBUTING.md#project-members) and merged into the asset repository.
