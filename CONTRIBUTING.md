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

## Adding a Feature

* Discuss your proposal ideally before starting with implementation. Open a GitHub Issue or Discussion to:
  * propose and motivate the new feature;
  * detail technical specifications;
  * and list changes or additions to the Newton API.
* Wait for feedback from [Project Members](https://github.com/newton-physics/newton-governance/blob/main/CONTRIBUTING.md#project-members) before proceeding.
* Implement the feature following the [style guide](https://newton-physics.github.io/newton/latest/guide/development.html#style-guide).
* Add comprehensive testing and benchmarking for the new feature.
* Ensure all existing tests pass and that existing benchmarks do not regress.
* Update or add documentation for the new feature.
* Have a signed CLA on file (see [Legal Requirements](https://github.com/newton-physics/newton-governance/blob/main/CONTRIBUTING.md#legal-requirements)).
* Have the pull request approved by a [Project Member](https://github.com/newton-physics/newton-governance/blob/main/CONTRIBUTING.md#project-members) and merged into the codebase.

## Adding a Solver or Large Extension

Newton is designed to be extended outside this repository. Most new solvers and domain-specific extensions should start as separate packages that depend on Newton, so the solver authors can own their maintenance, documentation, testing, and release cadence.

When developing an external solver or extension:

* publish it in its own repository or package with a dependency on Newton;
* use public Newton APIs only; do not import from `newton._src`;
* keep solver-specific tests, examples, and documentation with the external project;
* open a separate GitHub Issue or Discussion if you need new public extension points in Newton;
* add the `newton-physics` GitHub topic to the repository for discoverability.

If an external solver or extension becomes broadly useful and has clear long-term maintainers, Project Members may decide case-by-case to adopt part or all of it into the main Newton repository. This is conservative and not guaranteed. Contributors should not rely on upstream adoption as the expected path.

The bar for adding a solver to the main Newton repository is high because every upstream solver adds API, testing, review, release, and support responsibilities for the project. Project Members may ask that a solver remain external, or that only smaller reusable API hooks or improvements be contributed upstream.

Before investing in an upstream solver contribution, open a GitHub Issue or Discussion and wait for feedback from [Project Members](https://github.com/newton-physics/newton-governance/blob/main/CONTRIBUTING.md#project-members). Include:

* the solver's scope, algorithms, and expected users;
* why it should live in the main Newton repository instead of an external package;
* the proposed Newton API additions or changes;
* the long-term maintenance plan and owners;
* tests, benchmarks, examples, and documentation you expect to provide.

## Adding Simulation Assets and Tuned Models

Newton generally does not maintain a library of tuned or verified simulation content, such as actuator models, sensor models, robot models, calibrated material parameters, datasets, or trained policies. These assets are often tied to specific OEM products, firmware versions, calibration processes, validation datasets, or support commitments. They should be owned and maintained by the OEM, lab, or user project that can validate and support them.

Newton's role is to provide the public APIs, capabilities, tooling, workflows, examples, and tests needed for externally maintained models to work with Newton. If Newton is missing functionality that prevents you from building, tuning, validating, or distributing your own model assets, submit a pull request for the required Newton functionality, documentation, examples, or extension points rather than adding the tuned assets themselves to the main Newton repository.

When publishing externally maintained model assets:

* Host them in their own repository or package.
* Use public Newton APIs only; do not import from `newton._src`.
* Document ownership, license, supported Newton versions, what was and was not validated, and support expectations.
* Include tests or validation examples with the external project.
* Add the `newton-physics` GitHub topic to the repository for discoverability.

The Newton project may host simulation assets in the [newton-assets](https://github.com/newton-physics/newton-assets) repository when they are needed to test, document, or demonstrate Newton's core functionality. The `newton-assets` repository is not a general content library for OEM, lab, or user-maintained tuned models.

For example, a calibrated URDF for a specific robot should live in the maintainer's own repository, while a generic URDF used to test a Newton importer or document a Newton feature may belong in `newton-assets`.

Before proposing to add project-maintained assets:

* Make sure that the assets are properly licensed for use and distribution. If you are unsure about the license, open a new discussion.
* If a pull request in the main Newton repository relies on new assets, open a corresponding pull request in the [newton-assets](https://github.com/newton-physics/newton-assets) repository.
* Follow the instructions in the [README](https://github.com/newton-physics/newton-assets) of the `newton-assets` repository.
* Have a signed CLA on file (see [Legal Requirements](https://github.com/newton-physics/newton-governance/blob/main/CONTRIBUTING.md#legal-requirements)).
* Have the `newton-assets` pull request approved by a [Project Member](https://github.com/newton-physics/newton-governance/blob/main/CONTRIBUTING.md#project-members) and merged into the asset repository.
