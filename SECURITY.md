# Security Policy: Newton

Newton is a Linux Foundation project that is community-built and maintained.

## Scope of This Policy

The reporting policy and handling commitments that apply to every repository in
the [newton-physics organization](https://github.com/newton-physics) are defined
in the
[newton-governance security policy](https://github.com/newton-physics/newton-governance/blob/main/SECURITY.md).
This document does not replace them. Where the two overlap and this document is
the more specific one, follow this one. In particular, report a vulnerability in
one of Newton's dependencies as described under
[Vulnerabilities in dependencies](#vulnerabilities-in-dependencies) rather than
through this repository.

What this document adds is the security contract specific to Newton: which
boundaries Newton enforces, which it does not, and what an application embedding
Newton remains responsible for.

It names the risk surfaces Newton actually has and the commitments that
constrain them, and it is expected to be maintained alongside the code that
creates those surfaces. It deliberately does not enumerate inventory that would
understate the surface once it drifts, such as exact dependency counts or pinned
versions; `pyproject.toml` and `uv.lock` are authoritative for those. It also
does not track individual vulnerabilities in Newton or its dependencies. A
vulnerability that a dependency update resolves is handled through that update,
not by an entry here.

## Supported Versions

Only Newton's most recent minor release line is actively maintained and eligible
for security fixes. Users should upgrade to the latest available minor release.

See the
[Compatibility and Support guide](https://newton-physics.github.io/newton/latest/guide/compatibility.html)
for the current platform, dependency, and release-support policy.

## Reporting a Vulnerability

If you discover a potential security vulnerability in Newton, please **do not
open a public GitHub issue, pull request, or discussion**. Public reports may
expose users before a fix is available.

Use the **Security** tab of the
[Newton repository](https://github.com/newton-physics/newton/security) and
select **Report a vulnerability**. This confidential process delivers the report
to the appropriate Newton maintainers. For other repositories in the
organization, use that repository's Security tab.

Include the following information:

- Newton version, branch, or commit
- Affected component and vulnerability type
- Step-by-step reproduction instructions
- Proof-of-concept code, if available
- Required configuration or environmental preconditions
- Potential confidentiality, integrity, or availability impact
- Suggested mitigation, if known

Newton maintainers will review submitted reports promptly. For confirmed
vulnerabilities, maintainers will coordinate remediation and publish a GitHub
Security Advisory with guidance or patches as appropriate.

### Vulnerabilities in dependencies

Newton builds on projects maintained elsewhere, including Warp, MuJoCo, OpenUSD,
and PyTorch. Report a flaw in one of those projects through that project's own
private security process. Its maintainers own the fix and the advisory.

Report it to Newton as well when any of the following applies:

- Newton's use of the dependency exposes Newton users who would not otherwise be
  affected, for example through a default that Newton chooses.
- Newton pins or ships an affected version, so users need a coordinated update.
- A mitigation belongs in Newton's defaults, API, or documentation regardless of
  the upstream fix.

In those cases, state that the root cause is upstream and reference the upstream
report if one exists, so Newton maintainers can track the impact without
duplicating the upstream advisory.

## Security Architecture & Context

Newton is a Linux Foundation, community-built Python physics simulation engine
and SDK for robotics and simulation research. Built on NVIDIA Warp, it runs on
CPUs and can use NVIDIA GPUs for acceleration.

Newton operates primarily as an in-process **library and SDK**, alongside
optional command-line examples, viewers, recording tools, asset downloaders, and
release automation. Its primary security responsibility is to process simulation
inputs (scene and robot descriptions, geometry, textures, recordings, and
learned policies) without unexpectedly executing code, reaching unintended
network resources, corrupting application state, or exposing simulation data.

Newton is not a privilege, authentication, authorization, or tenant-isolation
boundary, and it does not sandbox the inputs, extensions, or dependencies it
loads. Applications embedding Newton are responsible for those controls.

**Repository Exposure Classification:** Public.
Basis: the canonical repository is public on github.com, so this document is
written for public consumption and omits exploitation detail beyond what the
published source already shows.

**Service Exposure Classification:** External / Regulated.
Basis: externally distributed open-source library and SDK, published to a public
package registry with public documentation and release automation. This tier is
a single bucket for externally distributed or regulated software; Newton falls
in it through external distribution, and no regulatory or compliance scope
applies to the project.

### Security Boundaries and Interfaces

- **Process boundary:** Newton runs inside the caller's Python process with that
  process's filesystem, network, and compute access, subject to the caller's
  operating-system privileges. Public APIs do not sandbox callers or
  user-defined extensions.
- **Asset boundary:** Scene and robot descriptions, geometry, textures, and
  recordings cross into Python and native parsers, several of which are
  third-party. These inputs can reference further local or remote resources, so
  parsing one file may read or fetch others. The accepted formats change across
  releases; the
  [documentation](https://newton-physics.github.io/newton/latest/guide/overview.html)
  describes the current set.
- **Serialized model boundary:** Newton loads serialized policies and
  checkpoints for its learned-controller integrations. Deserializing these
  artifacts can reconstruct arbitrary objects and execute code, whichever format
  is preferred at a given release.
- **Network boundary:** Asset resolution can issue outbound requests to fetch
  referenced resources. Optional viewers can open inbound network listeners or
  connect to remote endpoints.
- **Compute and native-code boundary:** Warp-generated kernels execute on CPUs
  or GPUs, and native dependencies and GPU drivers execute with the host
  process's privileges. Newton does not isolate failures originating in those
  layers.
- **Supply-chain boundary:** Source control, continuous integration,
  third-party actions, package registries, and release automation determine what
  users actually install. Three surfaces in `.github/workflows/` carry
  privileges beyond an ordinary build:
  - Tag-triggered release automation publishes to a public package index using
    a short-lived OIDC token rather than a stored registry credential, gated by
    a deployment environment that requires approval.
  - `pull_request_target` workflows run with repository context rather than
    fork context, so they can reach repository secrets and tokens. They are
    gated on the pull-request author's association with the organization.
  - Continuous integration authenticates to a cloud provider through OIDC to
    provision ephemeral self-hosted GPU runners, which then execute repository
    code on infrastructure the project controls.

### Threat Model

Newton's primary threat categories are:

1. **Unsafe deserialization of policies and checkpoints:** Loading a serialized
   policy or checkpoint from an untrusted source can execute attacker-controlled
   code in the host process at load time, before any simulation runs.

2. **Untrusted input processing:** Scene and robot descriptions, geometry,
   textures, and recordings reach Python and native parsers. Malformed or
   malicious inputs could cause memory-safety faults in native parsers,
   unintended file or network access through embedded references, data exposure,
   or corruption of application state.

3. **Network exposure through viewers:** Optional viewers create unauthenticated
   network listeners. On a host with routed or untrusted interfaces, this can
   expose simulation data and viewer controls to anyone who can reach the port.

4. **Untrusted extensions and native code:** Controllers, callbacks, kernels,
   solver extensions, and native dependencies execute with the host process's
   privileges. A compromised extension or dependency can affect the whole
   application and host.

5. **Server-side request forgery and unbounded fetching:** Asset references can
   direct outbound requests to attacker-chosen destinations, including internal
   network addresses, and can chain into further fetches.

6. **Supply-chain compromise:** A compromised third-party action, a workflow
   change that widens a token's permissions, or a weakness in the gate on
   `pull_request_target` workflows could expose repository secrets, the cloud
   credentials that provision runners, or the ability to publish a release.
   Because releases are consumed by downstream users, a compromise here reaches
   further than the repository itself.

7. **Resource exhaustion:** Large, deeply nested, or computationally expensive
   inputs and simulations can consume excessive CPU, GPU, memory, storage, or
   network resources.

### Critical Security Assumptions

- Serialized policies and checkpoints come only from fully trusted sources. No
  serialization format Newton loads is safe for untrusted input, and loading a
  file onto the CPU does not make an unsafe format safe.
- Scene and robot descriptions, geometry, textures, and recordings are either
  trusted or processed in an environment where parser failure and resource
  exhaustion cannot compromise sensitive workloads.
- Applications that accept asset URLs enforce their own network egress policy,
  destination allowlists, response-size limits, reference budgets, and timeouts.
  Newton's asset resolution is not a complete defense against server-side
  request forgery or denial of service.
- Viewers run only on trusted networks unless an authenticated, encrypted proxy
  or equivalent access-control layer protects them.
- Applications that expose Newton through a service provide their own
  authentication, authorization, tenant isolation, rate limiting, and TLS.
- The host operating system, Python runtime, Warp, solver backends, asset and
  image libraries, other native dependencies, and the selected compute stack
  (including GPU drivers) are trusted and kept current. Newton does not isolate
  failures in these components.
- User-defined controllers, callbacks, Warp kernels, solver extensions, and
  imported Python modules are trusted code running with the application
  process's privileges.
- Repository access controls, branch protection, the deployment-environment
  approval on publishing, and the author-association gate on
  `pull_request_target` workflows prevent unauthorized release operations and
  keep untrusted pull-request code away from privileged credentials.

## Deployment and Integration Guidance

- **Treat every serialized policy and checkpoint as executable code.** Newton's
  PyTorch integration loads exported programs (`.pt2`), TorchScript modules, and
  pickle-based checkpoints (`.pt`, `.pth`). PyTorch documents
  `torch.export.load()` as pickle-based and warns against loading data from an
  untrusted source, and the same holds for `torch.jit.load()` and
  `torch.load(..., weights_only=False)`.
  Newton's preference for newer formats is about compatibility and maintenance,
  not security. Do not load any of these formats from an untrusted source.
- **Do not assume the browser viewer is restricted to loopback.**
  `ViewerViser` exposes no host or bind-address setting, so the underlying Viser
  server's default applies, and Viser binds its unauthenticated HTTP and
  WebSocket server to all interfaces. This holds with `share=False` and despite
  the `localhost` URL Newton prints. On hosts with routed or untrusted
  interfaces, restrict access with a host firewall or container or
  network-namespace isolation, put an authenticated TLS proxy in front of any
  intentional remote access, and treat public share URLs as sensitive
  capabilities.
- **Constrain remote asset resolution.** Download and validate remote assets
  before simulation where practical, and enforce HTTPS, host allowlists,
  rejection of private address ranges, maximum response sizes, reference-count
  limits, and aggregate extraction budgets in the surrounding application.
- **Isolate untrusted parsing.** Parse untrusted assets or recordings in a
  separate, resource-limited process or container that holds no credentials and
  has no unrestricted network access.
- **Do not use Newton as a tenant boundary.** Applications serving mutually
  untrusted tenants must provide isolation appropriate to the inputs and
  extensions they permit.
- **Pin and review external asset sources.** Pin custom Git asset sources to
  verified commit hashes and review them as you would a dependency.
- **Keep the stack current.** Install current Newton releases and keep Warp,
  solver backends, asset and image libraries, viewers, and GPU drivers updated.

## Dependencies and Releases

`pyproject.toml` and `uv.lock` are the authoritative record of Newton's required
and optional dependencies, including the versions and artifact hashes used for
development and CI. Optional dependencies enlarge the attack surface: several
carry native code and, once installed, run at the host process's trust level.

Newton's published dependency constraints can resolve to newer versions for
downstream users than its own lockfile pins, so consumers should maintain a
tested lockfile or equivalent reproducible environment of their own.

### Release automation

The workflows in `.github/workflows/` define how Newton is built, tested, and
published, and they are the code that produces what users install. Maintainers
hold the following commitments for them:

- Pin third-party actions to a commit hash, not a tag or branch.
- Grant each job the narrowest token permissions it needs, and no `write`
  permission a job does not use.
- Publish only through a deployment environment that requires approval, using
  short-lived OIDC credentials rather than stored registry tokens.
- Keep untrusted pull-request code away from privileged credentials. Workflows
  that run with repository context or reach cloud credentials stay gated on the
  author's association with the organization.
- Treat a change that widens a workflow's permissions, weakens one of these
  gates, or adds a new privileged trigger as a security-relevant change, and
  update this document when the surface it describes changes.

These are obligations, not an audit result. A workflow that no longer satisfies
one of them is a defect to fix or a statement here to correct.
