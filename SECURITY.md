# Security Policy: Newton

Newton is a Linux Foundation project that is community-built and maintained.

## Supported Versions

Only Newton's most recent minor release line is actively maintained and eligible
for security fixes. Users should upgrade to the latest available minor release.

See the
[Compatibility and Support guide](https://newton-physics.github.io/newton/latest/guide/compatibility.html)
for the current platform, dependency, and release-support policy.

## Reporting a Vulnerability

If you discover a potential security vulnerability in Newton or one of its
dependencies, please **do not open a public GitHub issue, pull request, or
discussion**. Public reports may expose users before a fix is available.

Use the **Security** tab in the relevant repository within the
[newton-physics organization](https://github.com/newton-physics) and select
**Report a vulnerability**. For Newton itself, report through the
[Newton repository's Security tab](https://github.com/newton-physics/newton/security).

This confidential process delivers the report to the appropriate Newton
maintainers.

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

## Security Architecture & Context

Newton is a Linux Foundation, community-built Python physics simulation engine
and SDK for robotics and simulation research. Built on NVIDIA Warp, it can run
on CPUs and use NVIDIA GPUs for acceleration. Newton integrates MuJoCo Warp,
supports OpenUSD and other robotics asset formats, and provides optional
neural-controller and visualization integrations.

Newton operates primarily as an in-process **library and SDK**, with optional
command-line examples, browser viewers, recording tools, asset downloaders,
and CI/release automation. Its primary security responsibility is to process
simulation descriptions, meshes, textures, neural policies, recordings, and
user extensions without unexpectedly executing code, accessing unintended
network resources, corrupting application state, or exposing simulation data.

Newton does not provide an application-level authentication, authorization, or
multi-tenant isolation boundary. Applications embedding Newton are responsible
for those controls.

### Security Boundaries and Interfaces

- **Python application boundary:** Newton runs inside the caller's Python
  process with access to the caller's filesystem, network, and CPU or GPU
  compute resources, subject to the caller's operating-system privileges.
  Public APIs do not sandbox callers or user-defined extensions.
- **Asset boundary:** URDF, MJCF, OpenUSD, mesh, texture, heightfield, and
  recording inputs cross into Python and optional native parsers. Inputs may
  reference additional local or remote resources.
- **Neural-policy boundary:** ONNX, exported PyTorch, TorchScript, and legacy
  PyTorch checkpoints are loaded through different runtimes with different
  trust properties.
- **Network boundary:** Asset importers can make outbound HTTP, HTTPS, or Git
  requests. Optional Viser and Rerun viewers can create inbound listeners or
  connect to remote endpoints.
- **Compute and native-code boundary:** Warp-generated kernels can execute on
  CPUs or GPUs. CUDA drivers, when using an NVIDIA GPU, and optional native
  simulation or geometry dependencies execute with the host process's
  privileges.
- **Supply-chain boundary:** GitHub Actions build, test, publish documentation,
  create cloud GPU runners, and publish Python distributions using scoped OIDC
  permissions and protected environments.

### Threat Model

Newton's primary threat categories are:

1. **Untrusted input processing:** Simulation descriptions, meshes, textures,
   recordings, and neural policies may cross parser or runtime boundaries.
   Malformed or malicious inputs could cause unintended code execution, data
   exposure, or corruption of application state.

2. **Resource exhaustion:** Large, deeply nested, or computationally expensive
   inputs and simulations could consume excessive CPU, GPU, memory, storage, or
   network resources.

3. **Network exposure:** Asset downloaders and optional viewers may initiate
   outbound connections or create network listeners. Unsafe configuration
   could expose simulation data or allow access to unintended resources.

4. **Untrusted extensions and native code:** Controllers, callbacks, kernels,
   and native dependencies execute with the host process's privileges.
   Compromised extensions or dependencies could affect the application or host
   system.

5. **Supply-chain compromise:** Source control, CI workflows, third-party
   actions, package registries, and release automation are trust boundaries.
   Their compromise could affect published packages, build artifacts, or cloud
   resources.

### Critical Security Assumptions

- Serialized models and checkpoints come only from fully trusted sources.
  Loading a file onto a CPU does not make an unsafe serialization format safe.
- Applications accepting asset URLs enforce suitable network egress policy,
  destination allowlists, response-size limits, reference budgets, and
  timeouts. Newton's importers do not constitute a complete SSRF or
  denial-of-service defense.
- URDF, MJCF, OpenUSD, mesh, image, ONNX, and recording inputs are either
  trusted or processed in an environment where parser failure and resource
  exhaustion cannot compromise sensitive workloads.
- Browser and Rerun viewers run only on trusted networks unless an
  authenticated, encrypted proxy or equivalent access-control layer protects
  them.
- Applications that expose Newton through a service provide their own
  authentication, authorization, tenant isolation, rate limiting, and TLS.
- The host operating system, Python runtime, Warp, MuJoCo Warp, OpenUSD,
  optional native dependencies, and selected compute stack are trusted and
  kept current. The compute stack includes the CUDA driver and GPU isolation
  when using an NVIDIA GPU. Newton does not isolate failures in these
  components.
- User-defined controllers, callbacks, Warp kernels, solver extensions, and
  imported Python modules are trusted code running with the application
  process's privileges.
- Repository access controls, branch protection, GitHub environment
  protection, OIDC trust policies, and package-registry configuration prevent
  unauthorized release operations.

## Deployment and Integration Guidance

- Prefer ONNX or current exported PyTorch formats from verified sources. Do not
  load legacy `.pt` or `.pth` checkpoints received from untrusted users.
- Download and validate remote assets before simulation when possible. Enforce
  HTTPS, host allowlists, private-address rejection, maximum response sizes,
  reference-count limits, and aggregate extraction budgets.
- Run parsing of untrusted assets or recordings in a separate, resource-limited
  process or container without credentials or unrestricted network access.
- Do not assume `ViewerViser` is restricted to loopback. Newton does not expose
  a host setting, and the pinned Viser version binds its unauthenticated HTTP
  and WebSocket server to all interfaces (`0.0.0.0`) by default, even with
  `share=False` and despite the displayed `localhost` URL. On hosts with routed
  or untrusted interfaces, use a host firewall or network namespace/container
  isolation to limit access. Use an authenticated TLS proxy for intentional
  remote access, and treat public-share URLs as sensitive capabilities.
- Treat Newton as an in-process library, not as a tenant-isolation boundary.
  Applications serving mutually untrusted tenants must provide isolation
  appropriate to the inputs and extensions they permit.
- Install current Newton releases and keep Warp, OpenUSD, PyTorch, Viser,
  Rerun, image, mesh, and XML-processing dependencies updated. Keep CUDA
  drivers updated when using an NVIDIA GPU.
- Review custom Git asset sources and pin them to verified commit hashes.

## Dependency and Release Security

Newton has one required runtime dependency, `warp-lang`, and several optional
dependency groups for simulation, asset import, visualization, notebooks,
documentation, and machine-learning interoperability. Some optional packages
contain native code and therefore share the process's trust level.

The repository's `uv.lock` records resolved versions and artifact hashes for
development and CI. Published dependency constraints may resolve newer
versions for downstream users, so consumers should maintain their own tested
lockfile or equivalent reproducible environment.

Release workflows pin third-party GitHub Actions to commit hashes and use OIDC
trusted publishing for PyPI. Maintainers should continue reviewing action
updates, preserving protected release environments, minimizing workflow
permissions, and separating untrusted pull-request execution from privileged
release and cloud operations.
