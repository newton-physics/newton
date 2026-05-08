# ChysX AI Onboarding

This file is a quick start for future AI agents working in this repository.
Goal: make `chysx` usable quickly and avoid repeating setup mistakes.

## What ChysX Is

- `chysx` is a standalone CUDA extension package built from `ChysX/`.
- It is used by Newton through `newton._src.solvers.chysx.SolverChysX`.
- Data path is zero-copy: Warp device pointers are passed into ChysX.

## Environment Setup (From Repo Root)

Use `uv` commands first (project convention).

```powershell
# from D:\physics\newton-ChysX
uv sync --extra examples
```

Build/install ChysX wheel into current env:

```powershell
# Optional: set your GPU arch; default is 120 (RTX 5090)
$env:CHYSX_CUDA_ARCH = "120"
uv pip install --no-build-isolation ./ChysX
```

Quick import check:

```powershell
uv run python -c "import chysx; print(chysx.__version__)"
```

## Smoke Tests

Headless tests (recommended for agents):

```powershell
uv run python -m newton.examples chysx_freefall --test --viewer null
uv run python -m newton.examples chysx_hanging_cloth --test --viewer null
```

Why headless:
- avoids GUI pause/input behavior;
- deterministic for CI-like checks;
- catches solver regressions quickly.

## Current Solver Conventions (Important)

- Spring path is intentionally disabled in runtime solve pipeline.
  - Keep FEM-only in-plane model: `fem_stretch + fem_shear`.
  - Optional out-of-plane term: `bending`.
- Implicit step linearization uses previous-frame position `x_n` (not `x_tilde`).
- PCG uses warm-start:
  - initial guess comes from existing `dx` buffer;
  - first allocation/resizes zero-initialize `dx`.

If you change these, update both:
- `ChysX/src/cloth/cloth_simulator.cu`
- `ChysX/src/solver/pcg_solver.{h,cu}`

## Common Windows Pitfalls

- `uv pip install ./ChysX` fails with file lock / access denied:
  - close running Python processes first.
- Clangd may report CUDA syntax errors in `.cu/.cuh`:
  - treat as IDE diagnostics unless `nvcc` build actually fails.
- Example appears to "hang":
  - use `--test --viewer null` for non-interactive runs.

## Minimal Debug Workflow

1. Rebuild package:
   - `uv pip install --no-build-isolation ./ChysX`
2. Run both headless example tests.
3. If cloth behavior is suspicious, run:
   - `uv run python scripts/diagnose_pcg.py`
4. For performance profiling:
   - `uv run python scripts/profile_chysx.py --steps 300`

## File Map You Will Touch Most

- `ChysX/src/cloth/cloth_simulator.cu` (step pipeline, RHS/Hessian assembly)
- `ChysX/src/solver/pcg_solver.cu` (PCG iteration + graph path)
- `ChysX/src/constraint/*` (energy/gradient/Hessian kernels)
- `newton/_src/solvers/chysx/solver_chysx.py` (Python bridge/config)
- `newton/examples/chysx/example_chysx_hanging_cloth.py` (behavior checks)

## When Editing

- Keep API compatibility unless explicitly asked to break it.
- Prefer updating comments when changing numerical method choices.
- After substantial edits, always rebuild ChysX and run both headless examples.

# ChysX AI Onboarding

给后续 AI/开发者的快速说明：如何在本仓库中使用 `chysx`，以及如何正确配置和验证环境。

## 1. 你在做什么

`chysx` 是 `newton` 仓库里的一个独立 CUDA 子项目，目录在 `ChysX/`，通过 pybind11 暴露给 Python，并由 `newton/_src/solvers/chysx/solver_chysx.py` 接入 Newton。

核心事实：
- `chysx` 通过设备指针（`wp.array.ptr`）直接读写 Newton 的粒子缓冲，不做 host round-trip。
- 参数（材质、stiffness）是值拷贝；粒子数组是指针引用。
- 当前 cloth 主流程使用 FEM stretch + FEM shear + bending；`spring` 路径保留但不参与求解（避免与 FEM 重复计入面内刚度）。

## 2. 环境准备（仓库根目录执行）

先进入仓库根目录（不是 `ChysX/` 子目录）：

```powershell
cd d:\physics\newton-ChysX
```

安装/同步 Newton 开发环境（如果尚未完成）：

```powershell
uv sync --extra examples
```

设置 CUDA 架构（按显卡改；默认是 120，对应 RTX 5090）：

```powershell
# 示例：RTX 30 系列
$env:CHYSX_CUDA_ARCH = "86"
```

安装 `chysx`（推荐）：

```powershell
uv pip install --no-build-isolation ./ChysX
```

快速检查：

```powershell
uv run python -c "import chysx; print(chysx.__version__)"
```

## 3. 修改 C++/CUDA 后的标准验证

每次改了 `ChysX/src/*.cu|*.h|*.cuh`：

1) 重编译安装

```powershell
# Windows 常见 .pyd 文件锁，必要时先清理
taskkill /IM python.exe /F
uv pip install --no-build-isolation ./ChysX
```

2) 跑 smoke tests（无 viewer，避免 GUI 干扰）

```powershell
uv run python -m newton.examples chysx_freefall --test --viewer null
uv run python -m newton.examples chysx_hanging_cloth --test --viewer null
```

## 4. 新 AI 最容易踩的坑

### 4.1 忘记重装 wheel

只改了 `ChysX/src/*` 但没执行 `uv pip install ./ChysX`，Python 里加载的是旧 `.pyd`。

### 4.2 Windows 文件锁导致安装失败

如果 `uv pip install` 报访问拒绝，先 `taskkill /IM python.exe /F` 再安装。

### 4.3 用 GUI 跑测试导致“像卡住”

示例默认可能进入 viewer 逻辑。CI/验证请始终用：
- `--test --viewer null`

### 4.4 把 spring 和 FEM 同时打开

当前策略是 FEM-only（stretch + shear + bending），不要把 `spring` 重新接回求解主路径，否则会重复计入 edge stretch。

### 4.5 改了线性化点却没统一更新

当前 `ClothSimulator::step` 在线性化时使用 `x_n`（上一帧收敛位置），不是 `x_tilde`。如果改这个策略，要同步更新：
- 梯度/Hessian评估点
- RHS 组装公式
- finalize 的 `x_{n+1}` 更新形式
- 相关注释和诊断脚本

## 5. 当前求解器约定（重要）

- 线性系统：`(M/dt^2 + H_E) dx = rhs`
- `rhs` 按 `x_n` 线性化组装
- PCG 支持 warm-start：`x` 初值来自传入缓冲（不再强制清零）
- `ClothSimulator` 默认将上一帧 `dx` 作为下一帧 PCG 初值

这几个约定是稳定性和性能的关键，改动前请先验证 `freefall` 和 `hanging_cloth` 两个 case。

## 6. 快速定位关键文件

- `ChysX/src/cloth/cloth_simulator.cu`: 主 step 流程、RHS/Hessian 组装、finalize
- `ChysX/src/solver/pcg_solver.{h,cu}`: PCG + CUDA Graph + warm-start
- `ChysX/src/constraint/*`: pin / FEM stretch / FEM shear / bending
- `newton/_src/solvers/chysx/solver_chysx.py`: Python 侧桥接逻辑
- `newton/examples/chysx/*.py`: 最小可运行示例

## 7. 你可以直接复制的最小工作流

```powershell
cd d:\physics\newton-ChysX
taskkill /IM python.exe /F
uv pip install --no-build-isolation ./ChysX
uv run python -m newton.examples chysx_freefall --test --viewer null
uv run python -m newton.examples chysx_hanging_cloth --test --viewer null
```

如果这四条都过，说明本地 `chysx` 基本可用。
