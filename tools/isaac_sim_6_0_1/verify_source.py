"""Verify that Isaac Sim resolves Newton from this source checkout."""

import argparse
from pathlib import Path


def validate_source(module_file: str, expected_repo: str) -> Path:
    """Validate that a module is contained by the expected repository."""
    module_path = Path(module_file).resolve()
    repo_path = Path(expected_repo).resolve()
    if not module_path.is_relative_to(repo_path):
        raise RuntimeError(f"Newton source {module_path} is outside expected repository {repo_path}")
    return repo_path


def verify_runtime(expected_repo: str) -> tuple[Path, str, str]:
    """Import Newton and Warp and return their resolved runtime details."""
    import newton
    import warp as wp

    repo_path = validate_source(newton.__file__, expected_repo)
    return repo_path, newton.__version__, wp.__version__


def main() -> None:
    """Run the development import preflight."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--expected-repo", required=True)
    args = parser.parse_args()
    repo, newton_version, warp_version = verify_runtime(args.expected_repo)
    print(f"NEWTON_DEV source={repo}")
    print(f"NEWTON_DEV newton_version={newton_version}")
    print(f"NEWTON_DEV warp_version={warp_version}")


if __name__ == "__main__":
    main()
