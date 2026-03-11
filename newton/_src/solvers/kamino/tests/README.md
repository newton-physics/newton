# Running Unit Tests

## CLI

1. Enable the python environment (`conda`, `virtualenv`, `uv`), e.g. for `pyenv` + `virtualenv`:
```bash
pyenv activate newton-dev
```

```bash
cd <path-to-parent-dir>/newton/newton/_src/solvers/kamino/tests
```

```bash
python -m unittest discover -s . -p 'test_*.py'
```

## VS Code (& Cursor)

We will use the built-in unit-test discovery system of the VS Code IDE described [here](https://code.visualstudio.com/docs/python/testing).


0. Install Newton in editable mode (*can be skip if already installed like this*):
```bash
cd <path-to-parent-dir>/newton
pip install -e .[dev,data,docs]
```

1. Open `newton/newton/_src/solvers/kamino` folder in VSCode or Cursor:
```bash
cd <path-to-parent-dir>/newton/newton/_src/solvers/kamino
code .
```

3. Create a `.vscode/settings.json` in `newton/newton/_src/solvers/kamino` with contents:
```json
{
    "python.testing.unittestArgs": [
        "-v",
        "-s",
        "./tests",
        "-p",
        "test_*.py"
    ],
    "python.testing.pytestEnabled": false,
    "python.testing.unittestEnabled": true
}
```

4. Run test discovery via `ctrl/cmd + shift + P` and typing `Testing: Focus on Test Explorer View`.

5. The play buttons displayed to the right of the test hierarchy can be used to automatically launch test sets.

----