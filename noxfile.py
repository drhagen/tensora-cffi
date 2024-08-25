import platform

from nox import options, parametrize
from nox_poetry import Session, session

options.sessions = ["test", "coverage", "lint"]


@session(python=["3.10", "3.11", "3.12"])
def test(s: Session):
    # Needs to be kept in sync with poetry.lock
    s.install(".", "pytest", "pytest-cov")
    s.env["COVERAGE_FILE"] = f".coverage.{s.python}.{platform.system()}"
    s.run("python", "-m", "pytest", "--cov", "tensora_cffi", "tests")


@session(venv_backend="none")
def coverage(s: Session):
    s.run("coverage", "combine")
    s.run("coverage", "html")
    s.run("coverage", "xml", "--fail-under=100")


@session(venv_backend="none")
@parametrize("command", [["ruff", "check", "."], ["ruff", "format", "--check", "."]])
def lint(s: Session, command: list[str]):
    s.run(*command)


@session(venv_backend="none")
def format(s: Session) -> None:
    s.run("ruff", "check", ".", "--select", "I", "--fix")
    s.run("ruff", "format", ".")
