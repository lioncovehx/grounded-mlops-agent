"""DVC tools: dataset versions, lineage and changes, read-only.

Works on any Git + DVC repo: datasets are discovered from the tracked .dvc pointer files, and
dataset versions are the Git tags matching the configured pattern (data-* by default).
"""

import json
import re
import subprocess
import sys
from typing import Annotated

import yaml
from pydantic import AfterValidator, Field

from agent.config import settings
from agent.registry import Args, NoArgs, ServiceUnavailable, tool

REV_RE = re.compile(r"^[\w./-]+$")

# Only these (program, subcommand) pairs can ever run.
ALLOWED = {
    ("git", "show"),
    ("git", "rev-parse"),
    ("git", "tag"),
    ("git", "status"),
    ("git", "log"),
    ("git", "ls-files"),
    ("dvc", "diff"),
    ("dvc", "status"),
}


def repo_dir():
    return settings().dvc.repo_dir


def run(program: str, subcommand: str, *args: str) -> str:
    """Run an allowlisted git or dvc command in the repo, without a shell."""
    if (program, subcommand) not in ALLOWED:
        raise PermissionError(f"{program} {subcommand} is not an approved operation")
    base = ["git"] if program == "git" else [sys.executable, "-m", "dvc"]
    try:
        out = subprocess.run(
            [*base, subcommand, *args],
            cwd=repo_dir(),
            check=False,  # the return code is checked below
            shell=False,
            timeout=30,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as exc:
        raise ServiceUnavailable(f"{program} is not installed") from exc
    if out.returncode != 0:
        raise RuntimeError(f"{program} {subcommand} failed: {out.stderr.strip()[:300]}")
    return out.stdout.strip()


def check_rev(rev: str) -> str:
    if rev.startswith("-") or not REV_RE.match(rev):
        raise ValueError("revision must be a plain Git name such as data-v1, main or a commit hash")
    return rev


def tracked_datasets() -> list[str]:
    """Paths of every dataset with a committed .dvc pointer."""
    pointers = run("git", "ls-files", "--", "*.dvc").splitlines()
    return [p.removesuffix(".dvc") for p in pointers if p.endswith(".dvc")]


def resolve_dataset(path: str | None) -> str:
    """The dataset a tool works on: the given path, the configured default, or the only one."""
    tracked = tracked_datasets()
    path = path or settings().dvc.default_dataset
    if path is None and len(tracked) == 1:
        path = tracked[0]
    if path is None:
        raise ValueError(f"several datasets are tracked; pass one of {tracked}")
    if path not in tracked:
        raise ValueError(f"{path!r} is not a DVC-tracked dataset; tracked: {tracked}")
    return path


def pointer_at(rev: str, path: str) -> dict:
    """The md5 and size recorded in a .dvc pointer at a Git revision."""
    out = yaml.safe_load(run("git", "show", f"{rev}:{path}.dvc"))["outs"][0]
    return {"md5": out["md5"], "size_bytes": out.get("size")}


def data_tags(path: str) -> dict[str, dict]:
    """Every version tag that contains this dataset's pointer, with its commit and hash."""
    tags = {}
    for tag in run("git", "tag", "-l", settings().dvc.version_tag_pattern).splitlines():
        try:
            pointer = pointer_at(tag, path)
        except RuntimeError:
            continue  # this tag predates the dataset
        tags[tag] = {"commit": run("git", "rev-parse", f"{tag}^{{commit}}"), **pointer}
    return tags


def tag_for_md5(md5: str, path: str) -> str | None:
    return next((t for t, info in data_tags(path).items() if info["md5"] == md5), None)


Rev = Annotated[str, AfterValidator(check_rev)]
DatasetPath = Field(None, description="DVC-tracked dataset path; omit for the project default")


class DatasetArgs(Args):
    path: str | None = DatasetPath


class RevArgs(DatasetArgs):
    rev: Rev = Field(
        "HEAD", description="Git revision: a version tag such as data-v1, a branch or a commit"
    )


class CompareArgs(DatasetArgs):
    rev_a: Rev = Field(description="Older revision, for example data-v1")
    rev_b: Rev = Field("HEAD", description="Newer revision, for example data-v2 or HEAD")


@tool(DatasetArgs)
def list_tracked_datasets(args: DatasetArgs) -> dict:
    """List every DVC-tracked dataset, and the tagged versions of one dataset (default: the
    project's main dataset) with hash and commit."""
    path = resolve_dataset(args.path)
    return {
        "tracked_datasets": tracked_datasets(),
        "dataset": path,
        "version_tag_pattern": settings().dvc.version_tag_pattern,
        "versions": data_tags(path),
        "current_pointer": pointer_at("HEAD", path),
    }


@tool(RevArgs)
def get_dataset_version(args: RevArgs) -> dict:
    """Identify which dataset version a Git revision points to: its hash, size, commit and the
    version tag with the same hash."""
    path = resolve_dataset(args.path)
    pointer = pointer_at(args.rev, path)
    return {
        "rev": args.rev,
        "commit": run("git", "rev-parse", f"{args.rev}^{{commit}}"),
        "path": path,
        **pointer,
        "version_tag": tag_for_md5(pointer["md5"], path),
    }


@tool(CompareArgs)
def compare_dataset_versions(args: CompareArgs) -> dict:
    """Compare data between two Git revisions: which tracked files were added, deleted or
    modified, and the dataset's hash and size at each revision."""
    path = resolve_dataset(args.path)
    diff = json.loads(run("dvc", "diff", "--json", args.rev_a, args.rev_b) or "{}")
    return {
        "dataset": path,
        "rev_a": {"rev": args.rev_a, **pointer_at(args.rev_a, path)},
        "rev_b": {"rev": args.rev_b, **pointer_at(args.rev_b, path)},
        "diff": {k: v for k, v in diff.items() if v},
    }


@tool(NoArgs)
def detect_data_changes(_: NoArgs) -> dict:
    """Check whether workspace data differs from its committed DVC pointers, whether any pointer
    file has uncommitted changes, and which version tag each dataset is at."""
    status = json.loads(run("dvc", "status", "--json") or "{}")
    pointer_changes = run("git", "status", "--porcelain", "--", "*.dvc").splitlines()
    return {
        "dvc_status": status,
        "workspace_matches_pointers": not status,
        "uncommitted_pointer_files": pointer_changes,
        "current_version_tags": {
            p: tag_for_md5(pointer_at("HEAD", p)["md5"], p) for p in tracked_datasets()
        },
    }


@tool(RevArgs)
def get_reproduction_steps(args: RevArgs) -> dict:
    """Give the exact commands to restore a dataset as it was at a revision. Checks the revision
    exists; does not run anything."""
    path = resolve_dataset(args.path)
    pointer = pointer_at(args.rev, path)
    pointer_file = f"{path}.dvc"
    return {
        "rev": args.rev,
        "commit": run("git", "rev-parse", f"{args.rev}^{{commit}}"),
        "path": path,
        "md5": pointer["md5"],
        "commands": [
            f"git checkout {args.rev} -- {pointer_file}",
            f"dvc pull {pointer_file}",
            f"dvc checkout {pointer_file}",
        ],
        "return_to_current": f"git checkout HEAD -- {pointer_file} && dvc checkout {pointer_file}",
        "note": "dvc pull needs access to the DVC remote; skip it if the data is already cached.",
    }
