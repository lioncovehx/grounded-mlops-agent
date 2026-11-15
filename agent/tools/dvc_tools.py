"""DVC tools: dataset versions, lineage and changes, read-only."""

import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Annotated

import yaml
from pydantic import AfterValidator, Field

from agent.registry import Args, NoArgs, ServiceUnavailable, tool

REPO = Path(os.environ.get("AGENT_REPO_DIR", Path(__file__).resolve().parents[2]))
DATASET = "data/raw/wine.csv"
DATA_TAG_PATTERN = "data-*"
REV_RE = re.compile(r"^[\w./-]+$")

# Only these (program, subcommand) pairs can ever run.
ALLOWED = {
    ("git", "show"),
    ("git", "rev-parse"),
    ("git", "tag"),
    ("git", "status"),
    ("git", "log"),
    ("dvc", "list"),
    ("dvc", "diff"),
    ("dvc", "status"),
}


def run(program: str, subcommand: str, *args: str) -> str:
    """Run an allowlisted git or dvc command in the repo, without a shell."""
    if (program, subcommand) not in ALLOWED:
        raise PermissionError(f"{program} {subcommand} is not an approved operation")
    base = ["git"] if program == "git" else [sys.executable, "-m", "dvc"]
    try:
        out = subprocess.run(
            [*base, subcommand, *args],
            cwd=REPO,
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


def pointer_at(rev: str, path: str = DATASET) -> dict:
    """The md5 and size recorded in a .dvc pointer at a Git revision."""
    out = yaml.safe_load(run("git", "show", f"{rev}:{path}.dvc"))["outs"][0]
    return {"md5": out["md5"], "size_bytes": out.get("size")}


def data_tags() -> dict[str, dict]:
    """Every data-* tag with its commit and pointer hash."""
    tags = {}
    for tag in run("git", "tag", "-l", DATA_TAG_PATTERN).splitlines():
        tags[tag] = {"commit": run("git", "rev-parse", f"{tag}^{{commit}}"), **pointer_at(tag)}
    return tags


def tag_for_md5(md5: str) -> str | None:
    return next((t for t, info in data_tags().items() if info["md5"] == md5), None)


Rev = Annotated[str, AfterValidator(check_rev)]


class RevArgs(Args):
    rev: Rev = Field(
        "HEAD", description="Git revision: a tag such as data-v1, a branch or a commit hash"
    )


class CompareArgs(Args):
    rev_a: Rev = Field(description="Older revision, for example data-v1")
    rev_b: Rev = Field("HEAD", description="Newer revision, for example data-v2 or HEAD")


@tool(NoArgs)
def list_tracked_datasets(_: NoArgs) -> dict:
    """List the datasets DVC tracks and every tagged dataset version (data-* Git tags) with its hash
    and commit."""
    tracked = json.loads(run("dvc", "list", ".", "--dvc-only", "-R", "--json") or "[]")
    return {
        "tracked_paths": [entry["path"] for entry in tracked],
        "versions": data_tags(),
        "current_pointer": pointer_at("HEAD"),
    }


@tool(RevArgs)
def get_dataset_version(args: RevArgs) -> dict:
    """Identify which dataset version a Git revision points to: its hash, size, commit and the
    data-* tag with the same hash."""
    pointer = pointer_at(args.rev)
    return {
        "rev": args.rev,
        "commit": run("git", "rev-parse", f"{args.rev}^{{commit}}"),
        "path": DATASET,
        **pointer,
        "data_tag": tag_for_md5(pointer["md5"]),
    }


@tool(CompareArgs)
def compare_dataset_versions(args: CompareArgs) -> dict:
    """Compare the dataset between two Git revisions: which tracked files were added, deleted or
    modified, with hashes and sizes."""
    diff = json.loads(run("dvc", "diff", "--json", args.rev_a, args.rev_b) or "{}")
    return {
        "rev_a": {"rev": args.rev_a, **pointer_at(args.rev_a)},
        "rev_b": {"rev": args.rev_b, **pointer_at(args.rev_b)},
        "diff": {k: v for k, v in diff.items() if v},
    }


@tool(NoArgs)
def detect_data_changes(_: NoArgs) -> dict:
    """Check whether the workspace data differs from its committed DVC pointers, and whether any
    pointer file has uncommitted changes."""
    status = json.loads(run("dvc", "status", "--json") or "{}")
    pointer_changes = run("git", "status", "--porcelain", "--", "*.dvc").splitlines()
    return {
        "dvc_status": status,
        "workspace_matches_pointers": not status,
        "uncommitted_pointer_files": pointer_changes,
        "current_data_tag": tag_for_md5(pointer_at("HEAD")["md5"]),
    }


@tool(RevArgs)
def get_reproduction_steps(args: RevArgs) -> dict:
    """Give the exact commands to restore the dataset as it was at a revision. Checks the revision
    exists; does not run anything."""
    pointer = pointer_at(args.rev)
    pointer_file = f"{DATASET}.dvc"
    return {
        "rev": args.rev,
        "commit": run("git", "rev-parse", f"{args.rev}^{{commit}}"),
        "md5": pointer["md5"],
        "commands": [
            f"git checkout {args.rev} -- {pointer_file}",
            f"dvc pull {pointer_file}",
            f"dvc checkout {pointer_file}",
        ],
        "return_to_current": f"git checkout HEAD -- {pointer_file} && dvc checkout {pointer_file}",
        "note": "dvc pull needs access to the DVC remote; skip it if the data is already cached.",
    }
