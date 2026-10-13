"""A checkpoint inventory: what exactly is each deployment loading, and has it moved?

One YAML file holds one record per checkpoint. A record is the *facts about a
checkpoint at an immutable revision* and nothing else: the repository, the commit
SHA, the quantization recipe the checkpoint declares, the SHA-256 of every file
that defines it, and which deployments load it. How an engine runs it and how
fast it goes are other documents (the planner catalogue, benchmark results) and
are deliberately not in here.

Rules the code enforces, taken from the brief:

* ``UNVERIFIED`` is a legal value and is never replaced by a guess, a nearby
  checkpoint, or a floating branch. A ``revision`` of ``main`` is a schema error.
* A checkpoint's identity is repository + revision + hashes. ``model_type`` in a
  config is not consulted: Kimi-K2.6's config calls itself ``kimi_k25``.
* Adding a checkpoint is adding a record. Unknown keys in a record are kept and
  ignored, so there is no schema to change.

Commands (``gitm inventory ...``): ``add`` (network: pin and hash a checkpoint),
``sync`` (offline: refresh catalogue status and consumers from this repository),
``check`` (compare the file against the hub and against ``deploy/``).
"""

from __future__ import annotations

import argparse
import copy
import json
import re
import sys
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

import yaml

from gitm.planner import model_catalogue
from gitm.planner.revisions import (
    HubSource,
    Source,
    SourceError,
    differences,
    is_commit_sha,
    take_snapshot,
    tree_view,
)

UNVERIFIED = "UNVERIFIED"
SCHEMA_VERSION = 1

_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INVENTORY = Path(__file__).resolve().parent / "inventories" / "kimi.yaml"
DEPLOY_DIR = _REPO_ROOT / "deploy"

#: What defines a checkpoint: weights, their index, configs, the custom code a
#: ``trust_remote_code`` load executes, the chat template and tokenizer. Not the
#: README, figures or eval results, which change without changing the model.
DEFAULT_INCLUDE = (
    "*.safetensors",
    "*.safetensors.index.json",
    "config.json",
    "generation_config.json",
    "hf_quant_config.json",
    "preprocessor_config.json",
    "processor_config.json",
    "*.py",
    "*.jinja",
    "tokenizer*",
    "*.model",
    "special_tokens_map.json",
    "added_tokens.json",
    "docs/deploy_guidance.md",
)

_VERIFICATION = ("verified", "unverified")
_REQUIRED = (
    "id", "repository", "tracked_ref", "revision", "last_verified", "verification_status",
    "catalogue", "quantization", "artifacts", "consumers",
)


# --------------------------------------------------------------------- file io


class _Flow(dict):
    """Dumped on one line, so a 64-shard record is 64 lines and not 256."""


class _Dumper(yaml.SafeDumper):
    pass


_Dumper.add_representer(_Flow, lambda d, v: d.represent_mapping("tag:yaml.org,2002:map", v, flow_style=True))

_HEADER = """\
# Checkpoint inventory. One record per checkpoint; see gitm/planner/inventory.py.
# Maintained by `gitm inventory add|sync` — edit by hand only to change a status
# you have verified yourself, and say so in the commit.
#
#   revision  full commit sha, or UNVERIFIED. Never a branch.
#   artifacts sha256 of every file in `include`; git_oid is kept for plain git
#             files so `check` can compare without downloading them.
#   consumers every deploy/ manifest that loads this repository, and the
#             revision that manifest pins (UNVERIFIED when it pins none).
"""


def load(path: Path = DEFAULT_INVENTORY) -> dict[str, Any]:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected a mapping at the top level")
    return data


def dump(data: dict[str, Any], path: Path = DEFAULT_INVENTORY) -> None:
    out = copy.deepcopy(data)
    for rec in out.get("checkpoints", []):
        files = (rec.get("artifacts") or {}).get("files")
        if isinstance(files, dict):
            rec["artifacts"]["files"] = {k: _Flow(v) for k, v in files.items()}
        rec["consumers"] = [_Flow(c) for c in rec.get("consumers", [])]
    text = yaml.dump(out, Dumper=_Dumper, sort_keys=False, width=100_000, allow_unicode=True)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_HEADER + text, encoding="utf-8")


# ------------------------------------------------------------------ validation


def validate(data: dict[str, Any]) -> list[str]:
    """Structural problems. Says nothing about whether the hub still agrees."""
    problems: list[str] = []
    if data.get("schema_version") != SCHEMA_VERSION:
        problems.append(f"schema_version must be {SCHEMA_VERSION}, got {data.get('schema_version')!r}")
    if not isinstance(data.get("family"), str) or not data.get("family"):
        problems.append("family must be a non-empty string")
    records = data.get("checkpoints")
    if not isinstance(records, list):
        return problems + ["checkpoints must be a list"]
    seen_ids: set[str] = set()
    seen_repos: set[str] = set()
    for i, rec in enumerate(records):
        if not isinstance(rec, dict):
            problems.append(f"checkpoints[{i}] is not a mapping")
            continue
        name = rec.get("id", f"#{i}")
        for key in _REQUIRED:
            if key not in rec:
                problems.append(f"{name}: missing {key}")
        if rec.get("id") in seen_ids:
            problems.append(f"{name}: duplicate id")
        seen_ids.add(rec.get("id"))
        repo = rec.get("repository")
        if not isinstance(repo, str) or "/" not in repo:
            problems.append(f"{name}: repository must be 'owner/name', got {repo!r}")
        elif repo.lower() in seen_repos:
            problems.append(f"{name}: repository {repo} appears twice")
        else:
            seen_repos.add(repo.lower())
        rev = rec.get("revision")
        if rev != UNVERIFIED and not is_commit_sha(rev):
            problems.append(
                f"{name}: revision must be a full 40-hex commit sha or {UNVERIFIED}, got {rev!r} "
                "(a branch or tag is not an immutable revision)"
            )
        status = rec.get("verification_status")
        if status not in _VERIFICATION:
            problems.append(f"{name}: verification_status must be one of {_VERIFICATION}, got {status!r}")
        if status == "verified":
            if rev == UNVERIFIED:
                problems.append(f"{name}: verified, but revision is {UNVERIFIED}")
            if not (rec.get("artifacts") or {}).get("files"):
                problems.append(f"{name}: verified, but no artifact hashes are recorded")
        if status == "unverified" and rev != UNVERIFIED:
            problems.append(f"{name}: revision is pinned but verification_status is unverified")
        for c in rec.get("consumers") or []:
            r = c.get("revision") if isinstance(c, dict) else None
            if not isinstance(c, dict) or not isinstance(c.get("manifest"), str):
                problems.append(f"{name}: each consumer needs a manifest path")
            elif r != UNVERIFIED and not is_commit_sha(r):
                problems.append(f"{name}: consumer {c['manifest']} revision must be a commit sha or {UNVERIFIED}")
    return problems


# ------------------------------------------------------------------- catalogue


def catalogue_status(repo: str) -> dict[str, Any]:
    """Whether the planner catalogue has an entry for exactly this repository.

    Matched on the entry's declared ``name:`` in full, owner included — the same
    rule the catalogue itself uses — so ``moonshotai/Kimi-K2.6`` is *not* covered
    by the entry for ``nvidia/Kimi-K2.6-NVFP4``.
    """
    try:
        path = model_catalogue._resolve(repo)
    except FileNotFoundError:
        return {"status": "not-in-catalogue", "entry": None}
    return {"status": "covered", "entry": path.stem}


def _catalogue_entries(family: str) -> list[tuple[str, str]]:
    """``(stem, declared name)`` of catalogue entries belonging to ``family``."""
    out = []
    for path in sorted(model_catalogue.CATALOGUE_DIR.glob("*.yaml")):
        name = str((yaml.safe_load(path.read_text(encoding="utf-8")) or {}).get("name", ""))
        if family.lower() in path.stem.lower() or family.lower() in name.lower():
            out.append((path.stem, name))
    return out


# ----------------------------------------------------------------- deployments


@dataclass(frozen=True)
class DeploymentRef:
    manifest: str  # posix path relative to the repository root
    reference: str  # how the manifest names the model
    model_id: str
    revision: str  # commit sha or UNVERIFIED


_COMMENT = re.compile(r"^\s*#.*$", re.MULTILINE)
_SERVE = re.compile(r"""\b(?:vllm|sglang)\s+serve\s+(?!-)([^\s\\"']+)""")
_MODEL_FLAG = re.compile(r"""--model(?:-path)?(?:=|\s+)([^\s\\"']+)""")
_REVISION_FLAG = re.compile(r"""--revision(?:=|\s+)([^\s\\"']+)""")
_ENV_PAIR = re.compile(r"""-\s*name:\s*(\w+)\s*\n\s*value:\s*["']?([^\s"'#]+)""")
_VAR = re.compile(r"\$\((\w+)\)")


def scan_text(text: str, manifest: str) -> list[DeploymentRef]:
    """The models one manifest loads, and the revision it pins for them.

    Text-based on purpose: the serve command sits inside a shell script embedded
    in YAML, where structure ends at the string. Comments are dropped first so
    prose that says "vllm serve the model" is not read as a command.

    A ``--revision`` is attributed only when the manifest loads a single model;
    with several, which model it belongs to is a guess, so it stays UNVERIFIED.
    """
    body = _COMMENT.sub("", text)
    env = dict(_ENV_PAIR.findall(body))

    def expand(value: str) -> str:
        return _VAR.sub(lambda m: env.get(m.group(1), m.group(0)), value)

    found: dict[str, str] = {}
    for m in _SERVE.finditer(body):
        found.setdefault(expand(m.group(1)), "vllm serve")
    for m in _MODEL_FLAG.finditer(body):
        raw = m.group(1)
        label = f"--model-path (env {_VAR.search(raw).group(1)})" if _VAR.search(raw) else "--model-path"
        found.setdefault(expand(raw), label)

    revisions = {expand(m.group(1)) for m in _REVISION_FLAG.finditer(body)}
    revision = UNVERIFIED
    if len(found) == 1 and len(revisions) == 1:
        (only,) = revisions
        revision = only if is_commit_sha(only) else UNVERIFIED
    return [DeploymentRef(manifest, label, mid, revision) for mid, label in sorted(found.items())]


def scan_deployments(deploy_dir: Path = DEPLOY_DIR, repo_root: Path = _REPO_ROOT) -> list[DeploymentRef]:
    refs: list[DeploymentRef] = []
    for path in sorted(Path(deploy_dir).rglob("*.y*ml")):
        if path.suffix not in (".yaml", ".yml"):
            continue
        rel = path.resolve().relative_to(Path(repo_root).resolve()).as_posix()
        refs.extend(scan_text(path.read_text(encoding="utf-8", errors="replace"), rel))
    return refs


def _in_scope(ref: DeploymentRef, family: str) -> bool:
    return family.lower() in ref.model_id.lower() or family.lower() in Path(ref.manifest).name.lower()


def consumers_of(repo: str, refs: list[DeploymentRef]) -> list[dict[str, str]]:
    return [
        {"manifest": r.manifest, "reference": r.reference, "revision": r.revision}
        for r in refs
        if r.model_id.lower() == repo.lower()
    ]


# ---------------------------------------------------------------------- record


def _quantization(config_json: bytes | None, hf_quant_json: bytes | None) -> dict[str, Any]:
    """The recipe the checkpoint *declares*, verbatim, and where it was found.

    No interpretation: the shape differs per vendor (``quantization_config``
    nested under ``text_config`` for Moonshot, a separate ``hf_quant_config.json``
    for NVIDIA), and a normalised summary would be one more thing to drift.
    ``none-declared`` means the files were read and carry no recipe, which is a
    fact; ``UNVERIFIED`` means the files could not be read.
    """
    if config_json is None:
        return {"status": UNVERIFIED, "sources": []}
    sources: list[dict[str, Any]] = []

    def walk(node: Any, trail: tuple[str, ...]) -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                if key == "quantization_config" and isinstance(value, dict):
                    sources.append({"file": "config.json", "key": ".".join(trail + (key,)), "recipe": value})
                else:
                    walk(value, trail + (key,))

    walk(json.loads(config_json), ())
    if hf_quant_json is not None:
        sources.append({"file": "hf_quant_config.json", "key": "", "recipe": json.loads(hf_quant_json)})
    return {"status": "declared" if sources else "none-declared", "sources": sources}


def build_record(
    source: Source,
    repo: str,
    *,
    rec_id: str | None = None,
    revision: str | None = None,
    ref: str = "main",
    include: tuple[str, ...] = DEFAULT_INCLUDE,
    today: date | None = None,
) -> dict[str, Any]:
    """Pin ``repo`` to a commit and record what is in it.

    ``ref`` is resolved once, here, and only ever *remembered* as ``tracked_ref``
    so ``check`` knows which branch to watch. The recorded revision is the SHA.
    """
    sha = source.resolve(repo, revision or ref)
    snap = take_snapshot(source, repo, sha, include)
    paths = snap.artifacts

    def read(name: str) -> bytes | None:
        return source.fetch(repo, sha, name) if name in paths else None

    files = {}
    for path, art in sorted(paths.items()):
        entry = {"size": art.size, "sha256": art.sha256}
        if art.git_oid:
            entry["git_oid"] = art.git_oid
        files[path] = entry
    return {
        "id": rec_id or repo.split("/")[-1].lower(),
        "repository": repo,
        "tracked_ref": ref,
        "revision": sha,
        "last_verified": (today or date.today()).isoformat(),
        "verification_status": "verified",
        "catalogue": catalogue_status(repo),
        "quantization": _quantization(read("config.json"), read("hf_quant_config.json")),
        "artifacts": {"include": list(include), "files": files},
        "consumers": [],
    }


def sync(data: dict[str, Any], refs: list[DeploymentRef]) -> list[str]:
    """Refresh catalogue status and consumers in place. Returns what changed."""
    changed = []
    for rec in data["checkpoints"]:
        for key, new in (("catalogue", catalogue_status(rec["repository"])),
                         ("consumers", consumers_of(rec["repository"], refs))):
            if rec.get(key) != new:
                changed.append(f"{rec['id']}: {key}")
                rec[key] = new
    return changed


# ----------------------------------------------------------------------- check


@dataclass
class Finding:
    severity: str  # error | warn | info
    kind: str
    subject: str
    message: str


@dataclass
class Report:
    findings: list[Finding] = field(default_factory=list)
    unreachable: bool = False

    def add(self, severity: str, kind: str, subject: str, message: str) -> None:
        self.findings.append(Finding(severity, kind, subject, message))

    def count(self, severity: str) -> int:
        return sum(f.severity == severity for f in self.findings)

    def exit_code(self, strict: bool = False) -> int:
        if self.unreachable or any(f.kind == "schema" for f in self.findings):
            return 2
        return 1 if self.count("error") or (strict and self.count("warn")) else 0


def _summarise(paths: list[str], limit: int = 6) -> str:
    shown = ", ".join(paths[:limit])
    return shown + (f", ... (+{len(paths) - limit} more)" if len(paths) > limit else "")


def _describe(diff: dict[str, list[str]]) -> str:
    return "; ".join(f"{k} {len(v)}: {_summarise(v)}" for k, v in diff.items() if v)


def check(
    data: dict[str, Any],
    source: Source | None,
    *,
    deploy_dir: Path = DEPLOY_DIR,
    repo_root: Path = _REPO_ROOT,
    catalogue_entries: list[tuple[str, str]] | None = None,
) -> Report:
    """Compare the inventory with the hub (``source``; ``None`` skips it) and ``deploy/``.

    ``catalogue_entries`` defaults to the planner catalogue; tests pass their own.
    """
    rep = Report()
    for problem in validate(data):
        rep.add("error", "schema", "inventory", problem)
    if any(f.kind == "schema" for f in rep.findings):
        return rep

    family = data["family"]
    records = data["checkpoints"]

    # Catalogue coverage, both directions.
    for rec in records:
        want, have = catalogue_status(rec["repository"]), rec["catalogue"]
        if (have.get("status"), have.get("entry")) != (want["status"], want["entry"]):
            rep.add("error", "catalogue-status", rec["id"],
                    f"records {have.get('status')}/{have.get('entry')} but the catalogue says "
                    f"{want['status']}/{want['entry']}; run `gitm inventory sync`")
    known = {r["repository"].lower() for r in records}
    for stem, name in (_catalogue_entries(family) if catalogue_entries is None else catalogue_entries):
        if name.lower() not in known:
            rep.add("error", "catalogue-uncovered", stem,
                    f"catalogue entry {name!r} has no inventory record")

    # Deployments.
    refs = scan_deployments(deploy_dir, repo_root)
    for ref in refs:
        if _in_scope(ref, family) and ref.model_id.lower() not in known:
            rep.add("error", "unresolved-deployment", ref.manifest,
                    f"loads {ref.model_id!r}, which no record describes")
    for rec in records:
        expected = consumers_of(rec["repository"], refs)
        stored = rec["consumers"]
        stored_paths = {c["manifest"] for c in stored}
        for c in expected:
            if c not in stored:
                kind = "consumer-stale" if c["manifest"] in stored_paths else "consumer-missing"
                rep.add("error", kind, rec["id"], f"{c['manifest']} ({c['reference']}, revision {c['revision']}) "
                        "differs from the record; run `gitm inventory sync`")
        for c in stored:
            if c["manifest"] not in {e["manifest"] for e in expected}:
                rep.add("error", "consumer-stale", rec["id"], f"{c['manifest']} no longer loads this repository")
        for c in expected:
            if c["revision"] == UNVERIFIED:
                rep.add("warn", "unpinned", rec["id"],
                        f"{c['manifest']} loads {rec['repository']} with no --revision, so it gets whatever "
                        "the hub or the node's cache holds")
            elif rec["revision"] != UNVERIFIED and c["revision"] != rec["revision"]:
                rep.add("error", "pin-mismatch", rec["id"],
                        f"{c['manifest']} pins {c['revision'][:12]} but the record is {rec['revision'][:12]}")

    # The hub.
    for rec in records:
        if rec["revision"] == UNVERIFIED:
            rep.add("info", "unverified", rec["id"], "revision is UNVERIFIED; nothing to compare")
            continue
        if source is None:
            continue
        include = rec["artifacts"].get("include") or list(DEFAULT_INCLUDE)
        recorded = rec["artifacts"]["files"]
        try:
            pinned = tree_view(source, rec["repository"], rec["revision"], include)
        except SourceError as exc:
            if exc.status == 404:
                rep.add("error", "revision-drift", rec["id"],
                        f"recorded revision {rec['revision'][:12]} no longer exists on the hub")
            else:
                rep.unreachable = True
                rep.add("error", "unreachable", rec["id"], str(exc))
            continue
        diff = differences(recorded, pinned)
        if any(diff.values()):
            rep.add("error", "artifact-drift", rec["id"],
                    f"files at the recorded revision no longer match: {_describe(diff)}")
        try:
            head = source.resolve(rec["repository"], rec["tracked_ref"])
            if head != rec["revision"]:
                moved = differences(recorded, tree_view(source, rec["repository"], head, include))
                rep.add("error", "revision-drift", rec["id"],
                        f"{rec['tracked_ref']} moved {rec['revision'][:12]} -> {head[:12]}; "
                        f"{_describe(moved) or 'no tracked file differs'}")
        except SourceError as exc:
            rep.unreachable = True
            rep.add("error", "unreachable", rec["id"], str(exc))
    return rep


def _render(rep: Report) -> str:
    lines = [f"[{f.severity.upper():5}] {f.kind:22} {f.subject}: {f.message}" for f in rep.findings]
    lines.append(f"{rep.count('error')} error(s), {rep.count('warn')} warning(s), {rep.count('info')} note(s)")
    return "\n".join(lines)


# ------------------------------------------------------------------------- cli


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="gitm inventory", description=__doc__.split("\n\n")[0])
    p.add_argument("--file", type=Path, default=DEFAULT_INVENTORY, help="Inventory YAML (default: Kimi).")
    sub = p.add_subparsers(dest="action", required=True)
    add = sub.add_parser("add", help="Pin a repository to a commit, hash it, and record it (needs network).")
    add.add_argument("repository", help="owner/name on the hub.")
    add.add_argument("--id", dest="rec_id", help="Record id (default: lower-cased repo name).")
    add.add_argument("--ref", default="main", help="Ref to resolve and to watch for movement (default: main).")
    add.add_argument("--revision", help="Pin this exact commit instead of resolving --ref.")
    add.add_argument("--family", help="Family for a new inventory file.")
    sub.add_parser("sync", help="Refresh catalogue status and consumers from this repository (offline).")
    chk = sub.add_parser("check", help="Report revision, artifact, catalogue and deployment drift.")
    chk.add_argument("--offline", action="store_true", help="Skip the hub; check only the repository.")
    chk.add_argument("--strict", action="store_true", help="Treat warnings (unpinned deployments) as failures.")
    chk.add_argument("--json", action="store_true", dest="as_json")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    path: Path = args.file

    if args.action == "check":
        rep = check(load(path), None if args.offline else HubSource())
        if args.as_json:
            print(json.dumps([f.__dict__ for f in rep.findings], indent=2))
        else:
            print(_render(rep))
        return rep.exit_code(args.strict)

    if args.action == "sync":
        data = load(path)
        problems = validate(data)
        if problems:
            print("\n".join(problems), file=sys.stderr)
            return 2
        changed = sync(data, scan_deployments())
        if changed:
            dump(data, path)
        print("\n".join(changed) if changed else "already in sync")
        return 0

    # add
    source = HubSource()
    data = load(path) if path.exists() else {
        "schema_version": SCHEMA_VERSION, "family": args.family or "", "checkpoints": [],
    }
    if not data["family"]:
        print("a new inventory needs --family", file=sys.stderr)
        return 2
    existing = next((r for r in data["checkpoints"] if r["repository"].lower() == args.repository.lower()), None)
    try:
        rec = build_record(source, args.repository, rec_id=args.rec_id or (existing or {}).get("id"),
                           revision=args.revision, ref=args.ref)
    except SourceError as exc:
        print(f"could not pin {args.repository}: {exc}", file=sys.stderr)
        return 2
    if existing is None:
        data["checkpoints"].append(rec)
    else:
        data["checkpoints"][data["checkpoints"].index(existing)] = rec
    sync(data, scan_deployments())
    problems = validate(data)
    if problems:
        print("\n".join(problems), file=sys.stderr)
        return 2
    dump(data, path)
    print(f"{rec['id']}: {args.repository} @ {rec['revision']} ({len(rec['artifacts']['files'])} files hashed)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
