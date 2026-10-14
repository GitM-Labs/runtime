"""Immutable-revision snapshots of a model-hub repository, and the diff between two.

A checkpoint's name does not identify it. ``moonshotai/Kimi-K2.5`` is whatever
``main`` points at today, and ``main`` moves. What identifies it is the commit
SHA, plus the hashes of the files at that commit. This module knows how to
resolve a ref to a SHA, list the files at a SHA, and hash them; it knows nothing
about any one model family, so the inventory, a future per-deployment pin check,
or anything else that needs "did this repository change under me" can share it.

Hashes come from two places, and the difference matters:

* **LFS files** (weight shards): the hub's tree API returns the SHA-256 of the
  content directly, so a 600 GB checkpoint is hashed without downloading it.
* **Plain git files** (``config.json``, ``*.py``): the tree API returns only the
  git blob id (SHA-1 over ``blob <len>\\0<bytes>``). These are small, so they are
  fetched and hashed with SHA-256 too, and the downloaded bytes must reproduce
  the blob id the tree reported. That binds the SHA-256 to the commit: a
  download that does not match the tree is an error, not a hash.

Nothing here imports or executes anything from a downloaded repository.
"""

from __future__ import annotations

import fnmatch
import hashlib
import json
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, replace
from typing import Iterable, Mapping, Protocol

_COMMIT_SHA = re.compile(r"^[0-9a-f]{40}$")


class SourceError(RuntimeError):
    """The hub could not answer, or answered with something that does not add up.

    ``status`` is the HTTP status when there was one, so a caller can tell a
    revision that no longer exists (404) from a hub it simply could not reach.
    """

    def __init__(self, message: str, status: int | None = None):
        super().__init__(message)
        self.status = status


def is_commit_sha(value: object) -> bool:
    """A full 40-hex commit id. A branch, a tag or an abbreviation is not one."""
    return isinstance(value, str) and bool(_COMMIT_SHA.match(value))


@dataclass(frozen=True)
class Artifact:
    """One file at one revision.

    Exactly one of ``sha256`` / ``git_oid`` is set straight from the tree API:
    ``sha256`` for an LFS file, ``git_oid`` for a plain git file.
    """

    path: str
    size: int
    sha256: str | None = None
    git_oid: str | None = None


@dataclass(frozen=True)
class Snapshot:
    repo: str
    revision: str
    artifacts: Mapping[str, Artifact]


class Source(Protocol):
    def resolve(self, repo: str, ref: str) -> str: ...
    def tree(self, repo: str, revision: str) -> list[Artifact]: ...
    def fetch(self, repo: str, revision: str, path: str) -> bytes: ...


def git_blob_oid(data: bytes) -> str:
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()  # noqa: S324 - git's own id


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class HubSource:
    """The Hugging Face Hub over plain HTTPS. Standard library only.

    ``HF_ENDPOINT`` and ``HF_TOKEN`` are honoured. The token is attached as an
    *unredirected* header, so when ``resolve/`` bounces to a CDN host the token
    does not go with it.
    """

    def __init__(self, endpoint: str | None = None, token: str | None = None, timeout: float = 60.0):
        self.endpoint = (endpoint or os.environ.get("HF_ENDPOINT") or "https://huggingface.co").rstrip("/")
        self.token = token if token is not None else os.environ.get("HF_TOKEN")
        self.timeout = timeout

    def _same_origin(self, url: str) -> bool:
        """Same scheme and host as the configured endpoint. A downgrade to http, or a
        different host, is a different origin."""
        a, b = urllib.parse.urlsplit(url), urllib.parse.urlsplit(self.endpoint)
        return (a.scheme, a.netloc) == (b.scheme, b.netloc)

    def _open(self, url: str):
        req = urllib.request.Request(url, headers={"User-Agent": "gitm-inventory"})
        # Only to the endpoint it was configured for; never to a URL the server handed us.
        if self.token and self._same_origin(url):
            req.add_unredirected_header("Authorization", f"Bearer {self.token}")
        last: Exception | None = None
        for attempt in range(3):
            try:
                return urllib.request.urlopen(req, timeout=self.timeout)  # noqa: S310 - https endpoint
            except urllib.error.HTTPError as exc:
                if exc.code < 500:
                    raise SourceError(f"{url}: HTTP {exc.code} {exc.reason}", exc.code) from exc
                last = exc
            except urllib.error.URLError as exc:
                last = exc
            time.sleep(1.5 * (attempt + 1))
        raise SourceError(f"{url}: {last}") from last

    def resolve(self, repo: str, ref: str) -> str:
        """The commit SHA ``ref`` points at right now. A full SHA is verified to exist."""
        url = f"{self.endpoint}/api/models/{repo}/revision/{urllib.parse.quote(ref, safe='')}"
        with self._open(url) as resp:
            sha = json.load(resp).get("sha")
        if not is_commit_sha(sha):
            raise SourceError(f"{repo}@{ref}: the hub returned no commit sha ({sha!r})")
        return sha

    def tree(self, repo: str, revision: str) -> list[Artifact]:
        url: str | None = f"{self.endpoint}/api/models/{repo}/tree/{revision}?recursive=true"
        out: list[Artifact] = []
        while url:
            with self._open(url) as resp:
                entries = json.load(resp)
                link = resp.headers.get("Link", "")
            for e in entries:
                if e.get("type") != "file":
                    continue
                lfs = e.get("lfs")
                if lfs:
                    out.append(Artifact(e["path"], int(lfs["size"]), sha256=lfs["oid"]))
                else:
                    out.append(Artifact(e["path"], int(e["size"]), git_oid=e["oid"]))
            m = re.search(r'<([^>]+)>;\s*rel="next"', link)
            url = urllib.parse.urljoin(url, m.group(1)) if m else None
            if url and not self._same_origin(url):
                raise SourceError(
                    f"{repo}@{revision[:12]}: the hub's next-page link leaves {self.endpoint}; refusing to follow it"
                )
        return out

    def fetch(self, repo: str, revision: str, path: str) -> bytes:
        quoted = urllib.parse.quote(path)
        with self._open(f"{self.endpoint}/{repo}/resolve/{revision}/{quoted}") as resp:
            return resp.read()


def matches(path: str, patterns: Iterable[str]) -> bool:
    return any(fnmatch.fnmatchcase(path, p) for p in patterns)


def take_snapshot(source: Source, repo: str, revision: str, include: Iterable[str]) -> Snapshot:
    """Every file at ``revision`` matching ``include``, each with a SHA-256."""
    if not is_commit_sha(revision):
        raise ValueError(f"{revision!r} is not a full commit sha; resolve the ref first")
    include = list(include)
    artifacts: dict[str, Artifact] = {}
    for art in source.tree(repo, revision):
        if not matches(art.path, include):
            continue
        if art.sha256 is None:
            data = source.fetch(repo, revision, art.path)
            if git_blob_oid(data) != art.git_oid:
                raise SourceError(
                    f"{repo}@{revision[:12]}:{art.path}: downloaded bytes do not match the "
                    "git blob id the tree reported"
                )
            art = replace(art, sha256=sha256_hex(data), size=len(data))
        artifacts[art.path] = art
    return Snapshot(repo, revision, artifacts)


def tree_view(source: Source, repo: str, revision: str, include: Iterable[str]) -> dict[str, Artifact]:
    """The matching files at ``revision`` as the tree reports them, with no downloads."""
    include = list(include)
    return {a.path: a for a in source.tree(repo, revision) if matches(a.path, include)}


def differences(recorded: Mapping[str, Mapping], current: Mapping[str, Artifact]) -> dict[str, list[str]]:
    """Paths added, removed or changed between recorded hashes and a tree.

    ``recorded`` maps path to ``{sha256, size, [git_oid]}``. An LFS file in the
    tree is compared by SHA-256; a plain file by git blob id, which the record
    keeps next to its SHA-256 for exactly this reason: a drift check should not
    have to download every config to notice one changed. A file that moved
    between LFS and plain git cannot be compared without a download, so it is
    reported as changed.
    """
    added = sorted(set(current) - set(recorded))
    removed = sorted(set(recorded) - set(current))
    changed: list[str] = []
    for path in sorted(set(recorded) & set(current)):
        rec, cur = recorded[path], current[path]
        if cur.sha256 is not None:
            same = rec.get("sha256") == cur.sha256
        else:
            same = rec.get("git_oid") is not None and rec.get("git_oid") == cur.git_oid
        if not same:
            changed.append(path)
    return {"added": added, "removed": removed, "changed": changed}
