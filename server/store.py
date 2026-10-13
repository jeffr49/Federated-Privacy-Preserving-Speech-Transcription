"""
Storage backends for the aggregation server.

The Space's own disk is EPHEMERAL (wiped on every restart/sleep on the free tier), so production
state lives on the Hugging Face Hub:

  ADAPTER_REPO      (private *model* repo)
      main branch       -> the PROMOTED adapter clients download:
                           adapter_model.safetensors, adapter_config.json, version.json
      candidate branch  -> the newest aggregated adapter, awaiting the CI evaluation gate
      tags v0, v1, ...  -> one tag per promoted version (instant rollback = re-point main)

  SUBMISSIONS_REPO  (private *dataset* repo)
      pending/v{N}/{client_id}.safetensors + .json   -> client deltas based on version N,
                                                        deleted as soon as they're aggregated
                                                        (data minimisation)

STORE_BACKEND=local uses a plain directory with the same layout, for development and tests.
"""
from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

ADAPTER_FILES = ("adapter_model.safetensors", "adapter_config.json")


class LocalStore:
    """Filesystem backend. Layout mirrors the Hub layout."""

    def __init__(self, root: str):
        self.root = Path(root)
        (self.root / "main").mkdir(parents=True, exist_ok=True)
        (self.root / "submissions").mkdir(parents=True, exist_ok=True)
        (self.root / "candidate").mkdir(parents=True, exist_ok=True)

    # ---- promoted adapter ----
    def read_version(self) -> dict:
        return json.loads((self.root / "main" / "version.json").read_text(encoding="utf-8"))

    def read_adapter_file(self, name: str) -> bytes:
        return (self.root / "main" / name).read_bytes()

    def bootstrap(self, files: Dict[str, bytes]) -> None:
        for name, data in files.items():
            (self.root / "main" / name).write_bytes(data)

    # ---- submissions ----
    def _sub_dir(self, version: int) -> Path:
        d = self.root / "submissions" / f"v{version}"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def pending_clients(self, version: int) -> List[str]:
        return sorted(p.stem for p in self._sub_dir(version).glob("*.safetensors"))

    def put_submission(self, version: int, client_id: str, delta_bytes: bytes, meta: dict) -> None:
        d = self._sub_dir(version)
        (d / f"{client_id}.safetensors").write_bytes(delta_bytes)
        (d / f"{client_id}.json").write_text(json.dumps(meta), encoding="utf-8")

    def get_submission(self, version: int, client_id: str) -> Tuple[bytes, dict]:
        d = self._sub_dir(version)
        return (d / f"{client_id}.safetensors").read_bytes(), json.loads((d / f"{client_id}.json").read_text("utf-8"))

    def delete_submissions(self, version: int, client_ids: List[str]) -> None:
        d = self._sub_dir(version)
        for cid in client_ids:
            (d / f"{cid}.safetensors").unlink(missing_ok=True)
            (d / f"{cid}.json").unlink(missing_ok=True)

    # ---- candidate ----
    def publish_candidate(self, files: Dict[str, bytes]) -> str:
        cdir = self.root / "candidate"
        if cdir.exists():
            shutil.rmtree(cdir)
        cdir.mkdir(parents=True)
        h = hashlib.sha1()
        for name, data in sorted(files.items()):
            (cdir / name).write_bytes(data)
            h.update(data)
        return h.hexdigest()


class HubStore:
    """Hugging Face Hub backend (production)."""

    def __init__(self, token: str, adapter_repo: str, submissions_repo: str):
        from huggingface_hub import HfApi  # imported lazily so LocalStore needs no hub dependency

        self.token = token
        self.adapter_repo = adapter_repo
        self.submissions_repo = submissions_repo
        self.api = HfApi(token=token)

    # ---- promoted adapter ----
    def _dl(self, repo: str, filename: str, repo_type: str, revision: str = "main") -> Path:
        from huggingface_hub import hf_hub_download

        # hf_hub_download re-checks the revision's etag on every call, so a changed file is re-fetched.
        return Path(hf_hub_download(repo_id=repo, filename=filename, repo_type=repo_type,
                                    revision=revision, token=self.token))

    def read_version(self) -> dict:
        return json.loads(self._dl(self.adapter_repo, "version.json", "model").read_text("utf-8"))

    def read_adapter_file(self, name: str) -> bytes:
        return self._dl(self.adapter_repo, name, "model").read_bytes()

    # ---- submissions ----
    def _prefix(self, version: int) -> str:
        return f"pending/v{version}/"

    def pending_clients(self, version: int) -> List[str]:
        files = self.api.list_repo_files(self.submissions_repo, repo_type="dataset")
        pre = self._prefix(version)
        return sorted(Path(f).stem for f in files if f.startswith(pre) and f.endswith(".safetensors"))

    def put_submission(self, version: int, client_id: str, delta_bytes: bytes, meta: dict) -> None:
        from huggingface_hub import CommitOperationAdd

        pre = self._prefix(version)
        self.api.create_commit(
            repo_id=self.submissions_repo, repo_type="dataset",
            operations=[
                CommitOperationAdd(path_in_repo=f"{pre}{client_id}.safetensors", path_or_fileobj=delta_bytes),
                CommitOperationAdd(path_in_repo=f"{pre}{client_id}.json", path_or_fileobj=json.dumps(meta).encode()),
            ],
            commit_message=f"submission from {client_id} (base v{version})",
        )

    def get_submission(self, version: int, client_id: str) -> Tuple[bytes, dict]:
        pre = self._prefix(version)
        blob = self._dl(self.submissions_repo, f"{pre}{client_id}.safetensors", "dataset").read_bytes()
        meta = json.loads(self._dl(self.submissions_repo, f"{pre}{client_id}.json", "dataset").read_text("utf-8"))
        return blob, meta

    def delete_submissions(self, version: int, client_ids: List[str]) -> None:
        from huggingface_hub import CommitOperationDelete

        pre = self._prefix(version)
        ops = []
        for cid in client_ids:
            ops += [CommitOperationDelete(path_in_repo=f"{pre}{cid}.safetensors"),
                    CommitOperationDelete(path_in_repo=f"{pre}{cid}.json")]
        if ops:
            self.api.create_commit(repo_id=self.submissions_repo, repo_type="dataset", operations=ops,
                                   commit_message=f"aggregated base v{version}; removed raw client deltas")

    # ---- candidate ----
    def publish_candidate(self, files: Dict[str, bytes]) -> str:
        """Commit the candidate adapter to the `candidate` branch and return the exact commit SHA,
        so CI evaluates (and later promotes) precisely this revision even if a newer one lands."""
        from huggingface_hub import CommitOperationAdd

        self.api.create_branch(repo_id=self.adapter_repo, branch="candidate", repo_type="model", exist_ok=True)
        info = self.api.create_commit(
            repo_id=self.adapter_repo, repo_type="model", revision="candidate",
            operations=[CommitOperationAdd(path_in_repo=n, path_or_fileobj=d) for n, d in files.items()],
            commit_message="candidate adapter from federated aggregation",
        )
        return info.oid


def make_store(cfg) -> "LocalStore | HubStore":
    if cfg.store_backend == "local":
        return LocalStore(cfg.local_store_dir)
    if not (cfg.hf_token and cfg.adapter_repo and cfg.submissions_repo):
        raise RuntimeError("STORE_BACKEND=hub requires HF_TOKEN, ADAPTER_REPO and SUBMISSIONS_REPO.")
    return HubStore(cfg.hf_token, cfg.adapter_repo, cfg.submissions_repo)
