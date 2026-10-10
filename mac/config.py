"""The recipe's settings: recipe.toml defaults, local.toml overrides, environment overrides, then the guards.

Every value is validated before anything is built, downloaded or started. A profile (chat, reference) fills the
settings it owns only where nothing more specific set them, so PROFILE=reference with DRAFTS=1 serves drafts.
"""

from __future__ import annotations

import hashlib
import os
import re
import tomllib
from dataclasses import dataclass, fields
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROFILES = ("chat", "reference")
MAX_CONTEXT = 1048576                # the checkpoint's max_position_embeddings
# flags `serve_args` builds itself; EXTRA_ARGS may not set them (the parser keeps the last value); TensorFold's parser
# takes unambiguous prefixes, so any prefix of these is refused too
OWNED_FLAGS = ("--host", "--port", "--name", "--context", "--parallel", "--max-tokens", "--thinking", "--no-thinking",
               "--no-drafts", "--no-update-check", "--temperature", "--prompt-cache-gib", "--keep-warm", "--learn",
               "--learn-dir", "--learn-gib", "--compact-at", "--dashboard", "--api-key", "--vision", "--vision-image-tokens",
               "--vision-max-images")
# flags the native server has for other families or platforms; refused so a setting is not silently ignored
UNUSED_FLAGS = ("--speed-up", "--backend", "--device", "--segments", "--kv-dtype", "--tp", "--rank", "--master",
                "--master-port", "--vision-urls", "--vision-offload", "--tool-system")


class ConfigError(ValueError):
    """A setting the guards refuse; the message says which and why."""


@dataclass
class Settings:
    # [model]
    repo: str
    revision: str
    served_name: str
    same_as: str
    # [engine]
    tensorfold_repo: str
    tensorfold_sha: str
    tensorfold_src: Path
    patch: str
    patch_sha256: str
    zig: Path
    zig_cpu: str
    binary: str
    # [local]
    host: str
    port: int
    hf_cache: Path
    tf_cache: Path
    request_log: str
    # [serve]
    profile: str
    context: int
    parallel: int
    drafts: bool
    thinking: bool
    max_tokens: int
    temperature: str
    prompt_cache_gib: int
    learn: bool
    learn_dir: str
    learn_gib: int
    keep_warm: int
    compact_at: str
    dashboard: bool
    vision: bool
    vision_image_tokens: int
    vision_max_images: int
    api_key: str
    extra_args: str
    extra_env: str
    memory_limit_gb: int
    residency: bool
    memory_margin_gib: float
    memguard: bool
    memguard_min_free_gib: float
    memguard_max_swap_growth_mb: int
    memguard_max_compressor_growth_gib: float
    # [external]
    spark_recipe_repo: str
    spark_recipe_sha: str

    # ----- derived paths -------------------------------------------------------------------------------------
    def snapshot_dir(self, repo: str | None = None) -> Path:
        repo = repo or self.repo
        return self.hf_cache / "hub" / f"models--{repo.replace('/', '--')}" / "snapshots" / self.revision

    @property
    def candidate_repos(self) -> list[str]:
        return [self.repo] + [r.strip() for r in self.same_as.split(",") if r.strip()]

    @property
    def build_dir(self) -> Path:
        """The worktree at the pinned sha, patched, and the binary built from it."""
        return self.tf_cache / "engine" / self.tensorfold_sha

    @property
    def built_binary(self) -> Path:
        return self.build_dir / "zig-out" / "native" / "bin" / "tensorfold-native"

    @property
    def run_binary(self) -> Path:
        return Path(self.binary).expanduser() if self.binary else self.built_binary

    @property
    def patch_path(self) -> Path:
        return ROOT / "patches" / self.patch

    @property
    def state_dir(self) -> Path:
        return ROOT / ".run-state"

    @property
    def pid_file(self) -> Path:
        return self.state_dir / "server.pid"

    @property
    def log_file(self) -> Path:
        return self.tf_cache / "server.log"

    @property
    def memguard_log(self) -> Path:
        return self.tf_cache / "memguard.log"

    @property
    def external_dir(self) -> Path:
        return ROOT / ".external" / "spark-recipe"

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}"

    @property
    def extra_env_pairs(self) -> list[str]:
        return [kv for kv in self.extra_env.split() if kv]

    # ----- the engine's command line ------------------------------------------------------------------------
    def serve_args(self, model_dir: Path | str) -> list[str]:
        """``tensorfold-native serve <dir> ...`` for GLM-5.3-Flash on one Mac (no --speed-up: no link, no pair)."""
        args = ["serve", str(model_dir), "--name", self.served_name, "--host", self.host, "--port", str(self.port),
                "--context", str(self.context), "--parallel", str(self.parallel), "--max-tokens", str(self.max_tokens),
                "--thinking" if self.thinking else "--no-thinking", "--no-update-check"]
        if not self.drafts:
            args.append("--no-drafts")
        if self.temperature:
            args += ["--temperature", self.temperature]
        if self.prompt_cache_gib >= 0:
            args += ["--prompt-cache-gib", str(self.prompt_cache_gib)]
        args += ["--keep-warm", str(self.keep_warm)]
        if self.learn:
            args.append("--learn")
            if self.learn_dir:
                args += ["--learn-dir", str(Path(self.learn_dir).expanduser())]
            args += ["--learn-gib", str(self.learn_gib)]
        if self.compact_at:
            args += ["--compact-at", self.compact_at]
        if self.dashboard:
            args.append("--dashboard")
        if self.vision:
            args += ["--vision", "--vision-image-tokens", str(self.vision_image_tokens),
                     "--vision-max-images", str(self.vision_max_images)]
        if self.api_key:
            args += ["--api-key", self.api_key]
        return args + self.extra_args.split()

    def server_env(self) -> dict[str, str]:
        env = dict(os.environ)
        env.update({"HF_HOME": str(self.hf_cache), "HF_HUB_OFFLINE": "1", "TENSORFOLD_NO_UPDATE_CHECK": "1",
                    "TENSORFOLD_MEMORY_LIMIT_GB": str(self.memory_limit_gb)})
        if self.residency:
            env["GLM_RESIDENCY"] = "1"
            env["GLM_WIRE"] = "1"
        if self.request_log:
            env["TENSORFOLD_REQUEST_LOG"] = str(Path(self.request_log).expanduser())
        for kv in self.extra_env_pairs:
            k, v = kv.split("=", 1)
            env[k] = v
        return env

    def summary(self) -> str:
        bin_ = self.binary or f"build@{self.tensorfold_sha[:8]}+{self.patch_sha256[:8]}"
        return (f"profile={self.profile} ctx={self.context} parallel={self.parallel} drafts={int(self.drafts)} "
                f"thinking={int(self.thinking)} temp={self.temperature or 'checkpoint'} prompt_cache={self.prompt_cache_gib} "
                f"learn={int(self.learn)} vision={self.vision_image_tokens if self.vision else 0} limit={self.memory_limit_gb}GB "
                f"residency={int(self.residency)} memguard={int(self.memguard)} engine={bin_} "
                f"model={self.repo}@{self.revision[:8]} url={self.url}")


# ----- loading ---------------------------------------------------------------------------------------------------

_SECTIONS = ("model", "engine", "local", "serve", "external")
_INT = re.compile(r"^-?(0|[1-9][0-9]*)$")             # bash would read 010 as octal; refuse zero padding everywhere
_KINDS = {"str": str, "int": int, "bool": bool, "float": float, "Path": Path}


def _coerce(name: str, kind: type, raw) -> object:
    text = raw if isinstance(raw, str) else None
    if kind is bool:
        if isinstance(raw, bool):
            return raw
        if text in ("1", "true", "True"):
            return True
        if text in ("0", "false", "False"):
            return False
        raise ConfigError(f"{name}={raw!r} must be 0/1 or true/false")
    if kind is int:
        if isinstance(raw, bool) or (not isinstance(raw, int) and not (text and _INT.match(text))):
            raise ConfigError(f"{name}={raw!r} is not a decimal integer")
        return int(raw)
    if kind is float:
        try:
            return float(raw)
        except (TypeError, ValueError):
            raise ConfigError(f"{name}={raw!r} is not a number") from None
    if kind is Path:
        p = Path(str(raw)).expanduser()
        if not p.is_absolute():
            raise ConfigError(f"{name}={raw!r} must be an absolute path (or start with ~)")
        return p
    return str(raw)


def load(local_file: Path | str | None = None, env: dict[str, str] | None = None) -> Settings:
    """Settings from recipe.toml, then ``local.toml`` (or ``local_file``; "none" skips it), then the environment."""
    env = os.environ if env is None else env
    raw = tomllib.loads((ROOT / "recipe.toml").read_text())
    values: dict[str, object] = {}
    for section in _SECTIONS:
        values.update(raw.get(section, {}))
    profiles = raw.get("profiles", {})
    explicit: set[str] = set()
    local = local_file if local_file is not None else env.get("MAC_LOCAL", ROOT / "local.toml")
    if str(local) != "none" and Path(local).is_file():
        extra = tomllib.loads(Path(local).read_text())
        for section, body in extra.items():
            if not isinstance(body, dict):
                raise ConfigError(f"{local}: top-level key {section} must be a table")
            for k, v in body.items():
                values[k] = v
                explicit.add(k)
    kinds = {f.name: f.type for f in fields(Settings)}
    for name in kinds:
        if name.upper() in env:
            values[name] = env[name.upper()]
            explicit.add(name)
    unknown = sorted(k for k in values if k not in kinds)
    if unknown:
        raise ConfigError(f"unknown setting(s): {', '.join(unknown)}")
    profile = str(values.get("profile", "chat"))
    if profile not in PROFILES:
        raise ConfigError(f"profile={profile} must be one of {', '.join(PROFILES)}")
    for k, v in profiles.get(profile, {}).items():          # a profile fills what nothing more specific set
        if k in kinds and k not in explicit:
            values[k] = v
    typed = {}
    for name, kind in kinds.items():
        if name not in values:
            raise ConfigError(f"recipe.toml lacks {name}")
        real = _KINDS.get(kind if isinstance(kind, str) else kind.__name__, kind)
        typed[name] = _coerce(name, real, values[name])
    s = Settings(**typed)
    validate(s)
    return s


def validate(s: Settings) -> None:
    for name, lo, hi in (("port", 1, 65535), ("parallel", 1, 64), ("max_tokens", 1, 1 << 24), ("prompt_cache_gib", -1, 1024),
                         ("learn_gib", 1, 4096), ("keep_warm", 0, 86400), ("memory_limit_gb", 1, 4096),
                         ("memory_margin_gib", 0, 256), ("memguard_min_free_gib", 0, 256),
                         ("memguard_max_swap_growth_mb", 0, 1 << 20), ("memguard_max_compressor_growth_gib", 0, 4096),
                         ("vision_image_tokens", 16, 16384), ("vision_max_images", 1, 64)):
        v = getattr(s, name)
        if not lo <= v <= hi:
            raise ConfigError(f"{name}={v} is outside [{lo}, {hi}]")
    if not 256 <= s.context <= MAX_CONTEXT:
        raise ConfigError(f"context={s.context} must be in [256, {MAX_CONTEXT}] (the engine needs a positive window for GLM)")
    if s.context % 256:
        raise ConfigError(f"context={s.context} must be a multiple of 256")
    for name in ("revision", "tensorfold_sha", "spark_recipe_sha"):
        if not re.fullmatch(r"[0-9a-f]{40}", getattr(s, name)):
            raise ConfigError(f"{name}={getattr(s, name)} is not a 40-hex commit")
    for repo in s.candidate_repos:
        if not re.fullmatch(r"[A-Za-z0-9._-]+/[A-Za-z0-9._-]+", repo):
            raise ConfigError(f"repository {repo!r} is not org/name")
    if s.temperature and not re.fullmatch(r"[0-9]+(\.[0-9]+)?", s.temperature):
        raise ConfigError(f"temperature={s.temperature!r} must be a decimal such as 0 or 0.7, or empty for the checkpoint's")
    if not s.temperature or float(s.temperature) != 0:
        raise ConfigError(f"temperature={s.temperature!r}: the native GLM-5.3-Flash engine decodes greedily only, and refuses "
                          "every request at any other default (the checkpoint's is 1.0); keep temperature = \"0\"")
    if s.compact_at and s.compact_at != "auto" and not re.fullmatch(r"0?\.[0-9]+|1(\.0+)?", s.compact_at):
        raise ConfigError(f"compact_at={s.compact_at!r} must be empty, auto or a fraction in (0, 1]")
    if s.binary and not Path(s.binary).expanduser().is_file():
        raise ConfigError(f"binary={s.binary} is not a file")
    if not re.fullmatch(r"[A-Za-z0-9._-]+\.patch", s.patch):
        raise ConfigError(f"patch={s.patch} must be a file name under patches/")
    if not s.patch_path.is_file():
        raise ConfigError(f"patches/{s.patch} is missing")
    if not re.fullmatch(r"[0-9a-f]{64}", s.patch_sha256):
        raise ConfigError(f"patch_sha256={s.patch_sha256} is not a sha256 (recipe.toml [engine])")
    have = hashlib.sha256(s.patch_path.read_bytes()).hexdigest()
    if have != s.patch_sha256:
        raise ConfigError(f"patches/{s.patch} has sha256 {have}, not the pinned {s.patch_sha256}: never edit a patch in "
                          "place, write a new one and re-pin")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", s.zig_cpu):
        raise ConfigError(f"zig_cpu={s.zig_cpu!r} is not a Zig cpu name")
    for word in s.extra_args.split():
        flag = word.split("=", 1)[0]
        if not flag.startswith("--"):
            continue
        for owned in OWNED_FLAGS:
            if owned.startswith(flag):
                raise ConfigError(f"extra_args sets {word} (the engine reads it as {owned}), which the recipe passes "
                                  "itself; use the matching setting instead")
        for unused in UNUSED_FLAGS:
            if unused.startswith(flag):
                raise ConfigError(f"extra_args sets {word} ({unused}): not a GLM-on-one-Mac flag in this engine "
                                  "(a pair's link, CUDA ranks, remote image URLs and tool-system are other recipes)")
    for kv in s.extra_env_pairs:
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*=.*", kv):
            raise ConfigError(f"extra_env entry {kv!r} is not KEY=VALUE")
        if kv.split("=", 1)[0] == "TENSORFOLD_MEMORY_LIMIT_GB":
            raise ConfigError("extra_env sets TENSORFOLD_MEMORY_LIMIT_GB: use memory_limit_gb")
        if kv.split("=", 1)[0] == "GLM_RESIDENCY":
            raise ConfigError("extra_env sets GLM_RESIDENCY: use residency")
