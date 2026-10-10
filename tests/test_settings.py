"""The guards: python3 -m unittest discover tests. Stdlib only; nothing here touches the engine or the network."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mac import config  # noqa: E402


def load(**env):
    return config.load(local_file="none", env={k.upper(): str(v) for k, v in env.items()})


class Defaults(unittest.TestCase):
    def test_defaults_load_and_serve_args(self):
        s = load()
        self.assertEqual(s.profile, "chat")
        self.assertTrue(s.drafts)
        self.assertEqual(s.context, 262144)
        self.assertEqual(s.parallel, 1)
        self.assertEqual(s.url, "http://127.0.0.1:8888")  # not 8080: everything else on this Mac uses it
        args = s.serve_args("/m")
        self.assertEqual(args[:2], ["serve", "/m"])
        for flag in ("--thinking", "--dashboard", "--no-update-check", "--parallel", "--context"):
            self.assertIn(flag, args)
        for flag in ("--no-drafts", "--learn", "--speed-up"):
            self.assertNotIn(flag, args)
        self.assertEqual(args[args.index("--prompt-cache-gib") + 1], "4")
        self.assertEqual(args[args.index("--temperature") + 1], "0")        # the native GLM engine is greedy-only
        self.assertEqual(args[args.index("--reasoning-effort") + 1], "high")
        self.assertEqual(load(reasoning_effort="max").serve_args("/m")[-1:], load(reasoning_effort="max").serve_args("/m")[-1:])
        self.assertIn("xhigh", load(reasoning_effort="max").serve_args("/m"))
        self.assertNotIn("--reasoning-effort", load(reasoning_effort="").serve_args("/m"))
        self.assertEqual(s.server_env()["GLM_RESIDENCY"], "1")
        self.assertEqual(s.server_env()["GLM_WIRE"], "1")
        self.assertEqual(args[args.index("--vision-image-tokens") + 1], "4096")
        self.assertIn("--vision", args)
        self.assertNotIn("--vision", load(vision="0").serve_args("/m"))
        self.assertNotIn("GLM_RESIDENCY", load(residency="0").server_env())   # bounded, so the load stays predictable

    def test_long_profile_is_the_full_window(self):
        s = load(profile="long")
        self.assertEqual((s.context, s.memory_limit_gb), (1048576, 215))
        self.assertTrue(s.drafts)
        self.assertEqual(load(profile="long", context="524288").context, 524288)  # an explicit context still wins

    def test_reference_profile_drops_drafts(self):
        s = load(profile="reference")
        self.assertFalse(s.drafts)
        self.assertIn("--no-drafts", s.serve_args("/m"))

    def test_explicit_setting_beats_profile(self):
        self.assertTrue(load(profile="reference", drafts="1").drafts)

    def test_env_overrides(self):
        s = load(port="8090", temperature="0", prompt_cache_gib="0", parallel="4", learn="1", learn_dir="~/l")
        args = s.serve_args("/m")
        self.assertEqual(s.url, "http://127.0.0.1:8090")
        self.assertEqual(args[args.index("--temperature") + 1], "0")
        self.assertEqual(args[args.index("--prompt-cache-gib") + 1], "0")
        self.assertEqual(args[args.index("--parallel") + 1], "4")
        self.assertIn("--learn", args)
        self.assertTrue(args[args.index("--learn-dir") + 1].endswith("/l"))
        self.assertIn("limit=200GB", s.summary())

    def test_server_env_carries_the_load_limit(self):
        s = load(extra_env="TF_X=1 TF_Y=a=b", request_log="~/req.log", memory_limit_gb="190")
        env = s.server_env()
        self.assertEqual(env["TENSORFOLD_MEMORY_LIMIT_GB"], "190")
        self.assertEqual(env["HF_HUB_OFFLINE"], "1")
        self.assertEqual(env["TF_Y"], "a=b")
        self.assertTrue(env["TENSORFOLD_REQUEST_LOG"].endswith("/req.log"))

    def test_patch_is_pinned(self):
        s = load()
        self.assertTrue(s.patch_path.is_file())
        self.assertEqual(len(s.patch_sha256), 64)


class Refusals(unittest.TestCase):
    def refuse(self, fragment, **env):
        with self.assertRaises(config.ConfigError) as cm:
            load(**env)
        self.assertIn(fragment, str(cm.exception))

    def test_context_bounds(self):
        self.refuse("must be in [256, 1048576]", context="2097152")
        self.refuse("multiple of 256", context="1000")
        self.refuse("not a decimal integer", context="0100")

    def test_profile(self):
        self.refuse("must be one of", profile="concurrent")

    def test_owned_and_unused_flags(self):
        self.refuse("passes itself", extra_args="--port 1")
        self.refuse("passes itself", extra_args="--no-dra")          # a prefix
        self.refuse("passes itself", extra_args="--parallel 4")
        self.refuse("not a GLM-on-one-Mac flag", extra_args="--speed-up r.json")
        self.refuse("passes itself", extra_args="--vision")  # the recipe owns it: the vision setting

    def test_memory_settings(self):
        self.refuse("outside", memory_limit_gb="0")
        self.refuse("use memory_limit_gb", extra_env="TENSORFOLD_MEMORY_LIMIT_GB=300")

    def test_patch_pin(self):
        self.refuse("not a sha256", patch_sha256="abc")
        self.refuse("not the pinned", patch_sha256="0" * 64)
        self.refuse("is missing", patch="nope.patch")

    def test_binary_must_exist(self):
        self.refuse("is not a file", binary="/nonexistent/tensorfold-native")

    def test_shapes(self):
        self.refuse("40-hex", revision="abc")
        self.refuse("not KEY=VALUE", extra_env="NOEQUALS")
        self.refuse("decimal", temperature="hot")
        self.refuse("greedily", temperature="0.7")
        self.refuse("greedily", temperature="")
        self.refuse("reasoning_effort", reasoning_effort="extreme")
        self.refuse("passes itself", extra_args="--reasoning-effort low")
        self.refuse("vision_image_tokens", vision_image_tokens="8")
        self.refuse("vision-urls", extra_args="--vision-urls")
        self.refuse("passes", extra_args="--vision-max-images 2")
        self.refuse("fraction", compact_at="2")
        self.refuse("outside", port="70000")
        self.refuse("org/name", repo="nothing")

    def test_unknown_setting(self):
        with self.assertRaises(config.ConfigError):
            config.load(local_file=Path(__file__).with_name("bad.toml"), env={})


if __name__ == "__main__":
    unittest.main()


class RecentKill(unittest.TestCase):
    def test_a_fresh_memguard_kill_blocks_a_restart(self):
        import tempfile
        import time

        from mac import serve
        with tempfile.TemporaryDirectory() as d:
            s = load(tf_cache=d)
            log = Path(d) / "memguard.log"
            stamp = time.strftime("%Y-%m-%d %H:%M:%S")
            log.write_text(f"[{stamp}] memguard: free 3.0 GiB, pressure 4, swap grew 0 MB, compressor grew 0.0 GiB: killing pid 1\n")
            self.assertIsNotNone(serve.recent_kill(s))
            old = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time() - 3600))
            log.write_text(f"[{old}] memguard: free 3.0 GiB: killing pid 1\n")
            self.assertIsNone(serve.recent_kill(s))
            log.write_text(f"[{stamp}] memguard: pid 1 gone, exiting\n")
            self.assertIsNone(serve.recent_kill(s))
