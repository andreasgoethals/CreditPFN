"""Upload local ``data/raw/`` to the VSC scratch via parallel SFTP.

A pure-Python replacement for WinSCP / scp when the dataset count is
large and you don't want to hand-drag-and-drop each one. Uses
``paramiko`` so it works on Windows without WSL / Git Bash / rsync,
and parallelises the upload across multiple SFTP sessions so a 5 GB
corpus finishes in minutes rather than half an hour.

Auth
----
Tries SSH key first, then falls back to **keyboard-interactive**
prompts (password + MFA code) if no usable key is found. Auth happens
exactly once — every parallel SFTP transfer multiplexes over the same
authenticated SSH connection.

The script reads ``~/.ssh/config`` for the host's ``HostName`` /
``User`` / ``IdentityFile`` entries, then falls back to the SSH agent
(OpenSSH ``ssh-agent`` / pageant) and default key paths. If none of
those work, you'll be prompted for whatever the server asks for —
typically your VSC password followed by an MFA token, the same flow
WinSCP uses.

**Recommended one-time setup:** register your SSH public key with VSC
at https://account.vscentrum.be/django/sshkey/. After that, key-based
auth works silently and you never see the MFA prompt for SFTP/SSH
again. Until then, the keyboard-interactive fallback gets you running.

Where the files land
--------------------
By default::

    <local-root>/{pd,lgd}/*.csv  →  $VSC_SCRATCH/CreditPFN/data/raw/{pd,lgd}/

``$VSC_SCRATCH`` is expanded on the **remote** side (via ``echo``)
so the path resolves to whatever the user's shell sees on the VSC
login node. Override the remote location with ``--remote-root`` if you
want to drop files on ``$VSC_DATA`` instead (e.g. when scratch was
purged).

Speed
-----
The hot path is parallelism: each worker thread opens its own
``Transport`` + ``SFTPClient``, uploads one file, closes the session.
Default 4 workers; the VSC firewall happily handles 8 concurrent SSH
sessions. Files that already exist remote-side with matching size are
skipped (cheap stat call — no transfer).

Examples
--------
::

    # Default — uploads ./data/raw/{pd,lgd}/*.csv to $VSC_SCRATCH/CreditPFN/data/raw/
    python src/utils/upload_to_vsc.py --user vsc38338

    # Override target dir (e.g. for a scratch-purge workaround)
    python src/utils/upload_to_vsc.py --user vsc38338 \\
        --remote-root '$VSC_DATA/CreditPFN/data/raw'

    # Re-upload everything (skip the size-match check)
    python src/utils/upload_to_vsc.py --user vsc38338 --force

    # More parallel workers (default 4; raise for fast networks)
    python src/utils/upload_to_vsc.py --user vsc38338 --workers 8

Dependencies
------------
``pip install paramiko``  (in ``requirements.txt``).
"""

from __future__ import annotations

import argparse
import getpass
import logging
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

LOGGER = logging.getLogger(__name__)

DEFAULT_HOST = "login.hpc.kuleuven.be"
DEFAULT_REMOTE_ROOT = "$VSC_SCRATCH/CreditPFN/data/raw"
DEFAULT_LOCAL_ROOT = "data/raw"
DEFAULT_WORKERS = 4


# --------------------------------------------------------------------------- #
# SSH config + auth helpers
# --------------------------------------------------------------------------- #


@dataclass
class HostConfig:
    """The resolved set of values used to open an SSH session.

    Pulled from ``~/.ssh/config`` (if present), with CLI overrides
    layered on top. ``identity_file`` is None when paramiko should
    fall back to the SSH agent / default key paths.
    """
    hostname: str
    port:     int
    username: str
    identity_file: Path | None


def resolve_host_config(
    host_alias: str, *, user: str | None, port: int | None,
    identity: str | None,
) -> HostConfig:
    """Mirror what OpenSSH does — read ``~/.ssh/config`` for the host."""
    try:
        import paramiko
    except ImportError as exc:                                       # pragma: no cover
        raise ImportError(
            "paramiko is required for the upload script. Install with: "
            "pip install paramiko"
        ) from exc

    ssh_cfg_path = Path.home() / ".ssh" / "config"
    cfg = paramiko.SSHConfig()
    if ssh_cfg_path.exists():
        cfg.parse(ssh_cfg_path.open())
    host_cfg = cfg.lookup(host_alias)

    hostname = host_cfg.get("hostname", host_alias)
    resolved_user = (
        user
        or host_cfg.get("user")
        or os.environ.get("USER")
        or os.environ.get("USERNAME")
    )
    if not resolved_user:
        raise ValueError(
            "Could not resolve SSH username — pass --user explicitly "
            "or add a `User` line to your ~/.ssh/config."
        )
    resolved_port = int(port or host_cfg.get("port", 22))

    identity_path: Path | None = None
    if identity:
        identity_path = Path(identity).expanduser()
    elif "identityfile" in host_cfg and host_cfg["identityfile"]:
        # paramiko returns a list (last-match-wins in SSH semantics).
        identity_path = Path(host_cfg["identityfile"][0]).expanduser()
    if identity_path and not identity_path.exists():
        LOGGER.warning(
            "SSH config points at %s but it doesn't exist; falling back "
            "to the SSH agent + default key paths.", identity_path,
        )
        identity_path = None

    return HostConfig(
        hostname=hostname,
        port=resolved_port,
        username=resolved_user,
        identity_file=identity_path,
    )


def _kbd_interactive_handler(title, instructions, prompt_list):
    """Forward each server prompt to the user (echoed or hidden).

    Called by paramiko's keyboard-interactive auth. VSC's flow is
    typically:
      1. a password prompt (echo=False, hidden via getpass)
      2. an MFA-token prompt (echo=False)
    Some installations add an informational ``title`` / ``instructions``
    block on the first call — we just print those to stderr so the
    user knows what's happening.
    """
    if title:
        print(f"\n  [auth] {title}", file=sys.stderr)
    if instructions:
        print(f"  [auth] {instructions}", file=sys.stderr)
    answers: list[str] = []
    for prompt_text, echo in prompt_list:
        label = prompt_text if prompt_text else "Password: "
        if echo:
            answers.append(input(label))
        else:
            answers.append(getpass.getpass(label))
    return answers


def _gather_candidate_keys(host: "HostConfig"):
    """Build the ordered list of private keys to try.

    Priority:
      1. The explicit ``--identity`` / ssh-config ``IdentityFile`` path.
      2. Keys in the SSH agent (``ssh-agent`` on *nix, pageant on Windows
         if paramiko was built against it).
      3. The standard ``~/.ssh/id_{ed25519,rsa,ecdsa,dsa}`` files.

    Yields ``paramiko.PKey`` instances — already parsed so the caller
    doesn't have to handle encoding errors mid-auth.
    """
    import paramiko

    if host.identity_file and host.identity_file.exists():
        try:
            yield _load_pkey(host.identity_file)
        except paramiko.SSHException as exc:
            LOGGER.warning("could not parse %s: %s", host.identity_file, exc)

    try:
        agent = paramiko.Agent()
        for key in agent.get_keys():
            yield key
    except paramiko.SSHException:                                      # pragma: no cover
        pass

    home_ssh = Path.home() / ".ssh"
    for name, cls in (
        ("id_ed25519", paramiko.Ed25519Key),
        ("id_rsa",     paramiko.RSAKey),
        ("id_ecdsa",   paramiko.ECDSAKey),
        ("id_dsa",     paramiko.DSSKey),
    ):
        path = home_ssh / name
        if not path.exists():
            continue
        try:
            yield cls.from_private_key_file(str(path))
        except paramiko.PasswordRequiredException:
            pw = getpass.getpass(f"  Passphrase for {path}: ")
            try:
                yield cls.from_private_key_file(str(path), password=pw)
            except paramiko.SSHException as exc:
                LOGGER.warning("could not unlock %s: %s", path, exc)
        except paramiko.SSHException:
            continue


def open_authenticated_transport(host: "HostConfig"):
    """Open + authenticate a paramiko Transport with sensible fallbacks.

    Order of attempts:
      1. **Publickey** via every candidate key from
         :func:`_gather_candidate_keys`. Silent.
      2. **Keyboard-interactive** (handles password + MFA prompts).
         Prints a hint about registering an SSH key for future runs.

    The returned transport is single-authenticated; the caller spawns
    as many SFTP channels off it as it wants for parallelism. This
    means one MFA round, not one per worker.
    """
    import paramiko

    transport = paramiko.Transport((host.hostname, host.port))
    transport.start_client(timeout=20.0)

    # 1) Publickey auth — try every key we can find.
    for key in _gather_candidate_keys(host):
        try:
            transport.auth_publickey(host.username, key)
            return transport
        except paramiko.AuthenticationException:
            continue
        except paramiko.SSHException as exc:                            # pragma: no cover
            LOGGER.warning("unexpected SSHException during publickey auth: %s", exc)
            continue

    # 2) Keyboard-interactive — handles password + MFA prompts.
    print(
        "  No SSH key auth available — falling back to interactive auth.\n"
        "  (Tip: register your public key at\n"
        "        https://account.vscentrum.be/django/sshkey/\n"
        "   to skip the password + MFA prompt next time.)",
        file=sys.stderr,
    )
    try:
        transport.auth_interactive(host.username, _kbd_interactive_handler)
        return transport
    except paramiko.SSHException as exc:
        transport.close()
        raise RuntimeError(
            f"SSH authentication failed: {exc}\n\n"
            "Things to check:\n"
            "  * Is your VSC username correct? (passed via --user)\n"
            "  * Did you finish the MFA prompt? Some VSC sites push a\n"
            "    one-time token; others need you to confirm a notification.\n"
            "  * For a one-time fix, register an SSH key at\n"
            "    https://account.vscentrum.be/django/sshkey/ — after that\n"
            "    paramiko will auth silently via the SSH agent / default\n"
            "    key paths and no MFA is needed for SFTP / SSH."
        ) from exc


def expand_remote_path(transport, raw_path: str) -> str:
    """Expand ``$VAR`` references via the remote shell.

    Re-uses the already-authenticated transport — no extra SSH session
    (and therefore no extra MFA round) is needed.
    """
    if "$" not in raw_path:
        return raw_path
    chan = transport.open_session()
    try:
        chan.exec_command(f'echo "{raw_path}"')
        # Wait for the exec to finish and read all stdout.
        out_bytes = b""
        while True:
            data = chan.recv(4096)
            if not data:
                break
            out_bytes += data
        return out_bytes.decode().strip()
    finally:
        chan.close()


# --------------------------------------------------------------------------- #
# Local enumeration + remote mkdir
# --------------------------------------------------------------------------- #


def collect_local_files(local_root: Path) -> list[Path]:
    """All regular files under ``local_root/{pd,lgd}/`` (one level deep)."""
    out: list[Path] = []
    if not local_root.exists():
        raise FileNotFoundError(
            f"Local raw-data directory not found: {local_root}"
        )
    for track in ("pd", "lgd"):
        d = local_root / track
        if not d.exists():
            LOGGER.warning("local subdir missing: %s — skipped", d)
            continue
        for p in sorted(d.iterdir()):
            if p.is_file():
                out.append(p)
    return out


def remote_mkdirs(sftp, remote_path: str) -> None:
    """Recursive remote mkdir. ``remote_path`` may be absolute."""
    parts = remote_path.strip("/").split("/")
    cur = ""
    for part in parts:
        cur = f"{cur}/{part}" if cur else f"/{part}"
        try:
            sftp.stat(cur)
        except FileNotFoundError:
            try:
                sftp.mkdir(cur)
            except IOError as exc:                                   # pragma: no cover
                # Race condition: another worker created it first.
                # Re-stat to confirm.
                try:
                    sftp.stat(cur)
                except FileNotFoundError as inner:
                    raise IOError(
                        f"failed to mkdir {cur!r} remotely: {exc}"
                    ) from inner


# --------------------------------------------------------------------------- #
# Per-file uploader (one worker thread = one SFTP session)
# --------------------------------------------------------------------------- #


@dataclass
class UploadResult:
    """One row of the final summary."""
    local:    Path
    remote:   str
    status:   str            # "OK" | "SKIP" | "FAIL"
    bytes:    int            # 0 for SKIP/FAIL
    elapsed:  float          # seconds
    reason:   str | None = None


def upload_one(
    transport, local: Path, remote: str, *, force: bool,
    mkdir_lock: "threading.Lock",
    known_dirs: set[str],
) -> UploadResult:
    """Open one SFTP channel on the shared transport, upload one file.

    Multiple worker threads call this concurrently. Each one gets its
    own SFTP channel (paramiko's ``Transport`` handles that under a
    lock), so transfers run in parallel — but they all share a single
    authenticated SSH session, so MFA only happened once.

    The ``mkdir_lock`` + ``known_dirs`` cache avoids each worker
    re-mkdir'ing the same parent directory (e.g. all 17 PD files share
    ``/scratch/.../data/raw/pd``). One worker creates it; the rest read
    from the cache.
    """
    import paramiko

    t0 = time.monotonic()
    try:
        sftp = paramiko.SFTPClient.from_transport(transport)
        if sftp is None:                                              # pragma: no cover
            raise RuntimeError("SFTPClient.from_transport returned None")
        try:
            parent = remote.rsplit("/", 1)[0]
            with mkdir_lock:
                if parent not in known_dirs:
                    remote_mkdirs(sftp, parent)
                    known_dirs.add(parent)
            return _upload_via_sftp(sftp, local, remote, force=force, t0=t0)
        finally:
            sftp.close()
    except Exception as exc:                                          # noqa: BLE001
        return UploadResult(
            local=local, remote=remote, status="FAIL",
            bytes=0, elapsed=time.monotonic() - t0,
            reason=f"{type(exc).__name__}: {exc}",
        )


def _load_pkey(identity_file: Path):
    """Try every keytype paramiko supports until one parses."""
    import paramiko
    for cls in (paramiko.Ed25519Key, paramiko.RSAKey,
                paramiko.ECDSAKey, paramiko.DSSKey):
        try:
            return cls.from_private_key_file(str(identity_file))
        except paramiko.SSHException:
            continue
    raise paramiko.SSHException(
        f"could not parse private key at {identity_file} as any of "
        "ed25519, rsa, ecdsa, dss"
    )


def _upload_via_sftp(sftp, local: Path, remote: str, *, force: bool, t0: float) -> UploadResult:
    """Stat-check then put. Assumes the parent directory already exists
    (the caller takes care of that under a lock so workers don't race).
    """
    local_size = local.stat().st_size
    if not force:
        try:
            rstat = sftp.stat(remote)
            if rstat.st_size == local_size:
                return UploadResult(
                    local=local, remote=remote, status="SKIP",
                    bytes=0, elapsed=time.monotonic() - t0,
                    reason="size matches (use --force to overwrite)",
                )
        except FileNotFoundError:
            pass

    sftp.put(str(local), remote)
    return UploadResult(
        local=local, remote=remote, status="OK",
        bytes=local_size, elapsed=time.monotonic() - t0,
    )


# --------------------------------------------------------------------------- #
# Top-level driver
# --------------------------------------------------------------------------- #


def upload_all(
    *, transport, host: HostConfig, local_root: Path, remote_root: str,
    workers: int, force: bool,
) -> list[UploadResult]:
    """Walk ``local_root/{pd,lgd}/``, upload everything in parallel
    over channels of the supplied authenticated ``transport``.

    ``remote_root`` is expected to already have shell vars expanded
    (call :func:`expand_remote_path` first).
    """
    files = collect_local_files(local_root)
    if not files:
        LOGGER.warning("nothing to upload — no files under %s/{pd,lgd}/", local_root)
        return []

    LOGGER.info(
        "Uploading %d file(s) to %s@%s:%s with %d parallel SFTP channels...",
        len(files), host.username, host.hostname, remote_root, workers,
    )

    mkdir_lock = threading.Lock()
    known_dirs: set[str] = set()

    results: list[UploadResult] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(
                upload_one, transport, f,
                f"{remote_root}/{f.parent.name}/{f.name}",
                force=force,
                mkdir_lock=mkdir_lock,
                known_dirs=known_dirs,
            ): f
            for f in files
        }
        for i, fut in enumerate(as_completed(futures), start=1):
            res = fut.result()
            results.append(res)
            rate = (res.bytes / res.elapsed / 1024 / 1024) if res.bytes > 0 and res.elapsed > 0 else 0
            tag = res.status.ljust(4)
            print(
                f"  [{i:>3}/{len(files)}] [{tag}] {res.local.name:<40s} "
                f"{res.bytes/1024/1024:>8.1f} MB  {res.elapsed:>5.1f}s"
                + (f"  ({rate:.1f} MB/s)" if rate else "")
                + (f"  ← {res.reason}" if res.reason else ""),
                flush=True,
            )

    return results


def print_summary(results: list[UploadResult]) -> None:
    n_ok   = sum(1 for r in results if r.status == "OK")
    n_skip = sum(1 for r in results if r.status == "SKIP")
    n_fail = sum(1 for r in results if r.status == "FAIL")
    total_bytes = sum(r.bytes for r in results if r.status == "OK")

    bar = "-" * 70
    print()
    print(bar)
    print(f"  Upload summary — OK={n_ok}  SKIP={n_skip}  FAIL={n_fail}  "
          f"transferred={total_bytes/1024/1024:.1f} MB")
    print(bar)
    if n_fail:
        print("Failed files:")
        for r in results:
            if r.status == "FAIL":
                print(f"  * {r.local}  → {r.remote}")
                print(f"      reason: {r.reason}")
        print(bar)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Upload local data/raw/ to VSC scratch via parallel SFTP.",
    )
    parser.add_argument(
        "--host", default=DEFAULT_HOST,
        help=f"SSH host alias (resolved against ~/.ssh/config). "
             f"Default: {DEFAULT_HOST}",
    )
    parser.add_argument(
        "--user", default=None,
        help="SSH username (your VSC account, e.g. vsc38338). "
             "Falls back to ~/.ssh/config or $USER / $USERNAME.",
    )
    parser.add_argument(
        "--port", type=int, default=None,
        help="SSH port. Defaults to whatever ~/.ssh/config says, "
             "else 22.",
    )
    parser.add_argument(
        "--identity", default=None,
        help="Path to an SSH private key. Defaults to ~/.ssh/config's "
             "IdentityFile, else the SSH agent / default key paths.",
    )
    parser.add_argument(
        "--local-root", default=DEFAULT_LOCAL_ROOT,
        help=f"Local root containing {{pd,lgd}}/*.csv. "
             f"Default: {DEFAULT_LOCAL_ROOT}",
    )
    parser.add_argument(
        "--remote-root", default=DEFAULT_REMOTE_ROOT,
        help=f"Remote root. Shell vars (e.g. $VSC_SCRATCH) are expanded "
             f"on the VSC side. Default: {DEFAULT_REMOTE_ROOT}",
    )
    parser.add_argument(
        "--workers", type=int, default=DEFAULT_WORKERS,
        help=f"Number of parallel SFTP sessions. "
             f"Default: {DEFAULT_WORKERS}",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-upload every file, even if remote size matches.",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s", force=True)
    # Quiet paramiko's own logger by default (the "Connected (version 2.0,
    # client Go)" line and downstream chatter add noise; keep WARNINGs).
    logging.getLogger("paramiko").setLevel(logging.WARNING)

    local_root = Path(args.local_root).resolve()
    transport = None
    try:
        host = resolve_host_config(
            args.host, user=args.user, port=args.port, identity=args.identity,
        )
        # Authenticate ONCE — every parallel SFTP transfer multiplexes over
        # this single transport, so the MFA prompt (if any) happens once.
        transport = open_authenticated_transport(host)
        remote_root = expand_remote_path(transport, args.remote_root)
        results = upload_all(
            transport=transport, host=host, local_root=local_root,
            remote_root=remote_root,
            workers=int(args.workers), force=bool(args.force),
        )
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    except ImportError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 3
    finally:
        if transport is not None:
            transport.close()

    print_summary(results)
    n_fail = sum(1 for r in results if r.status == "FAIL")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
