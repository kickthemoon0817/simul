# Security policy

## Supported versions

Simul is pre-1.0 and the Claude Code marketplace installs it from `main` with
no version pin, so only the latest release on `main` is supported. Security
fixes ship there as a patch release.

## Reporting a vulnerability

Report it privately through GitHub:
[**Report a vulnerability**](https://github.com/kickthemoon0817/simul/security/advisories/new)
(Security tab → *Report a vulnerability*). Please don't open a public issue
or pull request for a security problem.

Include what you can of:

- the version (`simul --json version`), the backend and its version (Isaac
  Sim, Unreal Engine, Blender or headless USD) and the OS;
- the steps or tool calls that reproduce it;
- what an attacker gains, and from where (same user, another local user,
  another host).

The discussion stays in the private advisory until a fix is released; the
advisory is then published with credit to you unless you'd rather stay
anonymous.

## Scope

Simul exists to let an agent run code inside a 3D application, so the trust
model matters when deciding what is a vulnerability. The defaults are
described in [README.md § Security](README.md#security) and
[docs/configuration.md](docs/configuration.md#security).

In scope, for example:

- escaping the file sandbox (`security.sandbox.allowed_paths`) through a
  tool's path arguments;
- reaching an engine bridge from another host or another local user where
  the docs say it is loopback-only or protected: the Blender bridge's
  per-instance token in owner-only discovery/attachment files, the Isaac
  Sim socket auth token, or an Unreal Remote Control passphrase;
- leaking those tokens or passphrases through logs, process arguments,
  discovery files or tool results;
- code execution that remains possible with
  `SECURITY__ALLOW_SCRIPT_EXECUTION=false`;
- injection into the Python that simul generates from tool arguments.

Out of scope:

- an agent or user you have allowed to execute scripts running arbitrary
  code in the engine — that is the product's purpose;
- exposure you opted into, such as `simul unreal setup --bind <host>
  --allow-public` (Unreal Remote Control has no authentication of its own);
- vulnerabilities in Isaac Sim, Unreal Engine, Blender or OpenUSD
  themselves — please report those upstream.
