#!/usr/bin/env python3
"""Скачать образ с Docker Hub через прокси терминала и загрузить его в Docker (`docker load`).

Нужен, когда демон Docker не видит Docker Hub (`docker pull` → "server misbehaving", "TLS handshake timeout"),
а в терминале интернет есть через прокси (HTTPS_PROXY). Демону прокси не передаётся — образ качает этот скрипт.
Только стандартная библиотека Python 3.

    python3 docker/pull-image.py ubuntu:24.04
"""
from __future__ import annotations  # Python 3.8 (Ubuntu 20.04) не понимает `str | None` в аннотациях

import gzip
import hashlib
import io
import json
import platform
import subprocess
import sys
import tarfile
import urllib.parse
import urllib.request

REGISTRY = "https://registry-1.docker.io"
INDEX_TYPES = ("application/vnd.oci.image.index.v1+json", "application/vnd.docker.distribution.manifest.list.v2+json")
MANIFEST_TYPES = ("application/vnd.oci.image.manifest.v1+json", "application/vnd.docker.distribution.manifest.v2+json")


class DropAuthOnRedirect(urllib.request.HTTPRedirectHandler):
    """Слои отдаются редиректом на CDN; токен Docker Hub туда слать нельзя (подписанная ссылка его отвергает)."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        new = super().redirect_request(req, fp, code, msg, headers, newurl)
        if new is not None and urllib.parse.urlsplit(newurl).netloc != urllib.parse.urlsplit(req.full_url).netloc:
            new.remove_header("Authorization")
        return new


opener = urllib.request.build_opener(DropAuthOnRedirect)  # прокси берётся из HTTPS_PROXY/https_proxy


def get(url: str, token: str | None = None, accept: tuple[str, ...] = ()) -> tuple[bytes, str]:
    req = urllib.request.Request(url)
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    if accept:
        req.add_header("Accept", ", ".join(accept))
    with opener.open(req, timeout=120) as r:
        return r.read(), r.headers.get("Content-Type", "")


def main() -> None:
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    ref = sys.argv[1]
    name, _, tag = ref.partition(":")
    tag = tag or "latest"
    repo = name if "/" in name else f"library/{name}"
    arch = {"x86_64": "amd64", "aarch64": "arm64"}.get(platform.machine(), platform.machine())

    scope = urllib.parse.quote(f"repository:{repo}:pull")
    token = json.loads(get(f"https://auth.docker.io/token?service=registry.docker.io&scope={scope}")[0])["token"]

    body, ctype = get(f"{REGISTRY}/v2/{repo}/manifests/{tag}", token, INDEX_TYPES + MANIFEST_TYPES)
    if ctype.split(";")[0] in INDEX_TYPES:
        entry = next(m for m in json.loads(body)["manifests"]
                     if m["platform"]["os"] == "linux" and m["platform"]["architecture"] == arch)
        body, ctype = get(f"{REGISTRY}/v2/{repo}/manifests/{entry['digest']}", token, MANIFEST_TYPES)
    manifest = json.loads(body)

    def blob(digest: str) -> bytes:
        data = get(f"{REGISTRY}/v2/{repo}/blobs/{digest}", token)[0]
        if "sha256:" + hashlib.sha256(data).hexdigest() != digest:
            sys.exit(f"pull-image: контрольная сумма не сошлась для {digest}")
        return data

    # Архив в формате `docker save`: manifest.json + конфиг + слои. Слои распакованы: такой архив принимает
    # docker load любой версии (Docker 20.10 из Ubuntu 20.04 тоже).
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tar:
        def add(path: str, data: bytes) -> None:
            info = tarfile.TarInfo(path)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))

        config = manifest["config"]["digest"]
        add(f"{config[7:]}.json", blob(config))
        layers = []
        for i, layer in enumerate(manifest["layers"], 1):
            print(f"pull-image: {ref} слой {i}/{len(manifest['layers'])}, {layer['size'] / 1e6:.1f} МБ", file=sys.stderr)
            layers.append(f"{layer['digest'][7:]}/layer.tar")
            data = blob(layer["digest"])
            add(layers[-1], gzip.decompress(data) if data[:2] == b"\x1f\x8b" else data)
        add("manifest.json", json.dumps([{"Config": f"{config[7:]}.json", "RepoTags": [f"{name}:{tag}"],
                                          "Layers": layers}]).encode())
    subprocess.run(["docker", "load"], input=buf.getvalue(), check=True)


if __name__ == "__main__":
    main()
