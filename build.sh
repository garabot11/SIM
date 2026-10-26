#!/usr/bin/env bash
# Сборка образа gumanoid-sim (Ubuntu 24.04 + MuJoCo 3.15 + модели G1, H2, Zeroth-01 и др.).
#   ./build.sh                  обычная сборка
#   ./build.sh --no-cache       пересобрать всё заново (любые опции передаются в docker build)
# Если в окружении задан прокси (HTTP_PROXY/HTTPS_PROXY), он передаётся в сборку: apt, pip и git пойдут через него.
# Если Docker не может скачать базовый образ сам (Docker Hub недоступен демону), его скачает docker/pull-image.py
# через прокси терминала.
set -euo pipefail

# Только что добавлены в группу docker (install.sh), но ещё не перезаходили в систему: перезапуск с этой группой.
if ! docker info >/dev/null 2>&1 && [[ -z "${GUMANOID_SG:-}" ]] && ! id -nG | grep -qw docker &&
        id -nG "$(id -un)" | grep -qw docker; then
    GUMANOID_SG=1 exec sg docker -c "$(printf '%q ' "$0" "$@")"
fi

cd "$(dirname "${BASH_SOURCE[0]}")"
IMAGE="${IMAGE:-gumanoid-sim}"

# --- базовый образ (строка FROM в Dockerfile)
BASE="$(awk '/^FROM /{print $2; exit}' Dockerfile)"
if ! docker image inspect "$BASE" >/dev/null 2>&1; then
    echo "build.sh: скачиваю базовый образ $BASE"
    if ! docker pull "$BASE"; then
        echo "build.sh: docker pull не смог, пробую напрямую через прокси терминала (docker/pull-image.py)"
        python3 docker/pull-image.py "$BASE"
    fi
fi

# сеть хоста: прокси вида http://127.0.0.1:порт доступен из сборки только так
args=(-t "$IMAGE" --network host)
for v in HTTP_PROXY HTTPS_PROXY NO_PROXY http_proxy https_proxy no_proxy; do
    if [[ -n "${!v:-}" ]]; then args+=(--build-arg "$v=${!v}"); fi
done

docker build "${args[@]}" "$@" .
echo
echo "Готово: образ $IMAGE, $(docker images "$IMAGE" --format '{{.Size}}' | head -1)."
echo "Проверка:  ./run.sh test      Запуск:  ./run.sh"
