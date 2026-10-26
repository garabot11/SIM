#!/usr/bin/env bash
# Запуск gumanoid-sim из Docker с окном на экране хоста (X11 / XWayland).
#   ./run.sh                 меню
#   ./run.sh g1 --band       любая команда образа (./run.sh help — список)
# Переменные окружения:
#   GPU=auto|nvidia|dri|cpu  чем рисовать (по умолчанию auto: NVIDIA -> Intel/AMD -> программный рендер)
#   PRIME=auto|0|1           гибридная графика (ноутбук iGPU + NVIDIA): рисовать на NVIDIA через PRIME offload
#   IMAGE=gumanoid-sim       имя образа;   OUT_DIR=./out   папка хоста для скриншотов, файлов из viewer'а и своих скриптов
#   MODELS_DIR=./models      папка хоста со своими роботами (в контейнере /models, только чтение)
#   NET=host                 сеть хоста (для examples/udp_bridge.py, ROS 2 и т.п.); по умолчанию — своя сеть контейнера
#   DOCKER_ARGS="..."        любые дополнительные опции docker run, например  DOCKER_ARGS="-v $HOME/robots:/robots:ro"
set -euo pipefail

IMAGE="${IMAGE:-gumanoid-sim}"
GPU="${GPU:-auto}"
PRIME="${PRIME:-auto}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT_DIR="${OUT_DIR:-$HERE/out}"
MODELS_DIR="${MODELS_DIR:-$HERE/models}"
NET="${NET:-}"

die() { echo "run.sh: $*" >&2; exit 1; }

command -v docker >/dev/null || die "docker не установлен: запустите ./install.sh"
# Только что добавлены в группу docker (install.sh), но ещё не перезаходили в систему: перезапуск с этой группой.
if ! docker info >/dev/null 2>&1 && [[ -z "${GUMANOID_SG:-}" ]] && ! id -nG | grep -qw docker &&
        id -nG "$(id -un)" | grep -qw docker; then
    GUMANOID_SG=1 exec sg docker -c "$(printf '%q ' "$0" "$@")"
fi
docker info >/dev/null 2>&1 || die "нет доступа к Docker: запущен ли сервис и состоите ли вы в группе docker? (README.md, «Частые проблемы»)"
docker image inspect "$IMAGE" >/dev/null 2>&1 || die "образ $IMAGE не найден: запустите ./install.sh (или ./build.sh)"

NAME="gumanoid-sim-$$"
args=(--rm --init --name "$NAME")
if [[ -t 0 && -t 1 ]]; then args+=(-it); fi
if [[ "$NET" == host ]]; then
    args+=(--network host)        # порты симулятора (UDP 9870 и др.) видны на хосте как 127.0.0.1
else
    args+=(--hostname gumanoid-sim)
fi

# --- экран: X11-сокет хоста. Под Wayland (Ubuntu 24.04 по умолчанию) работает через XWayland, DISPLAY тоже задан.
# Без окна (самопроверка, справка, список моделей) экран не нужен: так работает и по ssh.
if [[ -n "${DISPLAY:-}" ]]; then
    args+=(-e DISPLAY="$DISPLAY" -v /tmp/.X11-unix:/tmp/.X11-unix:ro)
    if [[ -n "${XAUTHORITY:-}" && -f "$XAUTHORITY" ]]; then
        args+=(-e XAUTHORITY=/tmp/.Xauthority -v "$XAUTHORITY:/tmp/.Xauthority:ro")
    elif [[ -f "$HOME/.Xauthority" ]]; then
        args+=(-e XAUTHORITY=/tmp/.Xauthority -v "$HOME/.Xauthority:/tmp/.Xauthority:ro")
    fi
    # Разрешить X-серверу окна от своего же пользователя (контейнер работает с вашим UID). Сбросится после перезахода.
    if command -v xhost >/dev/null; then xhost "+SI:localuser:$(id -un)" >/dev/null 2>&1 || true; fi
else
    case "${1:-menu}" in
        test|selftest|help|-h|--help|models) ;;
        *) die "переменная DISPLAY пуста: запускайте из терминала графического сеанса (не по ssh без -X)" ;;
    esac
fi

# --- пользователь: ваш UID/GID, чтобы файлы в out/ были ваши, а не root
mkdir -p "$OUT_DIR" "$MODELS_DIR"
args+=(--user "$(id -u):$(id -g)" -e HOME=/tmp -v "$OUT_DIR:/work" -w /work -v "$MODELS_DIR:/models:ro")

# --- видеокарта
has_nvidia() {
    command -v nvidia-smi >/dev/null && nvidia-smi -L >/dev/null 2>&1 &&
        docker info --format '{{json .Runtimes}}' 2>/dev/null | grep -q nvidia
}
gargs=()   # опции docker run для выбранной графики (set_gpu)
add_dri() {
    [[ -d /dev/dri ]] || return 0
    gargs+=(--device /dev/dri)
    local gid
    for gid in $(stat -c %g /dev/dri/* 2>/dev/null | sort -u); do
        if [[ "$gid" != 0 ]]; then gargs+=(--group-add "$gid"); fi
    done
    return 0
}
set_gpu() {
    gargs=()
    case "$1" in
        nvidia)
            gargs+=(--gpus all -e NVIDIA_DRIVER_CAPABILITIES=all)
            add_dri
            if [[ "$PRIME" == auto ]]; then
                # гибрид: экран на встроенной графике, NVIDIA — «Sink Offload» провайдер (NVIDIA-G0)
                PRIME=0
                if command -v xrandr >/dev/null && xrandr --listproviders 2>/dev/null | grep -q 'NVIDIA-G'; then PRIME=1; fi
            fi
            if [[ "$PRIME" == 1 ]]; then gargs+=(-e __NV_PRIME_RENDER_OFFLOAD=1 -e __GLX_VENDOR_LIBRARY_NAME=nvidia); fi
            ;;
        dri) add_dri ;;
        cpu) gargs+=(-e LIBGL_ALWAYS_SOFTWARE=1) ;;
        *) die "GPU=$1: допустимо auto, nvidia, dri, cpu" ;;
    esac
}
GPU_AUTO=0
if [[ "$GPU" == auto ]]; then
    GPU_AUTO=1
    if has_nvidia; then GPU=nvidia
    elif [[ -d /dev/dri ]]; then GPU=dri
    else GPU=cpu
    fi
fi
set_gpu "$GPU"

# --- свои опции docker run (разбиваются по пробелам)
if [[ -n "${DOCKER_ARGS:-}" ]]; then
    read -ra extra <<<"$DOCKER_ARGS"
    args+=("${extra[@]}")
fi

echo "run.sh: образ $IMAGE, графика: $GPU$([[ "$GPU" == nvidia && "$PRIME" == 1 ]] && echo ' (PRIME offload)'), файлы: $OUT_DIR, модели: $MODELS_DIR$([[ "$NET" == host ]] && echo ', сеть хоста')" >&2
if [[ "$GPU_AUTO" == 1 && "$GPU" == nvidia ]]; then
    # NVIDIA есть, но контейнер с ней может не запуститься (сломан nvidia-container-toolkit, драйвер обновился без
    # перезагрузки): тогда docker run выходит с кодом 125 до старта программы — повторить без NVIDIA.
    rc=0
    docker run "${args[@]}" "${gargs[@]}" "$IMAGE" "$@" || rc=$?
    [[ "$rc" == 125 ]] || exit "$rc"
    docker rm -f "$NAME" >/dev/null 2>&1 || true
    if [[ -d /dev/dri ]]; then GPU=dri; else GPU=cpu; fi
    echo "run.sh: контейнер с NVIDIA не запустился, графика: $GPU (выбрать вручную: GPU=nvidia|dri|cpu ./run.sh)" >&2
    set_gpu "$GPU"
fi
exec docker run "${args[@]}" "${gargs[@]}" "$IMAGE" "$@"
