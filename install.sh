#!/usr/bin/env bash
# Установка gumanoid-sim одной командой: Docker (если его нет), NVIDIA Container Toolkit (если есть видеокарта
# NVIDIA с драйвером), сборка образа, самопроверка, команда `gumanoid` и ярлык в меню приложений.
#   ./install.sh               из папки проекта
#   curl -fsSL https://raw.githubusercontent.com/garabot11/SIM/main/install.sh | bash
#                              без папки проекта: сам скачает его в ~/gumanoid (или в GUMANOID_DIR)
# Опции: --no-nvidia  не ставить NVIDIA Container Toolkit (рисовать встроенной графикой или процессором)
#        --no-test    не запускать самопроверку
# Запускать без sudo: пароль спросят только для установки пакетов. Повторный запуск безопасен: готовые шаги
# пропускаются, образ пересобирается из кэша за секунды. Прокси терминала (HTTP_PROXY/HTTPS_PROXY) передаётся
# в apt, git и сборку.
set -euo pipefail

# Всё тело — один блок { … }: bash читает его целиком до выполнения, поэтому `curl … | bash` безопасен
# (программы внутри не «съедят» остаток скрипта из stdin).
{
REPO_URL="${REPO_URL:-https://github.com/garabot11/SIM.git}"
DIR="${GUMANOID_DIR:-$HOME/gumanoid}"
DOCKER_MIN=20.10.10   # более старый Docker не запускает ubuntu:24.04 (README, пункт 2.4)
NVIDIA=1 TEST=1

usage() {
    cat <<'EOF'
Установка gumanoid-sim: Docker, сборка образа, команда gumanoid и ярлык в меню.
  ./install.sh [--no-nvidia] [--no-test]
  --no-nvidia  не ставить NVIDIA Container Toolkit (рисовать встроенной графикой или процессором)
  --no-test    не запускать самопроверку
Переменные: GUMANOID_DIR — куда скачать проект (по умолчанию ~/gumanoid), REPO_URL — откуда.
EOF
}
for a in "$@"; do
    case "$a" in
        --no-nvidia) NVIDIA=0 ;;
        --no-test) TEST=0 ;;
        -h|--help) usage; exit 0 ;;
        *) usage >&2; exit 2 ;;
    esac
done

if [[ -t 1 ]]; then B=$'\e[1m' G=$'\e[32m' Y=$'\e[33m' R=$'\e[31m' N=$'\e[0m'; else B='' G='' Y='' R='' N=''; fi
step() { printf '\n%s==> %s%s\n' "$B" "$*" "$N"; }
ok()   { printf '    %s✓%s %s\n' "$G" "$N" "$*"; }
warn() { printf '    %s!%s %s\n' "$Y" "$N" "$*" >&2; }
die()  { printf '\n%sinstall.sh: %s%s\n' "$R" "$*" "$N" >&2; exit 1; }

[[ $EUID -ne 0 ]] || die "запускайте без sudo: ./install.sh (пароль спросят, когда понадобится)"

# sudo сбрасывает переменные окружения, поэтому прокси передаётся явно.
PROXY_ENV=()
for v in http_proxy https_proxy no_proxy HTTP_PROXY HTTPS_PROXY NO_PROXY; do
    if [[ -n "${!v:-}" ]]; then PROXY_ENV+=("$v=${!v}"); fi
done
as_root() { sudo env "${PROXY_ENV[@]}" "$@"; }
APT_UPDATED=0
apt_install() {
    if ((!APT_UPDATED)); then as_root apt-get update; APT_UPDATED=1; fi
    as_root apt-get install -y "$@"
}

# --- 1. система
step "Проверка системы"
OS_ID='' OS_LIKE='' CODENAME='' OS_NAME="$(uname -s)"
if [[ -r /etc/os-release ]]; then
    # shellcheck disable=SC1091
    . /etc/os-release
    OS_ID="${ID:-}" OS_LIKE="${ID_LIKE:-}" OS_NAME="${PRETTY_NAME:-$OS_NAME}"
    CODENAME="${UBUNTU_CODENAME:-${VERSION_CODENAME:-}}"   # Linux Mint и др.: название базовой Ubuntu
fi
UBUNTU=0
if [[ "$OS_ID" == ubuntu || " $OS_LIKE " == *" ubuntu "* ]]; then UBUNTU=1; fi
ok "$OS_NAME, $(uname -m)"
((UBUNTU)) || warn "не Ubuntu: Docker и пакеты придётся ставить вручную, если их нет"
[[ "$(uname -m)" == x86_64 ]] || warn "процессор $(uname -m): проверено только на x86_64"

# --- 2. проект
step "Проект"
SELF="${BASH_SOURCE[0]:-}"
SRC=''
if [[ -n "$SELF" && -f "$SELF" ]]; then SRC="$(cd "$(dirname "$SELF")" && pwd)"; fi
if [[ -n "$SRC" && -f "$SRC/Dockerfile" && -f "$SRC/run.sh" ]]; then
    DIR="$SRC"
    ok "папка $DIR"
else
    if ! command -v git >/dev/null; then
        ((UBUNTU)) || die "нужен git"
        apt_install git
    fi
    if [[ -d "$DIR/.git" ]]; then
        git -C "$DIR" pull --ff-only || warn "не удалось обновить $DIR (git pull), продолжаю с тем, что есть"
    elif [[ -e "$DIR" ]]; then
        die "папка $DIR уже есть и это не проект; укажите другую: GUMANOID_DIR=~/другая-папка bash install.sh"
    else
        git clone --depth 1 "$REPO_URL" "$DIR"
    fi
    ok "скачан в $DIR"
fi
cd "$DIR"

# --- 3. Docker
step "Docker"
docker_version() { docker --version 2>/dev/null | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' | head -1; }
version_ge() { [[ "$(printf '%s\n%s\n' "$1" "$2" | sort -V | head -1)" == "$2" ]]; }   # $1 >= $2
install_docker() {
    ((UBUNTU)) || die "поставьте Docker $DOCKER_MIN или новее вручную (https://docs.docker.com/engine/install/) и запустите install.sh снова"
    # те же команды, что в README, пункт 2.1
    apt_install ca-certificates curl
    as_root install -m 0755 -d /etc/apt/keyrings
    if as_root curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc; then
        as_root chmod a+r /etc/apt/keyrings/docker.asc
        echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] \
https://download.docker.com/linux/ubuntu $CODENAME stable" | as_root tee /etc/apt/sources.list.d/docker.list >/dev/null
        if as_root apt-get update && as_root apt-get install -y docker-ce docker-ce-cli containerd.io \
                docker-buildx-plugin docker-compose-plugin; then
            return 0
        fi
        as_root rm -f /etc/apt/sources.list.d/docker.list
    fi
    warn "репозиторий download.docker.com недоступен, ставлю Docker из репозитория Ubuntu"
    as_root apt-get update
    as_root apt-get install -y docker.io docker-buildx || as_root apt-get install -y docker.io
}
DOCKER_BIN="$(command -v docker || true)"
if [[ "$DOCKER_BIN" == /snap/* ]]; then
    die "Docker установлен из snap: окно симулятора с ним не открывается. Удалите его (sudo snap remove docker) и запустите install.sh снова"
fi
if [[ -z "$DOCKER_BIN" ]]; then
    echo "    Docker не установлен, ставлю (нужен пароль sudo)"
    install_docker
elif ! version_ge "$(docker_version)" "$DOCKER_MIN"; then
    echo "    Docker $(docker_version) слишком старый (нужен $DOCKER_MIN или новее), обновляю"
    install_docker
fi
command -v docker >/dev/null || die "docker не установился"
ok "Docker $(docker_version)"

# Пользователь в группе docker: тогда docker работает без sudo. Только что добавленному группа достаётся после
# перезахода в систему, а до него docker вызывается через `sg docker` (run.sh и build.sh делают так же).
if ! getent group docker >/dev/null; then as_root groupadd docker; fi
if ! id -nG "$(id -un)" | grep -qw docker; then
    as_root usermod -aG docker "$(id -un)"
    ok "пользователь $(id -un) добавлен в группу docker"
fi
dk() {
    if docker info >/dev/null 2>&1 || id -nG | grep -qw docker; then "$@"; else sg docker -c "$(printf '%q ' "$@")"; fi
}
if ! dk docker info >/dev/null 2>&1; then
    echo "    служба Docker не отвечает, запускаю"
    as_root systemctl enable --now docker || true
    dk docker info >/dev/null 2>&1 || die "Docker не запускается: посмотрите  sudo systemctl status docker"
fi
ok "служба Docker работает"

# --- 4. видеокарта: NVIDIA — по возможности; без неё всё работает на встроенной графике или процессоре
step "Видеокарта"
if [[ -d /dev/dri ]]; then ok "встроенная/обычная графика (Mesa): /dev/dri есть"; fi
NV_CARD=0
if command -v lspci >/dev/null && lspci 2>/dev/null | grep -iE 'vga|3d|display' | grep -qi nvidia; then NV_CARD=1; fi
if command -v nvidia-smi >/dev/null && nvidia-smi -L >/dev/null 2>&1; then
    ok "NVIDIA: $(nvidia-smi -L | head -1 | sed 's/ (UUID.*//')"
    if dk docker info --format '{{json .Runtimes}}' 2>/dev/null | grep -q nvidia; then
        ok "NVIDIA Container Toolkit уже настроен"
    elif ((!NVIDIA)); then
        warn "NVIDIA Container Toolkit не ставлю (--no-nvidia): рисовать будет встроенная графика или процессор"
    elif ((!UBUNTU)); then
        warn "NVIDIA Container Toolkit поставьте вручную (README, раздел 3); пока будет встроенная графика или процессор"
    else
        echo "    ставлю NVIDIA Container Toolkit (README, пункт 3.2)"
        # Любая ошибка здесь не мешает установке: run.sh тогда рисует без NVIDIA.
        if curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
                | as_root gpg --batch --yes --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg &&
            curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
                | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
                | as_root tee /etc/apt/sources.list.d/nvidia-container-toolkit.list >/dev/null &&
            as_root apt-get update &&
            as_root apt-get install -y nvidia-container-toolkit &&
            as_root nvidia-ctk runtime configure --runtime=docker &&
            as_root systemctl restart docker; then
            ok "NVIDIA Container Toolkit установлен"
        else
            as_root rm -f /etc/apt/sources.list.d/nvidia-container-toolkit.list || true
            warn "NVIDIA Container Toolkit не установился: рисовать будет встроенная графика или процессор"
        fi
        dk docker info >/dev/null 2>&1 || as_root systemctl start docker || true
    fi
elif ((NV_CARD)); then
    warn "видеокарта NVIDIA есть, но драйвер не работает (nvidia-smi): пока рисует встроенная графика или процессор."
    warn "Чтобы рисовала NVIDIA: sudo ubuntu-drivers install, перезагрузка, затем ./install.sh ещё раз"
else
    ok "NVIDIA нет: рисует $([[ -d /dev/dri ]] && echo 'встроенная графика' || echo 'процессор (программный рендер)')"
fi
if ! command -v xhost >/dev/null && ((UBUNTU)); then
    apt_install x11-xserver-utils || warn "xhost не установился; если окно не откроется, см. README, «Частые проблемы»"
fi

# --- 5. образ
step "Сборка образа (первый раз 5–15 минут, дальше секунды)"
./build.sh

# --- 6. самопроверка; если GPU не справился с рендером — запасной вариант на процессоре
GPU_DEFAULT=''
if ((TEST)); then
    step "Самопроверка"
    if ./run.sh test; then
        ok "самопроверка прошла"
    elif [[ -z "${GPU:-}" ]] && GPU=cpu ./run.sh test; then
        GPU_DEFAULT=cpu
        warn "с видеокартой самопроверка не прошла, а на процессоре прошла: команда gumanoid будет рисовать процессором"
        warn "(попробовать видеокарту: GPU=auto gumanoid)"
    else
        warn "самопроверка не прошла, см. вывод выше и README, «Частые проблемы»"
    fi
fi

# --- 7. команда gumanoid и ярлык
step "Команда gumanoid и ярлык в меню приложений"
BIN_DIR="$HOME/.local/bin"
mkdir -p "$BIN_DIR"
{
    echo '#!/usr/bin/env bash'
    echo '# gumanoid-sim (создано install.sh): gumanoid — меню, gumanoid help — все команды, опции — как у run.sh.'
    if [[ -n "$GPU_DEFAULT" ]]; then
        echo "export GPU=\"\${GPU:-$GPU_DEFAULT}\"  # при установке рендер на видеокарте не прошёл самопроверку"
    fi
    printf 'exec %q "$@"\n' "$DIR/run.sh"
} >"$BIN_DIR/gumanoid"
chmod 755 "$BIN_DIR/gumanoid"
ok "$BIN_DIR/gumanoid"

APP_DIR="${XDG_DATA_HOME:-$HOME/.local/share}/applications"
mkdir -p "$APP_DIR"
cat >"$APP_DIR/gumanoid-sim.desktop" <<EOF
[Desktop Entry]
Type=Application
Name=gumanoid-sim
Comment=MuJoCo: Unitree G1, H2, Zeroth-01 и другие роботы с ползунками
Exec="$BIN_DIR/gumanoid"
Terminal=true
Icon=applications-science
Categories=Science;
EOF
ok "ярлык «gumanoid-sim» в меню приложений"

# --- итог
step "Готово"
CMD=gumanoid
case ":$PATH:" in *":$BIN_DIR:"*) ;; *) CMD="$BIN_DIR/gumanoid" ;; esac
cat <<EOF
    $CMD                  меню с выбором робота (или ярлык «gumanoid-sim» в меню приложений)
    $CMD g1 --band        сразу Unitree G1 на подвеске
    $CMD help             все команды
    Инструкция: $DIR/README.md
EOF
if [[ "$CMD" != gumanoid ]]; then
    echo "    (короткая команда gumanoid заработает после перезахода в систему: ~/.local/bin попадёт в PATH)"
fi
if ! id -nG | grep -qw docker && ! docker info >/dev/null 2>&1; then
    echo "    Команда docker без sudo заработает после перезахода в систему; gumanoid и run.sh работают уже сейчас."
fi
exit 0
}
