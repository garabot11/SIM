# MuJoCo 3.15 + гуманоиды Unitree G1, Unitree H2 и Zeroth-01 с ползунками суставов, Ubuntu 24.04.
#   сборка:  ./build.sh   (или docker build -t gumanoid-sim .)
#   запуск:  ./run.sh            (меню)   или   ./run.sh g1 | h2 | zbot | robot <файл> ...   — подробно в README.md
FROM ubuntu:24.04

ARG DEBIAN_FRONTEND=noninteractive
# Python 3.12 (+venv) и git (скачать модели). OpenGL/EGL — через libglvnd: при --gpus драйвер NVIDIA подкладывает
# nvidia-container-toolkit, для Intel/AMD и программного рендера (llvmpipe) — Mesa. X11-библиотеки нужны GLFW
# (окно viewer'а), mesa-utils — для `sim glinfo`.
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-venv git ca-certificates \
        libglvnd0 libgl1 libglx0 libegl1 libgles2 libglx-mesa0 libegl-mesa0 libgl1-mesa-dri libosmesa6 mesa-utils \
        libx11-6 libxext6 libxrandr2 libxinerama1 libxcursor1 libxi6 libxxf86vm1 libxkbcommon0 libxkbcommon-x11-0 \
    && rm -rf /var/lib/apt/lists/*

# В Ubuntu 24.04 системный pip закрыт (PEP 668), поэтому пакеты Python — в отдельном venv; он первый в PATH,
# так что `python3` в контейнере — это он. Версии проверены вместе со скриптами (API MuJoCo меняется от версии
# к версии — не поднимать без проверки `sim test`).
ENV VIRTUAL_ENV=/opt/venv PATH=/opt/venv/bin:$PATH
RUN python3 -m venv /opt/venv \
    && pip install --no-cache-dir mujoco==3.15.0 glfw==2.10.2 numpy==2.2.6 onnxruntime==1.23.2

WORKDIR /opt/gumanoid

# Модели качаются из GitHub на зафиксированных коммитах; берутся только нужные папки (sparse checkout).
# Zeroth-01 (K-Scale Z-Bot): MJCF + меши, обученные политики ходьбы (ONNX), записанные демо, позы/лимиты сервоприводов.
ARG ZEROTH_REPO=https://github.com/Justin-Riekehof/zeroth-01-build.git
ARG ZEROTH_COMMIT=4b47eef85a40348958974dd2f347989085bef4f8
RUN git clone --quiet --filter=blob:none --no-checkout "$ZEROTH_REPO" zeroth-01-build \
    && cd zeroth-01-build \
    && git sparse-checkout set --no-cone '/LICENSE' '/README.md' '/sim/assets/' '/sim/policies/' \
           '!/sim/policies/*/ckpt.bin' '/demos/' '/hardware/*.json' \
    && git checkout --quiet "$ZEROTH_COMMIT" \
    && rm -rf .git

# Модели Unitree из официального xr_teleoperate: G1 (g1_body29_hand14.xml: 29 суставов + кисти Dex3), H2 (H2.urdf),
# а также H1, H1-2, R1 и кисти Dex3/Inspire/BrainCo — их открывает универсальный загрузчик (sim robot h1 и т.п.).
ARG XR_REPO=https://github.com/unitreerobotics/xr_teleoperate.git
ARG XR_COMMIT=845b25a32f7febedf220e830952a7134897adb9d
RUN git clone --quiet --filter=blob:none --no-checkout "$XR_REPO" xr_teleoperate \
    && cd xr_teleoperate \
    && git sparse-checkout set --no-cone '/LICENSE' '/assets/' \
    && git checkout --quiet "$XR_COMMIT" \
    && rm -rf .git

COPY zbot_sliders.py zbot_walk.py zbot_demo.py h2_sliders.py g1_sliders.py robot_sliders.py ./
COPY examples/ examples/
COPY docker/sim docker/selftest.py docker/
RUN chmod 755 docker/sim && ln -s /opt/gumanoid/docker/sim /usr/local/bin/sim \
    && python3 -m compileall -q /opt/gumanoid/*.py /opt/gumanoid/examples /opt/gumanoid/docker/selftest.py

# PYTHONPATH: свои скрипты из /work могут делать `from robot_sliders import load_robot` и т.п.
# GUMANOID_MODELS: сюда run.sh монтирует папку models/ хоста — свои роботы.
ENV XR_TELEOP_ASSETS=/opt/gumanoid/xr_teleoperate/assets \
    GUMANOID_MODELS=/models \
    PYTHONPATH=/opt/gumanoid \
    NVIDIA_DRIVER_CAPABILITIES=all \
    LANG=C.UTF-8 LC_ALL=C.UTF-8 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1

LABEL org.opencontainers.image.title="gumanoid-sim" \
      org.opencontainers.image.description="MuJoCo 3.15: Unitree G1, Unitree H2, Zeroth-01 и любые URDF/MJCF с ползунками суставов (Ubuntu 24.04)"

# Рабочая папка: сюда viewer пишет скриншоты (Ctrl+P), Save xml и т.п.; run.sh монтирует в неё ./out хоста.
WORKDIR /work
ENTRYPOINT ["sim"]
CMD ["menu"]
