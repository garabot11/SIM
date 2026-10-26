#!/usr/bin/env python3
"""Пример: управление симулятором из другой программы по сети (UDP, сообщения в JSON).

Симулятор слушает UDP-порт и принимает команды; тем, кто прислал хоть одну команду, он шлёт состояние
робота с частотой --rate. Клиент может быть на любом языке и где угодно: на хосте, в другом контейнере, на
другом компьютере (ROS 2-нода, контроллер, телеуправление). Пример клиента — examples/udp_client.py.

    NET=host ./run.sh example udp_bridge g1           # на хосте: окно G1, порт 9870
    python3 examples/udp_client.py --joint left_elbow  # на хосте, в другом терминале

Команды (один JSON-объект в датаграмме, поля можно сочетать):
    {"info": true}                          ответ: {"robot", "actuators": [имена], "ctrlrange": [[lo, hi], ...], "rate"}
    {"ctrl": {"left_elbow": 0.5, ...}}      цели приводов по именам (рад или м; выход за пределы обрезается)
    {"ctrl": [0.0, 0.1, ...]}               все цели сразу, по порядку actuators
    {"reset": true}                         исходная поза
    {"push": [Fx, Fy, Fz]}                  толкнуть корень робота на 0.1 с (Н)
Состояние: {"t", "q": [углы приводов], "dq": [скорости], "tau": [моменты], "base_pos": [x, y, z],
            "base_quat": [w, x, y, z], "ncon"}  — порядок как в actuators.
"""
import argparse
import json
import socket
import sys
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # скрипты проекта (в образе — /opt/gumanoid)
from robot_sliders import load_robot  # noqa: E402


class Bridge:
    def __init__(self, robot: str, free: bool, band: bool, port: int, rate: float):
        self.robot = robot
        self.model = load_robot(robot, free, band).compile()
        self.data = mujoco.MjData(self.model)
        self.home = self.model.key("home").id
        mujoco.mj_resetDataKeyframe(self.model, self.data, self.home)
        self.names = [self.model.actuator(i).name for i in range(self.model.nu)]
        self.index = {n: i for i, n in enumerate(self.names)}
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(("0.0.0.0", port))
        self.sock.setblocking(False)
        self.clients: set[tuple[str, int]] = set()
        self.period = 1.0 / rate
        self.next_send = 0.0
        self.push_until = -1.0

    def handle(self, msg: dict, addr: tuple[str, int]) -> None:
        m, d = self.model, self.data
        self.clients.add(addr)
        if msg.get("info"):
            reply = {"robot": self.robot, "actuators": self.names, "ctrlrange": m.actuator_ctrlrange.tolist(),
                     "rate": 1.0 / self.period}
            self.sock.sendto(json.dumps(reply).encode(), addr)
        if msg.get("reset"):
            mujoco.mj_resetDataKeyframe(m, d, self.home)
        ctrl = msg.get("ctrl")
        if isinstance(ctrl, dict):
            for name, value in ctrl.items():
                if name in self.index:
                    d.ctrl[self.index[name]] = value
        elif isinstance(ctrl, list) and len(ctrl) == m.nu:
            d.ctrl[:] = ctrl
        if ctrl is not None:
            lo, hi = m.actuator_ctrlrange.T
            d.ctrl[:] = np.where(m.actuator_ctrllimited, np.clip(d.ctrl, lo, hi), d.ctrl)
        if "push" in msg:
            d.xfrc_applied[1, :3] = msg["push"]
            self.push_until = d.time + 0.1

    def state(self) -> bytes:
        d = self.data
        return json.dumps({"t": round(d.time, 4), "q": d.actuator_length.round(5).tolist(),
                           "dq": d.actuator_velocity.round(5).tolist(), "tau": d.actuator_force.round(4).tolist(),
                           "base_pos": d.xpos[1].round(5).tolist(), "base_quat": d.xquat[1].round(5).tolist(),
                           "ncon": int(d.ncon)}).encode()

    def step(self) -> None:
        """Принять все пришедшие команды, сделать шаг физики, по таймеру разослать состояние."""
        while True:
            try:
                raw, addr = self.sock.recvfrom(65536)
            except BlockingIOError:
                break
            try:
                self.handle(json.loads(raw), addr)
            except (ValueError, TypeError, AttributeError) as e:
                self.sock.sendto(json.dumps({"error": str(e)}).encode(), addr)
        if self.data.time > self.push_until:
            self.data.xfrc_applied[1, :3] = 0
        mujoco.mj_step(self.model, self.data)
        if self.data.time >= self.next_send and self.clients:
            packet = self.state()
            for addr in self.clients:
                self.sock.sendto(packet, addr)
            self.next_send = self.data.time + self.period


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("robot", nargs="?", default="g1", help="g1, h2, zbot, короткое имя или путь к модели")
    ap.add_argument("--free", action="store_true", help="на полу без опоры")
    ap.add_argument("--band", action="store_true", help="на полу на подвеске")
    ap.add_argument("--port", type=int, default=9870, help="UDP-порт (по умолчанию 9870)")
    ap.add_argument("--rate", type=float, default=50.0, help="как часто слать состояние, Гц")
    ap.add_argument("--headless", action="store_true", help="без окна")
    ap.add_argument("--seconds", type=float, default=float("inf"), help="сколько секунд работать")
    args = ap.parse_args()

    bridge = Bridge(args.robot, args.free, args.band, args.port, args.rate)
    print(f"udp_bridge: {args.robot}, {bridge.model.nu} приводов, порт {args.port}/udp. Приводы: {', '.join(bridge.names)}",
          flush=True)
    start = time.time()
    if args.headless:
        while time.time() - start < args.seconds:
            bridge.step()
            lag = bridge.data.time - (time.time() - start)
            if lag > 0:
                time.sleep(lag)
        return
    steps = max(1, round((1 / 60) / bridge.model.opt.timestep))  # шагов физики на кадр окна (~60 к/с)
    with mujoco.viewer.launch_passive(bridge.model, bridge.data) as viewer:
        while viewer.is_running() and time.time() - start < args.seconds:
            frame_start = time.time()
            with viewer.lock():  # окно читает data из своего потока — менять её только под замком
                for _ in range(steps):
                    bridge.step()
            viewer.sync()
            time.sleep(max(0.0, steps * bridge.model.opt.timestep - (time.time() - frame_start)))


if __name__ == "__main__":
    main()
