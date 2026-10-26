#!/usr/bin/env python3
"""Пример: управление роботом из своего Python-скрипта, окно MuJoCo открыто рядом.

Скрипт по очереди качает каждый сустав синусом вокруг исходной позы и раз в полсекунды печатает: какой привод
сейчас двигается, заданный и настоящий угол, момент привода, число контактов и силу давления на пол.
С --push раз в 4 секунды толкает корпус робота в бок. Это шаблон: замените функцию control() своей логикой.

    sim example script_control g1              # робот: g1, h2, zbot, h1, r1 … или путь к .urdf/.xml
    sim example script_control h2 --band --push
    sim example script_control g1 --headless --seconds 5    # без окна, быстрее реального времени

Скопируйте файл в out/ (в контейнере это /work), поправьте и запускайте как  sim python script_control.py g1
"""
import argparse
import sys
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # скрипты проекта (в образе — /opt/gumanoid)
from robot_sliders import load_robot  # noqa: E402

PERIOD = 2.0   # секунд на один сустав
FRAME = 1 / 60  # кадр окна, с


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("robot", nargs="?", default="g1", help="g1, h2, zbot, короткое имя или путь к модели")
    ap.add_argument("--free", action="store_true", help="на полу без опоры")
    ap.add_argument("--band", action="store_true", help="на полу на подвеске")
    ap.add_argument("--push", action="store_true", help="раз в 4 с толкать корпус силой 0.5·mg")
    ap.add_argument("--headless", action="store_true", help="без окна")
    ap.add_argument("--seconds", type=float, default=float("inf"), help="сколько секунд симулировать")
    args = ap.parse_args()

    # 1. Модель. MjModel — неизменяемое описание (массы, суставы, приводы), MjData — состояние (углы, скорости, силы).
    model = load_robot(args.robot, args.free, args.band).compile()
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)  # исходная поза — ключ "home"

    # 2. Имена -> индексы. Приводы (ползунки) — model.actuator(i), суставы — model.joint(name), тела — model.body(name).
    names = [model.actuator(i).name for i in range(model.nu)]
    home = data.ctrl.copy()
    lo, hi = model.actuator_ctrlrange.T
    root = 1  # тело №1 — корень робота (таз/база); №0 — мир
    weight = model.body_subtreemass[root] * np.linalg.norm(model.opt.gravity)
    print(f"{args.robot}: {model.nu} приводов, {model.body_subtreemass[root]:.1f} кг; приводы: {', '.join(names)}")

    def control(t: float) -> int:
        """Ваша логика: записать цели приводов в data.ctrl (для сервоприводов — углы в радианах)."""
        i = int(t // PERIOD) % model.nu
        data.ctrl[:] = home
        amp = 0.4 * (hi[i] - lo[i]) / 2
        data.ctrl[i] = np.clip(home[i] + amp * np.sin(2 * np.pi * (t % PERIOD) / PERIOD), lo[i], hi[i])
        # Внешняя сила на тело: xfrc_applied[тело] = [Fx, Fy, Fz, Mx, My, Mz] в мировых осях, действует пока не обнулить.
        pushing = args.push and t % 4.0 > 3.8
        data.xfrc_applied[root] = [0, 0.5 * weight if pushing else 0, 0, 0, 0, 0]
        return i

    def report(i: int) -> None:
        # actuator_length — положение того, что двигает привод (для сустава — угол), actuator_force — момент привода.
        q, tau = data.actuator_length[i], data.actuator_force[i]
        # Контакты: data.contact[:ncon]; сила каждого — mj_contactForce (нормаль первой компонентой).
        f = np.zeros(6)
        normal = 0.0
        for c in range(data.ncon):
            mujoco.mj_contactForce(model, data, c, f)
            normal += f[0]
        print(f"t={data.time:6.2f}  {names[i]:<26} цель {np.degrees(data.ctrl[i]):7.1f}°  факт {np.degrees(q):7.1f}°"
              f"  момент {tau:7.2f} Н·м   контактов {data.ncon:2d}, на пол {normal:7.1f} Н", flush=True)

    next_report = 0.0
    if args.headless:
        while data.time < args.seconds:
            i = control(data.time)
            mujoco.mj_step(model, data)  # один шаг физики (model.opt.timestep, обычно 2 мс)
            if data.time >= next_report:
                report(i)
                next_report += 0.5
        return

    # 3. Пассивное окно: физику крутит этот скрипт, окно только показывает. Мышь и клавиши окна работают
    #    (камера, Ctrl+правая — толкнуть тело), а ползунки Control перезаписываются функцией control().
    #    Окно перерисовывается ~60 раз в секунду, поэтому на кадр делается несколько шагов физики (8 × 2 мс).
    steps = max(1, round(FRAME / model.opt.timestep))
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running() and data.time < args.seconds:
            frame_start = time.time()
            with viewer.lock():  # окно читает data из своего потока — менять её только под замком
                for _ in range(steps):
                    i = control(data.time)
                    mujoco.mj_step(model, data)
            viewer.sync()  # передать новое состояние в окно (и забрать возмущения мышью)
            if data.time >= next_report:
                report(i)
                next_report += 0.5
            time.sleep(max(0.0, steps * model.opt.timestep - (time.time() - frame_start)))  # реальное время


if __name__ == "__main__":
    main()
