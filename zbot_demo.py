#!/usr/bin/env python3
"""Записанные движения Zeroth-01 (отжимания, колено, взмах рукой, шаги) в MuJoCo с физикой.

Демо — это не обученные политики, а последовательности поз из zeroth-01-build/demos/*.json, которые на настоящем
роботе проигрывает src/zbot_core/zbot_core/motion.py (play_demo). Здесь то же самое, только вместо шины Feetech — симулятор:
  * углы в демо — градусы сервопривода от центра; модель zbot-cad построена так, что угол сустава в симуляторе равен
    углу сервопривода (model_invert и model_zero_offsets уже учтены в robot.mjcf), поэтому рад = град·π/180;
  * шаг-«центр» (center: true или все углы ровно 0) идёт в позу из hardware/center_pose.json;
  * цели обрезаются по пределам суставов, speed/acc — по hardware/motion_limits.json (300 тиков/с, 30·100 тиков/с²);
  * сервопривод: трапецеидальный профиль со скоростью и ускорением шага + модель Feetech из zbot_walk.py;
    сервоприводы шага стартуют с задержкой stagger_ms друг за другом, как на роботе;
  * шаг выполнен, когда все суставы ближе 25 тиков (~2.2°) к цели, иначе таймаут max|Δ|/speed + 2 с;
  * на шагах с паузой и на последнем — подстройка под провисание от нагрузки (settle, до двух раз).
В robot.mjcf с полом сталкиваются только подошвы, поэтому для отжиманий и колена к полу добавлены выпуклые оболочки
всех звеньев (руки, колени, корпус, рюкзак). Друг с другом звенья не сталкиваются.

Робот стартует в позе из center_pose.json (бёдра ≈17°, корпус наклонён вперёд), подошвами на полу: так он стоит перед
демо. В нулевой позе из-за рюкзака центр масс всего в 1.5 см от края пятки, и первое же сгибание бёдер валит его назад.

Что получается (2026-10-13, --headless):
    push_ups      работает: падает вперёд на руки, 5 отжиманий (корпус 0.174 <-> 0.119 м), как на гифке из README;
                  на шагах отжиманий провисание 2–3° под нагрузкой, поэтому шаг ждёт таймаут (~5 с вместо ~3)
    wave          работает
    kneeing       с рюкзаком падает назад на рюкзак на шаге 5; с --no-backpack проходит целиком
    walk_0        падает и с рюкзаком, и без (большой шаг без баланса)
    careful_walk  падает и с рюкзаком (назад на шаге 3), и без

Запуск:
    .venv/bin/python zbot_demo.py                       # отжимания
    .venv/bin/python zbot_demo.py wave                  # другие: kneeing, wave, walk_0, careful_walk
    .venv/bin/python zbot_demo.py --list
    .venv/bin/python zbot_demo.py kneeing --no-backpack # без рюкзака (минус 0.57 кг сзади)
    .venv/bin/python zbot_demo.py push_ups --until 3    # только первые 3 шага, как «▶ to here» в GUI
    .venv/bin/python zbot_demo.py push_ups --headless   # без окна, итог по шагам в консоль
    .venv/bin/python zbot_demo.py wave --start zero     # старт из нулевой позы вместо стойки
В окне: пробел — пауза. После последнего шага робот держит позу, пока окно открыто.
"""
import argparse
import json
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

from zbot_sliders import HERE, load_scene_spec, load_servo_params

REPO = HERE / "zeroth-01-build"
ASSETS = REPO / "sim" / "assets" / "zbot-cad"
DEMOS = REPO / "demos"
HW = REPO / "hardware"
DT = 0.002
TICK = 2 * np.pi / 4096  # рад на тик энкодера Feetech
TOLERANCE = 25 * TICK  # motion.py: цель достигнута
POLL = 0.04  # motion.py: период опроса позиций
SETTLE_MIN, SETTLE_MAX = 10 * TICK, 300 * TICK  # ошибка такого размера — провисание, а не упор
SETTLE_SPEED, SETTLE_ACC, SETTLE_WAIT = 150, 30, 0.45  # медленная подстройка, как в motion.py
START_HOLD = 1.0  # сколько секунд стоять в нулевой позе перед первым шагом


def add_floor_contacts(spec: mujoco.MjSpec) -> None:
    """Все меши робота (выпуклая оболочка) касаются пола через явные <pair>, как подошвы в robot.mjcf.
    Визуальные геомы остаются contype=conaffinity=0, так что звенья друг с другом не сталкиваются."""
    for g in spec.geoms:
        if g.type == mujoco.mjtGeom.mjGEOM_MESH:
            spec.add_pair(geomname1=g.name, geomname2="floor")


class DemoSim:
    def __init__(self, start: str = "center", backpack: bool = True) -> None:
        spec = load_scene_spec(ASSETS)
        if not backpack:
            spec.delete(spec.body("backpack"))  # 0.57 кг сзади: аккумулятор, Pi, корпус рюкзака
        add_floor_contacts(spec)
        spec.visual.global_.azimuth = 135
        spec.visual.global_.elevation = -15
        spec.stat.extent = 0.6
        servos = load_servo_params(ASSETS)
        for act in spec.actuators:  # те же параметры сервоприводов, что в zbot_walk.py
            p, joint = servos[act.target], spec.joint(act.target)
            joint.damping = [p["damping"], 0, 0]
            joint.armature = p["armature"]
            joint.frictionloss = p["frictionloss"]
            act.forcelimited = mujoco.mjtLimited.mjLIMITED_TRUE
            act.forcerange = [-p["max_torque"], p["max_torque"]]
        self.model = m = spec.compile()
        self.data = mujoco.MjData(m)

        jid = m.actuator_trnid[:, 0]
        self.joints = [m.joint(j).name for j in jid]
        self.index = {name: i for i, name in enumerate(self.joints)}
        self.qadr, self.vadr = m.jnt_qposadr[jid], m.jnt_dofadr[jid]
        self.lo, self.hi = m.jnt_range[jid].T
        arr = lambda key: np.array([float(servos[j][key]) for j in self.joints])
        self.kp_eg, self.kd, self.max_pwm = arr("kp") * arr("error_gain"), arr("kd"), arr("max_pwm")
        self.nm_per_duty = arr("vin") * arr("kt") / arr("R")
        self.base = m.body("base").id
        self.floor = m.geom("floor").id

        limits = json.loads((HW / "motion_limits.json").read_text())
        self.max_speed, self.max_acc = limits["max_speed"], limits["max_acc"]
        self.stagger = limits.get("stagger_ms", 0) / 1000
        self.center = json.loads((HW / "center_pose.json").read_text())
        self.reset(start)

    def reset(self, start: str = "center") -> None:
        """start="center": стоять в позе из center_pose.json, подошвы ровно на полу (так робот стоит перед демо;
        с рюкзаком в нулевой позе центр масс в 1.5 см от края пятки, и первое же сгибание бёдер валит его назад).
        start="zero": все суставы в 0, как в robot.mjcf."""
        m, d = self.model, self.data
        mujoco.mj_resetData(m, d)
        if start == "center":
            for j, deg in self.center.items():
                k = self.index[j]
                d.qpos[self.qadr[k]] = np.clip(np.radians(deg), self.lo[k], self.hi[k])
            mujoco.mj_forward(m, d)
            Rf = d.xmat[m.body("left_ankle_pitch_link").id].reshape(3, 3)
            tilt = np.zeros(4)  # повернуть базу вокруг y так, чтобы стопа стала горизонтальной
            mujoco.mju_axisAngle2Quat(tilt, [0, 1, 0], np.arctan2(Rf[2, 0], Rf[2, 2]))
            quat = d.qpos[3:7].copy()
            mujoco.mju_mulQuat(d.qpos[3:7], tilt, quat)
            mujoco.mj_forward(m, d)
            d.qpos[2] += 0.001 - min(d.site_xpos[m.site(n).id][2] for n in ("left_foot", "right_foot"))
        mujoco.mj_forward(m, d)
        self.goal = self.q().copy()
        self.plan_pos, self.plan_vel = self.goal.copy(), np.zeros(self.model.nu)
        self.vmax = np.full(self.model.nu, self.max_speed * TICK)
        self.amax = np.full(self.model.nu, self.max_acc * 100 * TICK)

    def q(self) -> np.ndarray:
        return self.data.qpos[self.qadr]

    # ---- сервопривод ------------------------------------------------------------------------------------------------
    def command(self, i: int, target: float, speed: int, acc: int) -> None:
        """Как запись цели в Feetech: позиция + скорость (тик/с) + ускорение (100 тик/с²)."""
        self.goal[i], self.vmax[i], self.amax[i] = target, speed * TICK, acc * 100 * TICK

    def physics_step(self) -> None:
        # trapezoidal_step из sim/train/common.py, но скорость и ускорение у каждого сустава свои (из шага демо)
        err = self.goal - self.plan_pos
        direction = np.sign(err)
        accelerate = np.abs(err) > self.plan_vel ** 2 / (2 * self.amax)
        vel = self.plan_vel + np.where(accelerate, direction, -direction) * self.amax * DT
        vel = np.clip(vel, -self.vmax, self.vmax)
        vel = np.where(direction * vel < 0, 0.0, vel)
        self.plan_pos, self.plan_vel = self.plan_pos + vel * DT, vel
        d = self.data
        duty = self.kp_eg * (self.plan_pos - self.q()) + self.kd * (self.plan_vel - d.qvel[self.vadr])
        d.ctrl[:] = np.clip(duty, -self.max_pwm, self.max_pwm) * self.nm_per_duty  # ±max_torque режет forcerange
        mujoco.mj_step(self.model, d)

    def wait(self, seconds: float):
        for _ in range(max(1, round(seconds / DT))):
            self.physics_step()
            yield

    # ---- состояние робота для отчёта --------------------------------------------------------------------------------
    def pose_report(self) -> str:
        d = self.data
        z = d.xpos[self.base][2]
        R = d.xmat[self.base].reshape(3, 3)
        lean = np.degrees(np.arcsin(np.clip(-R[2, 0], -1, 1)))  # + вперёд, − назад
        tilt = np.degrees(np.arccos(np.clip(R[2, 2], -1, 1)))
        touching = set()
        for c in d.contact[:d.ncon]:
            other = c.geom2 if c.geom1 == self.floor else c.geom1 if c.geom2 == self.floor else None
            if other is not None:
                touching.add(self.model.body(self.model.geom_bodyid[other]).name.removesuffix("_link"))
        return (f"корпус: высота {z:.3f} м, наклон {tilt:4.0f}° (вперёд {lean:+4.0f}°); "
                f"пола касаются: {', '.join(sorted(touching)) or 'ничего'}")

    # ---- проигрывание демо, повторяет MotionEngine.play_demo ---------------------------------------------------------
    def play(self, demo: dict, log, until: int | None = None):
        """Генератор: один next() = один шаг физики. После последнего шага держит позу бесконечно."""
        steps = demo["steps"][:until] if until else demo["steps"]
        n = len(steps)
        self.finished, self.all_reached = False, True
        unknown = sorted({j for s in steps for j in s["angles"]} - set(self.index))
        if unknown:
            log(f"неизвестные суставы пропущены: {', '.join(unknown)}")
        log(f"--- демо '{demo['name']}': {n} шагов ---")
        yield from self.wait(START_HOLD)

        for i, step in enumerate(steps, 1):
            speed, acc = int(step.get("speed", 500)), int(step.get("acc", 50))  # умолчания DemoStep
            speed = min(speed, self.max_speed)
            acc = self.max_acc if acc == 0 or acc > self.max_acc else acc
            angles = step["angles"]
            is_center = bool(step.get("center")) or (bool(angles) and all(abs(v) < 1e-6 for v in angles.values()))
            label = f"шаг {i}/{n}" + (f" ({step['title']})" if step.get("title") else "") + (" [центр]" if is_center else "")
            targets, clamped = {}, []
            for j, deg in angles.items():
                if j not in self.index:
                    continue
                if is_center:
                    deg = self.center.get(j, 0.0)
                k = self.index[j]
                rad = float(np.clip(np.radians(deg), self.lo[k], self.hi[k]))
                if abs(rad - np.radians(deg)) > 1e-9:
                    clamped.append(f"{j} {deg:+.1f}->{np.degrees(rad):+.1f}")
                targets[k] = rad
            if clamped:
                log(f"{label}: обрезано по пределам суставов: {', '.join(clamped)}")

            t0 = self.data.time
            missed = yield from self.move_all(targets, speed, acc)
            if missed:
                log(f"{label}: НЕ ВСЕ ЦЕЛИ ДОСТИГНУТЫ за {self.data.time - t0:.1f} с: {', '.join(missed)}")
            settle = step.get("settle")
            if settle if settle is not None else (bool(step.get("pause_s")) or i == n):
                resid = yield from self.settle(targets)
                if resid:
                    log(f"{label}: после подстройки осталась ошибка позы: {', '.join(resid)}")
            self.all_reached &= not missed
            log(f"{label}: {'цели достигнуты' if not missed else 'таймаут'} за {self.data.time - t0:.1f} с; "
                + self.pose_report())
            if step.get("pause_s"):
                yield from self.wait(step["pause_s"])

        log(f"демо '{demo['name']}' закончено — робот держит последнюю позу")
        self.finished = True
        while True:
            yield from self.wait(1.0)

    def move_all(self, targets: dict[int, float], speed: int, acc: int):
        """_move_all_and_wait: команды с задержкой stagger, потом ждать, пока все суставы не дойдут, или таймаут."""
        start = self.q().copy()
        for n, (k, t) in enumerate(targets.items()):
            if n and self.stagger > 0:
                yield from self.wait(self.stagger)
            self.command(k, t, speed, acc)
        timeout = max((abs(t - start[k]) for k, t in targets.items()), default=0.0) / (speed * TICK) + 2.0
        t0 = self.data.time
        while True:
            missed = self.misses(targets)
            if not missed or self.data.time - t0 > timeout:
                return missed  # пусто = все дошли
            yield from self.wait(POLL)

    def misses(self, targets: dict[int, float]) -> list[str]:
        """Суставы дальше 25 тиков от цели: «имя ошибка°» (+ — перешёл цель по своей оси, − — не дошёл)."""
        q = self.q()
        return [f"{self.joints[k]} {np.degrees(q[k] - t):+.1f}°" for k, t in targets.items() if abs(q[k] - t) > TOLERANCE]

    def settle(self, targets: dict[int, float]):
        """Компенсация провисания: под нагрузкой P-регулятор не доходит до цели — командуем цель + ошибка."""
        for _ in range(2):
            q = self.q()
            trims = {}
            for k, t in targets.items():
                err = t - q[k]
                if SETTLE_MIN < abs(err) <= SETTLE_MAX:
                    cmd = float(np.clip(t + err, self.lo[k], self.hi[k]))
                    if cmd != t:
                        trims[k] = cmd
            if not trims:
                break
            for k, cmd in trims.items():
                self.command(k, cmd, SETTLE_SPEED, SETTLE_ACC)
            yield from self.wait(SETTLE_WAIT)
        return self.misses(targets)


def load_demo(name: str) -> dict:
    path = Path(name) if name.endswith(".json") else DEMOS / f"{name}.json"
    return json.loads(path.read_text())


def list_demos() -> None:
    for path in sorted(DEMOS.glob("*.json")):
        d = json.loads(path.read_text())
        print(f"  {path.stem:<14} {len(d['steps']):3d} шагов  {d.get('description', '')[:90]}")


def run_headless(sim: DemoSim, demo: dict, until: int | None, hold: float) -> None:
    player = sim.play(demo, print, until)
    while not getattr(sim, "finished", False):
        next(player)
    for _ in range(round(hold / DT)):
        next(player)
    print(f"через {hold:.0f} с удержания: {sim.pose_report()}")


def run_viewer(sim: DemoSim, demo: dict, until: int | None) -> None:
    m, d = sim.model, sim.data
    paused = [False]

    def on_key(key: int) -> None:
        if key == 32:  # пробел (другие клавиши встроенный viewer забирает себе)
            paused[0] = not paused[0]

    status = ["start"]

    def log(msg: str) -> None:
        print(msg)
        if msg.startswith("шаг"):
            status[0] = msg.split(":")[0].replace("шаг", "step")
        elif msg.startswith("демо") and "закончено" in msg:
            status[0] = "done, holding final pose"

    player = sim.play(demo, log, until)
    next(player)
    with mujoco.viewer.launch_passive(m, d, key_callback=on_key) as v:
        v.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        v.cam.trackbodyid = sim.base
        v.cam.distance, v.cam.azimuth, v.cam.elevation = 1.0, 135, -15
        wall0, sim0 = time.perf_counter(), d.time
        while v.is_running():
            if paused[0]:
                wall0, sim0 = time.perf_counter(), d.time
            else:
                target = sim0 + time.perf_counter() - wall0
                with v.lock():
                    for _ in range(50):  # не больше 0.1 с физики за кадр; если не успеваем — замедляемся
                        if d.time >= target:
                            break
                        next(player)
                if d.time < target - 0.1:
                    wall0, sim0 = time.perf_counter(), d.time
            if hasattr(v, "set_texts"):  # оверлей только ASCII
                v.set_texts((mujoco.mjtFontScale.mjFONTSCALE_150, mujoco.mjtGridPos.mjGRID_TOPLEFT,
                             f"{demo['name']}  t={d.time:5.1f}s", status[0] + ("  [PAUSED]" if paused[0] else "")))
            v.sync()
            time.sleep(0.005)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("demo", nargs="?", default="push_ups", help="имя из zeroth-01-build/demos или путь к .json")
    ap.add_argument("--list", action="store_true", help="показать все демо")
    ap.add_argument("--until", type=int, default=None, help="проиграть только первые N шагов")
    ap.add_argument("--headless", action="store_true", help="без окна: прогнать демо и вывести итог по шагам")
    ap.add_argument("--start", choices=["center", "zero"], default="center",
                    help="стартовая поза: center — стойка робота из center_pose.json (по умолчанию), zero — все суставы в 0")
    ap.add_argument("--no-backpack", action="store_true", help="убрать рюкзак (аккумулятор, Pi) — робот уже не перевешивает назад")
    ap.add_argument("--hold", type=float, default=3.0, help="сколько секунд держать финальную позу в --headless")
    args = ap.parse_args()
    if args.list:
        list_demos()
        return
    demo = load_demo(args.demo)
    if args.until is not None and not 1 <= args.until <= len(demo["steps"]):
        ap.error(f"--until: в демо {len(demo['steps'])} шагов")
    sim = DemoSim(args.start, backpack=not args.no_backpack)
    if args.headless:
        run_headless(sim, demo, args.until, args.hold)
    else:
        run_viewer(sim, demo, args.until)


if __name__ == "__main__":
    main()
