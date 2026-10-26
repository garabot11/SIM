#!/usr/bin/env python3
"""Ходьба Zeroth-01 в MuJoCo обученной политикой из репозитория (zeroth-01-build/sim/policies/*/policy.onnx).

Цикл повторяет обучение (sim/train/walking.py + ksim 0.0.31):
  * политика 50 Гц (ctrl_dt 0.02), физика 500 Гц (dt 0.002), Euler без EULERDAMP, CG-решатель, 6 итераций;
  * вход 64 = фаза походки [cos φ0, cos φ1, sin φ0, sin φ1], углы и скорости суставов, IMU (сырые оси чипа),
    команда скорости, частота шага, прошлое действие, курс [cos ψ, sin ψ];
  * выход 16 = целевые углы; сервопривод Feetech на каждом шаге физики: трапецеидальный планировщик
    (5 рад/с, 39 рад/с²) -> duty = kp·error_gain·Δq + kd·Δv, clip ±max_pwm -> момент = duty·vin·kt/R, clip ±max_torque.

Клавиши (окно своё, на GLFW: встроенный viewer MuJoCo забирает стрелки и буквы под свои горячие клавиши):
    ←/→ или A/D (держать) — поворот: смещает целевой курс, который политика держит;
    ↑/↓ или W/S (держать) — команда скорости;  пробел — пауза;  R/Backspace — заново;
    1–5 — сменить политику на ходу (от медленной к быстрой);  Tab/C — камера следит/свободна;  H — подсказка;  Esc — выход.
Остановиться на месте политики не умеют: их учили только идти вперёд (команда скорости > 0, награда за скорость
вперёд), поэтому «стоп» — это пауза.

Запуск:
    .venv/bin/python zbot_walk.py                               # политика по умолчанию
    .venv/bin/python zbot_walk.py --policy cad2_v10_calm2_step850
    .venv/bin/python zbot_walk.py --list                        # все политики и их скорость
"""
import argparse
import json
import time
from pathlib import Path

import glfw
import mujoco
import numpy as np
import onnxruntime as ort

from zbot_sliders import HERE, load_scene_spec, load_servo_params

REPO = HERE / "zeroth-01-build"
POLICIES = REPO / "sim" / "policies"
DEFAULT_POLICY = "cad2_v6_small_ft_step335"
DT, CTRL_DT = 0.002, 0.02
PLANNER_VMAX, PLANNER_AMAX = 5.0, 39.0  # trapezoidal_step в sim/train/common.py
FALL_HEIGHT = 0.18  # база ниже этого — робот упал (стоя ≈ 0.32 м)
TURN_RATE = np.radians(30)  # скорость поворота целевого курса, рад/с (резкий скачок курса политика не видела)
SPEED_RATE = 0.1  # как быстро меняется команда скорости при зажатой клавише, м/с за секунду
SPEED_LIMITS = (0.0, 0.5)  # проверено без падений; политики обучены на узком диапазоне > 0 (см. --list)


class Walker:
    def __init__(self, policy_dir: Path) -> None:
        self.meta = json.loads((policy_dir / "policy_meta.json").read_text())
        task = self.meta["task_config"]
        assets = REPO / "sim" / "assets" / self.meta["assets"]

        spec = load_scene_spec(assets)
        # Настройки физики из ZbotTask.get_mujoco_model
        spec.option.timestep = DT
        spec.option.iterations = 6
        spec.option.ls_iterations = 6
        spec.option.solver = mujoco.mjtSolver.mjSOL_CG
        spec.option.disableflags |= mujoco.mjtDisableBit.mjDSBL_EULERDAMP
        spec.visual.global_.azimuth = 135
        spec.visual.global_.elevation = -15
        spec.stat.extent = 0.6

        servos = load_servo_params(assets)
        for act in spec.actuators:
            p, joint = servos[act.target], spec.joint(act.target)
            joint.damping = [p["damping"], 0, 0]
            joint.armature = p["armature"]
            joint.frictionloss = p["frictionloss"]
            act.forcelimited = mujoco.mjtLimited.mjLIMITED_TRUE
            act.forcerange = [-p["max_torque"], p["max_torque"]]
        self.model = spec.compile()
        self.data = mujoco.MjData(self.model)
        m = self.model

        # Порядок действий политики = порядок суставов MuJoCo = порядок актуаторов; проверяем, а не предполагаем.
        joints = [m.joint(m.actuator_trnid[i, 0]).name for i in range(m.nu)]
        assert joints == self.meta["joint_order"], (joints, self.meta["joint_order"])
        self.qadr = m.jnt_qposadr[m.actuator_trnid[:, 0]]
        self.vadr = m.jnt_dofadr[m.actuator_trnid[:, 0]]
        arr = lambda key: np.array([float(servos[j][key]) for j in joints])
        nm_per_duty = arr("vin") * arr("kt") / arr("R")
        self.kp_eg, self.kd, self.max_pwm, self.nm_per_duty = arr("kp") * arr("error_gain"), arr("kd"), arr("max_pwm"), nm_per_duty
        self.acc_adr = m.sensor_adr[m.sensor("imu_acc").id]
        self.gyro_adr = m.sensor_adr[m.sensor("imu_gyro").id]
        self.base = m.body("base").id

        self.load_policy(policy_dir)
        self.reset()

    def load_policy(self, policy_dir: Path) -> None:
        """Загрузить ONNX-политику; модель робота общая (все политики обучены на одних и тех же assets)."""
        meta = json.loads((policy_dir / "policy_meta.json").read_text())
        if hasattr(self, "meta"):
            assert meta["assets"] == self.meta["assets"] and meta["joint_order"] == self.meta["joint_order"]
        self.meta, self.policy_name = meta, policy_dir.name
        task = meta["task_config"]
        self.session = ort.InferenceSession(str(policy_dir / "policy.onnx"), providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        self.n_in = meta["input"]["size"]
        self.heading_obs = any(x["name"] == "heading" for x in meta["input"]["layout_in_order"])
        self.action_scale = float(task.get("action_scale", 1.0))
        self.speed_range = (task["target_speed_min"], task["target_speed_max"])
        self.freq_range = (task["gait_freq_lower"], task["gait_freq_upper"])
        self.speed = float(np.mean(self.speed_range))
        self.freq = float(np.mean(self.freq_range))

    def switch_policy(self, policy_dir: Path) -> None:
        """Сменить политику на ходу: часы походки с нуля, прошлое действие = текущие углы, как после сброса."""
        self.load_policy(policy_dir)
        self.gait_t0 = self.data.time
        self.last_action = self.data.qpos[self.qadr].copy()

    def reset(self) -> None:
        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
        q = self.data.qpos[self.qadr].copy()
        self.plan_pos, self.plan_vel = q.copy(), np.zeros_like(q)  # get_default_state
        self.last_action = q.copy()  # get_default_action = текущие углы
        self.heading_ref = self.heading_target = 0.0
        self.gait_t0 = 0.0  # часы походки (TimestepPhaseObservation считает от начала эпизода)

    def yaw(self) -> float:
        w, x, y, z = self.data.qpos[3:7]
        return float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))

    def observation(self) -> np.ndarray:
        d = self.data
        phase = np.array([0.0, np.pi]) + 2 * np.pi * self.freq * (d.time - self.gait_t0)  # TimestepPhaseObservation
        phase = np.fmod(phase + np.pi, 2 * np.pi) - np.pi
        parts = [
            np.concatenate([np.cos(phase), np.sin(phase)]),
            d.qpos[self.qadr], d.qvel[self.vadr],
            d.sensordata[self.acc_adr:self.acc_adr + 3], d.sensordata[self.gyro_adr:self.gyro_adr + 3],
            [self.speed, 0.0], [0.0], [self.freq],
            self.last_action,
        ]
        if self.heading_obs:
            psi = self.yaw() - self.heading_ref
            parts.append([np.cos(psi), np.sin(psi)])
        obs = np.concatenate(parts).astype(np.float32)
        assert obs.shape == (self.n_in,), obs.shape
        return obs

    def servo_torque(self, target: np.ndarray) -> np.ndarray:
        # trapezoidal_step: профиль движения цели с ограничением скорости и ускорения
        err = target - self.plan_pos
        direction = np.sign(err)
        accelerate = np.abs(err) > self.plan_vel ** 2 / (2 * PLANNER_AMAX)
        vel = self.plan_vel + np.where(accelerate, direction, -direction) * PLANNER_AMAX * DT
        vel = np.clip(vel, -PLANNER_VMAX, PLANNER_VMAX)
        vel = np.where(direction * vel < 0, 0.0, vel)
        self.plan_pos, self.plan_vel = self.plan_pos + vel * DT, vel
        # FeetechActuators.get_stateful_ctrl
        d = self.data
        duty = self.kp_eg * (self.plan_pos - d.qpos[self.qadr]) + self.kd * (self.plan_vel - d.qvel[self.vadr])
        return np.clip(duty, -self.max_pwm, self.max_pwm) * self.nm_per_duty  # ±max_torque режет forcerange

    def step(self) -> None:
        """Один шаг политики (20 мс): наблюдение -> действие -> 10 шагов физики."""
        self.heading_ref += np.clip(self.heading_target - self.heading_ref, -TURN_RATE * CTRL_DT, TURN_RATE * CTRL_DT)
        obs = self.observation()
        action = self.session.run(None, {self.input_name: obs[None]})[0][0].astype(np.float64) * self.action_scale
        for _ in range(round(CTRL_DT / DT)):
            self.data.ctrl[:] = self.servo_torque(action)
            mujoco.mj_step(self.model, self.data)
        self.last_action = action

    def base_height(self) -> float:
        return float(self.data.xpos[self.base][2])

    def forward_speed(self) -> float:
        """Скорость базы вдоль направления, куда смотрит робот, м/с."""
        yaw = self.yaw()
        return float(self.data.qvel[0] * np.cos(yaw) + self.data.qvel[1] * np.sin(yaw))


def eval_speed(policy_dir: Path) -> float:
    """Средняя скорость политики из её eval.txt (замер автора), м/с."""
    line = next(l for l in (policy_dir / "eval.txt").read_text().splitlines() if "mean forward speed" in l)
    return float(line.split("mean forward speed")[1].split("m/s")[0])


def sorted_policies() -> list[Path]:
    """Все политики репозитория, от самой медленной к самой быстрой."""
    return sorted((p.parent for p in POLICIES.glob("*/policy.onnx")), key=eval_speed)


def list_policies() -> None:
    for i, p in enumerate(sorted_policies(), 1):
        task = json.loads((p / "policy_meta.json").read_text())["task_config"]
        print(f"  {i}  {p.name:<28} команда {task['target_speed_min']:.2f}–{task['target_speed_max']:.2f} м/с   "
              f"замер автора {eval_speed(p):.3f} м/с")


class KeyboardViewer:
    """Своё окно GLFW: в отличие от mujoco.viewer видит, какие клавиши зажаты (стрелки и буквы встроенный viewer
    забирает себе под горячие клавиши). Текст поверх сцены — только ASCII: шрифт MuJoCo кириллицу не рисует."""

    def __init__(self, w: Walker, policies: list[Path]) -> None:
        self.w, self.policies = w, policies
        self.paused, self.follow, self.show_help = False, True, True
        self.fallen_since: float | None = None
        self.speed_est = 0.0
        self.mouse = {"left": False, "right": False, "middle": False, "x": 0.0, "y": 0.0}

        if not glfw.init():
            raise RuntimeError("не удалось инициализировать GLFW")
        glfw.window_hint(glfw.SAMPLES, 4)
        mode = glfw.get_video_mode(glfw.get_primary_monitor())
        screen_w, screen_h = (mode.size.width, mode.size.height) if mode else (1600, 1000)
        self.win = glfw.create_window(int(screen_w * 0.75), int(screen_h * 0.75), "Zeroth-01 — ходьба с клавиатуры", None, None)
        if not self.win:
            glfw.terminate()
            raise RuntimeError("не удалось создать окно")
        glfw.make_context_current(self.win)
        glfw.swap_interval(1)

        m = w.model
        self.scn = mujoco.MjvScene(m, maxgeom=10000)
        self.opt = mujoco.MjvOption()
        self.cam = mujoco.MjvCamera()
        self.cam.distance, self.cam.azimuth, self.cam.elevation = 1.3, 135, -15
        self.set_follow(True)
        fs = mujoco.mjtFontScale
        font = fs.mjFONTSCALE_200 if screen_h >= 2000 else fs.mjFONTSCALE_150 if screen_h >= 1400 else fs.mjFONTSCALE_100
        self.ctx = mujoco.MjrContext(m, font)

        glfw.set_key_callback(self.win, self.on_key)
        glfw.set_mouse_button_callback(self.win, self.on_mouse_button)
        glfw.set_cursor_pos_callback(self.win, self.on_cursor)
        glfw.set_scroll_callback(self.win, self.on_scroll)

    # --- ввод
    def held(self, *keys: int) -> bool:
        return any(glfw.get_key(self.win, k) == glfw.PRESS for k in keys)

    def on_key(self, win, key: int, scancode: int, action: int, mods: int) -> None:
        if action != glfw.PRESS:
            return
        if key == glfw.KEY_SPACE:
            self.paused = not self.paused
        elif key in (glfw.KEY_R, glfw.KEY_BACKSPACE):
            self.restart()
        elif key in (glfw.KEY_TAB, glfw.KEY_C):  # C — на случай, если Tab перехватывает оконный менеджер
            self.set_follow(not self.follow)
        elif key == glfw.KEY_H:
            self.show_help = not self.show_help
        elif key == glfw.KEY_ESCAPE:
            glfw.set_window_should_close(win, True)
        elif glfw.KEY_1 <= key <= glfw.KEY_9 and key - glfw.KEY_1 < len(self.policies):
            self.w.switch_policy(self.policies[key - glfw.KEY_1])
            print(f"политика {self.w.policy_name}", flush=True)

    def on_mouse_button(self, win, button: int, action: int, mods: int) -> None:
        name = {glfw.MOUSE_BUTTON_LEFT: "left", glfw.MOUSE_BUTTON_RIGHT: "right", glfw.MOUSE_BUTTON_MIDDLE: "middle"}.get(button)
        if name:
            self.mouse[name] = action == glfw.PRESS
        self.mouse["x"], self.mouse["y"] = glfw.get_cursor_pos(win)

    def on_cursor(self, win, x: float, y: float) -> None:
        dx, dy = x - self.mouse["x"], y - self.mouse["y"]
        self.mouse["x"], self.mouse["y"] = x, y
        if not (self.mouse["left"] or self.mouse["right"] or self.mouse["middle"]):
            return
        shift = self.held(glfw.KEY_LEFT_SHIFT, glfw.KEY_RIGHT_SHIFT)
        A = mujoco.mjtMouse
        if self.mouse["right"]:
            action = A.mjMOUSE_MOVE_H if shift else A.mjMOUSE_MOVE_V
        elif self.mouse["left"]:
            action = A.mjMOUSE_ROTATE_H if shift else A.mjMOUSE_ROTATE_V
        else:
            action = A.mjMOUSE_ZOOM
        height = glfw.get_window_size(win)[1]
        mujoco.mjv_moveCamera(self.w.model, action, dx / height, dy / height, self.cam)

    def on_scroll(self, win, xoff: float, yoff: float) -> None:
        mujoco.mjv_moveCamera(self.w.model, mujoco.mjtMouse.mjMOUSE_ZOOM, 0, 0.05 * yoff, self.cam)  # колёсико вверх — ближе

    def set_follow(self, follow: bool) -> None:
        self.follow = follow
        if follow:
            self.cam.type, self.cam.trackbodyid = mujoco.mjtCamera.mjCAMERA_TRACKING, self.w.base
        else:
            self.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
            self.cam.lookat[:] = self.w.data.xpos[self.w.base]

    def restart(self) -> None:
        self.w.reset()
        self.fallen_since, self.speed_est = None, 0.0

    # --- симуляция
    def control_step(self) -> None:
        w = self.w
        turn = self.held(glfw.KEY_LEFT, glfw.KEY_A) - self.held(glfw.KEY_RIGHT, glfw.KEY_D)
        accel = self.held(glfw.KEY_UP, glfw.KEY_W) - self.held(glfw.KEY_DOWN, glfw.KEY_S)
        w.heading_target += turn * TURN_RATE * CTRL_DT
        w.speed = float(np.clip(w.speed + accel * SPEED_RATE * CTRL_DT, *SPEED_LIMITS))
        w.step()
        self.speed_est += 0.05 * (w.forward_speed() - self.speed_est)  # сглаживаем раскачку походки
        if w.base_height() < FALL_HEIGHT:  # упал — через секунду начинаем заново
            self.fallen_since = self.fallen_since if self.fallen_since is not None else w.data.time
            if w.data.time - self.fallen_since > 1.0:
                print("упал — перезапуск", flush=True)
                self.restart()
        else:
            self.fallen_since = None

    # --- отрисовка
    def render(self) -> None:
        w = self.w
        vp = mujoco.MjrRect(0, 0, *glfw.get_framebuffer_size(self.win))
        mujoco.mjv_updateScene(w.model, w.data, self.opt, None, self.cam, mujoco.mjtCatBit.mjCAT_ALL, self.scn)
        mujoco.mjr_render(vp, self.scn, self.ctx)

        lo, hi = w.speed_range
        status = "PAUSED" if self.paused else ("FALLEN" if self.fallen_since is not None else "WALKING")
        note = "" if lo - 1e-9 <= w.speed <= hi + 1e-9 else f"  (trained {lo:.2f}-{hi:.2f})"
        rows = [
            ("Policy", w.policy_name),
            ("State", status),
            ("Speed command", f"{w.speed:.2f} m/s{note}"),
            ("Actual speed", f"{self.speed_est:+.2f} m/s"),
            ("Heading target", f"{np.degrees(w.heading_target):+.0f} deg"),
            ("Heading actual", f"{np.degrees(w.yaw()):+.0f} deg"),
            ("Time", f"{w.data.time:.1f} s"),
        ]
        mujoco.mjr_overlay(mujoco.mjtFont.mjFONT_NORMAL, mujoco.mjtGridPos.mjGRID_TOPLEFT, vp,
                           "\n".join(r[0] for r in rows), "\n".join(r[1] for r in rows), self.ctx)
        if self.show_help:
            policies = [f"{i + 1}  {p.name}" + ("  <" if p.name == w.policy_name else "") for i, p in enumerate(self.policies)]
            keys = [("A/D, Left/Right", "turn (hold)"), ("W/S, Up/Down", "speed command (hold)"),
                    ("Space / R", "pause / restart"), ("1-%d" % len(self.policies), "policy (slow -> fast)"),
                    ("Tab", "camera follow on/off"), ("Mouse", "drag rotate, wheel zoom, R-drag pan (free cam)"), ("H / Esc", "help / quit")]
            mujoco.mjr_overlay(mujoco.mjtFont.mjFONT_NORMAL, mujoco.mjtGridPos.mjGRID_BOTTOMLEFT, vp,
                               "\n".join(k for k, _ in keys), "\n".join(v for _, v in keys), self.ctx)
            mujoco.mjr_overlay(mujoco.mjtFont.mjFONT_NORMAL, mujoco.mjtGridPos.mjGRID_TOPRIGHT, vp,
                               "Policies", "\n".join(policies), self.ctx)
        glfw.swap_buffers(self.win)

    def run(self) -> None:
        acc, last = 0.0, time.perf_counter()
        while not glfw.window_should_close(self.win):
            glfw.poll_events()
            now = time.perf_counter()
            acc = min(acc + now - last, 5 * CTRL_DT)  # если не успеваем — замедляемся, а не копим отставание
            last = now
            while acc >= CTRL_DT:
                acc -= CTRL_DT
                if not self.paused:
                    self.control_step()
            self.render()
        self.ctx.free()
        glfw.terminate()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--policy", default=DEFAULT_POLICY, help="папка в zeroth-01-build/sim/policies или путь к ней")
    ap.add_argument("--speed", type=float, default=None, help="команда скорости вперёд, м/с (по умолчанию середина диапазона обучения)")
    ap.add_argument("--freq", type=float, default=None, help="частота шага, Гц (по умолчанию середина диапазона обучения)")
    ap.add_argument("--list", action="store_true", help="показать доступные политики")
    args = ap.parse_args()

    if args.list:
        list_policies()
        return
    policy_dir = Path(args.policy) if Path(args.policy).is_dir() else POLICIES / args.policy
    w = Walker(policy_dir)
    if args.speed is not None:
        w.speed = args.speed
    if args.freq is not None:
        w.freq = args.freq
    print(f"политика {policy_dir.name}: скорость {w.speed:.3f} м/с (обучена на {w.speed_range[0]}–{w.speed_range[1]}), "
          f"частота шага {w.freq:.2f} Гц", flush=True)
    KeyboardViewer(w, sorted_policies()).run()


if __name__ == "__main__":
    main()
