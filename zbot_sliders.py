#!/usr/bin/env python3
"""Zeroth-01 (Z-Bot "Pixel") в MuJoCo с ползунками для 16 сервоприводов.

Берёт готовую модель из репозитория zeroth-01-build (sim/assets/<вариант>/robot.mjcf), добавляет пол и свет,
и превращает 16 torque-моторов в сервоприводы позиционного типа: ползунок = целевой угол сустава (рад).
Жёсткость и ограничение момента считаются из sys-ID-параметров Feetech из того же репозитория
(actuators/*.json + metadata.json), как в sim/train/common.py:
    duty = kp * error_gain * (q_target - q) - kd * dq,  clip ±max_pwm,  torque = duty * vin * kt / R,  clip ±max_torque

Ползунки: правая панель viewer'а -> раздел "Control". Сброс в исходную позу: Backspace. Пауза: пробел.

Запуск:
    .venv/bin/python zbot_sliders.py            # робот подвешен (база закреплена), ноги не касаются пола
    .venv/bin/python zbot_sliders.py --free     # робот стоит на полу и может упасть
    .venv/bin/python zbot_sliders.py --assets zeroth-01-build/sim/assets/zbot-pixel
"""
import argparse
import json
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

HERE = Path(__file__).resolve().parent
DEFAULT_ASSETS = HERE / "zeroth-01-build" / "sim" / "assets" / "zbot-cad"
HANG_HEIGHT = 0.12  # на сколько поднять базу над стартовой высотой в режиме подвеса, м


def load_servo_params(assets: Path) -> dict[str, dict]:
    """joint name -> параметры сервопривода (kp/kd из metadata.json + sys-ID из actuators/<type>.json)."""
    meta = json.loads((assets / "metadata.json").read_text())["joint_name_to_metadata"]
    out = {}
    for joint, jm in meta.items():
        sysid = json.loads((assets / "actuators" / f"{jm['actuator_type']}.json").read_text())
        out[joint] = {"kp": float(jm["kp"]), "kd": float(jm["kd"]), "type": jm["actuator_type"], "id": jm["id"], **sysid}
    return out


def load_scene_spec(assets: Path) -> mujoco.MjSpec:
    """robot.mjcf из репозитория + пол, свет и небо (то, что в обучении добавлял mujoco_scenes)."""
    # from_file не распознаёт расширение .mjcf, поэтому читаем XML сами; пути мешей — относительно папки модели
    spec = mujoco.MjSpec.from_string((assets / "robot.mjcf").read_text())
    spec.meshdir = str(assets)
    spec.option.timestep = 0.002

    # Сцена: пол с именем "floor" (на него ссылаются <contact><pair> модели), свет, небо.
    tex = spec.add_texture(name="grid", type=mujoco.mjtTexture.mjTEXTURE_2D, builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
                           rgb1=[0.22, 0.25, 0.30], rgb2=[0.30, 0.34, 0.40], mark=mujoco.mjtMark.mjMARK_EDGE,
                           markrgb=[0.5, 0.55, 0.6], width=512, height=512)
    spec.add_texture(name="sky", type=mujoco.mjtTexture.mjTEXTURE_SKYBOX, builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT,
                     rgb1=[0.45, 0.6, 0.8], rgb2=[0.05, 0.07, 0.1], width=512, height=3072)
    mat = spec.add_material(name="floor_mat", texrepeat=[8, 8], texuniform=True, reflectance=0.1)
    mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = tex.name
    world = spec.worldbody
    world.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[0, 0, 0.05], material="floor_mat")
    world.add_light(pos=[0, 0, 2.5], dir=[0, 0, -1], type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL,
                    diffuse=[0.6, 0.6, 0.6], castshadow=1)
    world.add_light(pos=[1.0, -1.0, 1.5], dir=[-1, 1, -1.2], diffuse=[0.3, 0.3, 0.3], castshadow=0)
    spec.visual.headlight.ambient = [0.35, 0.35, 0.35]
    spec.visual.headlight.diffuse = [0.4, 0.4, 0.4]
    spec.visual.global_.offwidth = 1920
    spec.visual.global_.offheight = 1080
    return spec


def build_spec(assets: Path, free: bool, kv: float | None) -> mujoco.MjSpec:
    spec = load_scene_spec(assets)
    servos = load_servo_params(assets)
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST  # демпфирование сервопривода считается неявно

    base = spec.body("base")
    if not free:
        spec.delete(spec.joint("floating_base"))  # база жёстко прикреплена к миру — робот «висит на стенде»
        base.pos = [base.pos[0], base.pos[1], base.pos[2] + HANG_HEIGHT]

    # Стартовая камера viewer'а: спереди-сбоку, робот на весь кадр.
    spec.visual.global_.azimuth = 150
    spec.visual.global_.elevation = -15
    spec.stat.center = [0, 0, base.pos[2] - 0.08]
    spec.stat.extent = 0.5

    # Моторы (сила = ctrl) -> сервоприводы позиционного типа (ctrl = целевой угол).
    for act in spec.actuators:
        joint = spec.joint(act.target)
        p = servos[act.target]
        nm_per_duty = p["vin"] * p["kt"] / p["R"]
        kp_eff = p["kp"] * p["error_gain"] * nm_per_duty
        # kd=3 из metadata в обучении работает вместе с трапецеидальным планировщиком (feed-forward по скорости);
        # без него он даёт τ = kd/(kp·error_gain) ≈ 1.1 с. Поэтому по умолчанию демпфирование близко к критическому.
        kv_eff = kv if kv is not None else 2.0 * np.sqrt(kp_eff * p["armature"])
        max_force = min(p["max_torque"], p["max_pwm"] * nm_per_duty)

        act.set_to_position(kp=kp_eff, kv=kv_eff)
        act.ctrllimited = mujoco.mjtLimited.mjLIMITED_TRUE
        act.ctrlrange = joint.range
        act.forcelimited = mujoco.mjtLimited.mjLIMITED_TRUE
        act.forcerange = [-max_force, max_force]

        joint.damping = [p["damping"], 0, 0]  # в MuJoCo 3.15 damping — вектор (линейный коэффициент первым)
        joint.armature = p["armature"]
        joint.frictionloss = p["frictionloss"]

    model = spec.compile()

    # Исходная поза: все суставы в нуле, цели ползунков тоже 0 (с учётом ограничений суставов).
    qpos0 = model.qpos0.copy()
    ctrl0 = np.clip(np.zeros(model.nu), model.actuator_ctrlrange[:, 0], model.actuator_ctrlrange[:, 1])
    spec.add_key(name="home", qpos=qpos0, ctrl=ctrl0)
    return spec


def build_model(assets: Path, free: bool, kv: float | None) -> mujoco.MjModel:
    return build_spec(assets, free, kv).compile()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--assets", type=Path, default=DEFAULT_ASSETS, help="папка с robot.mjcf, metadata.json, actuators/")
    ap.add_argument("--free", action="store_true", help="свободная база: робот стоит на полу (без опоры может упасть)")
    ap.add_argument("--kv", type=float, default=None, help="демпфирование сервопривода, Н·м·с/рад (по умолчанию ≈ критическое)")
    ap.add_argument("--save-xml", type=Path, default=None, help="сохранить собранную сцену в MJCF и выйти")
    args = ap.parse_args()

    spec = build_spec(args.assets.resolve(), args.free, args.kv)
    model = spec.compile()
    if args.save_xml:
        args.save_xml.write_text(spec.to_xml())  # пути мешей абсолютные (meshdir), файл можно открыть откуда угодно
        print("saved", args.save_xml)
        return

    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)

    print(f"{model.nu} сервоприводов. Ползунки: правая панель -> Control. Backspace — сброс позы, пробел — пауза.")
    for i in range(model.nu):
        lo, hi = np.degrees(model.actuator_ctrlrange[i])
        print(f"  {model.actuator(i).name:<26} {lo:7.1f}° .. {hi:6.1f}°   max {model.actuator_forcerange[i, 1]:.2f} Н·м")

    mujoco.viewer.launch(model, data)


if __name__ == "__main__":
    main()
