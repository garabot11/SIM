#!/usr/bin/env python3
"""Unitree H2 в MuJoCo с ползунками для 31 сустава.

Модель — H2.urdf из xr_teleoperate (assets/h2: URDF + STL-меши, 67 кг). К ней добавляются пол, свет и
31 сервопривод позиционного типа: ползунок = целевой угол сустава (рад), ctrlrange = пределы сустава из URDF.
    τ = kp·(q_target − q) − kv·dq,  clip ±effort (из URDF)
kp — штатные коэффициенты Unitree из unitree_ros2/example/src/src/h2/low_level/h2_ankle_swing_example.cpp.
Штатные Kd рассчитаны на реальные моторы (большая инерция ротора, трение); в симуляции с ними суставы
раскачиваются, поэтому по умолчанию kv ≈ критическое демпфирование (--unitree-kd — взять Kd как есть).
В режиме подвеса включена компенсация силы тяжести через приводы (как feed-forward у реального контроллера),
иначе запястья (kp 4) и поясница провисают на 6–16° (--no-gravcomp — увидеть провисание).
Самостолкновения выключены (меши звеньев перекрываются уже в нулевой позе); с полом сталкиваются все звенья.

Ползунки: правая панель viewer'а -> раздел "Control". Сброс в исходную позу: Backspace. Пауза: пробел.

Запуск:
    .venv/bin/python h2_sliders.py            # робот подвешен (таз закреплён), ноги не касаются пола
    .venv/bin/python h2_sliders.py --free     # робот на полу; со штатными Kp без контроллера баланса падает вперёд за ~2 с
    .venv/bin/python h2_sliders.py --band     # робот на полу на мягкой подвеске (выключается в панели Equality)
Папку assets репозитория xr_teleoperate можно задать переменной окружения XR_TELEOP_ASSETS.
"""
import argparse
import os
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

XR_ASSETS = Path(os.environ.get("XR_TELEOP_ASSETS", Path.home() / "xr_teleoperate" / "assets"))
DEFAULT_URDF = XR_ASSETS / "h2" / "H2.urdf"
STAND_HEIGHT = 1.045  # высота таза, при которой подошвы (нижняя точка меша стопы −1.0388 м) чуть над полом
HANG_HEIGHT = 0.15    # на сколько поднять таз в режиме подвеса, м
BAND_DROP = 0.0065   # с подвеской таз ниже на 6.5 мм: стопы стоят на полу и несут около половины веса
BAND_SOLREF = [0.02, 1.0]  # подвеска: постоянная времени, с (при 0.05 G1 «уплывает» при приседе), демпфирование

# Штатные Kp/Kd Unitree H2 (порядок моторов = порядок суставов ниже).
UNITREE_GAINS = {
    "left_hip_pitch_joint": (150, 2.0), "left_hip_roll_joint": (150, 2.0), "left_hip_yaw_joint": (150, 2.0),
    "left_knee_joint": (250, 2.0), "left_ankle_roll_joint": (60, 0.3), "left_ankle_pitch_joint": (90, 0.1),
    "right_hip_pitch_joint": (150, 2.0), "right_hip_roll_joint": (150, 2.0), "right_hip_yaw_joint": (150, 2.0),
    "right_knee_joint": (250, 2.0), "right_ankle_roll_joint": (60, 0.3), "right_ankle_pitch_joint": (90, 0.1),
    "waist_yaw_joint": (200, 2.5), "waist_roll_joint": (200, 5.0), "waist_pitch_joint": (200, 5.0),
    "left_shoulder_pitch_joint": (90, 2.0), "left_shoulder_roll_joint": (60, 1.0), "left_shoulder_yaw_joint": (20, 0.4),
    "left_elbow_joint": (60, 1.0), "left_wrist_roll_joint": (4, 0.2), "left_wrist_pitch_joint": (4, 0.2),
    "left_wrist_yaw_joint": (4, 0.2),
    "right_shoulder_pitch_joint": (90, 2.0), "right_shoulder_roll_joint": (60, 1.0), "right_shoulder_yaw_joint": (20, 0.4),
    "right_elbow_joint": (60, 1.0), "right_wrist_roll_joint": (4, 0.2), "right_wrist_pitch_joint": (4, 0.2),
    "right_wrist_yaw_joint": (4, 0.2),
    "head_pitch_joint": (30, 1.0), "head_yaw_joint": (30, 1.0),
}


def armature_for(effort: float) -> float:
    """Инерция ротора, приведённая к суставу (в URDF её нет) — оценка по классу мотора."""
    if effort >= 300:
        return 0.1
    if effort >= 100:
        return 0.05
    if effort >= 50:
        return 0.02
    return 0.01


def add_scene(spec: mujoco.MjSpec, prefix: str = "") -> None:
    """Пол, свет и небо. prefix — приставка к именам, если в модели уже есть свои "floor", "grid" и т.п."""
    tex = spec.add_texture(name=f"{prefix}grid", type=mujoco.mjtTexture.mjTEXTURE_2D, builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
                           rgb1=[0.22, 0.25, 0.30], rgb2=[0.30, 0.34, 0.40], mark=mujoco.mjtMark.mjMARK_EDGE,
                           markrgb=[0.5, 0.55, 0.6], width=512, height=512)
    spec.add_texture(name=f"{prefix}sky", type=mujoco.mjtTexture.mjTEXTURE_SKYBOX, builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT,
                     rgb1=[0.45, 0.6, 0.8], rgb2=[0.05, 0.07, 0.1], width=512, height=3072)
    mat = spec.add_material(name=f"{prefix}floor_mat", texrepeat=[8, 8], texuniform=True, reflectance=0.1)
    mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = tex.name
    world = spec.worldbody
    world.add_geom(name=f"{prefix}floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[0, 0, 0.05], material=f"{prefix}floor_mat")
    world.add_light(pos=[0, 0, 4], dir=[0, 0, -1], type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL,
                    diffuse=[0.6, 0.6, 0.6], castshadow=1)
    world.add_light(pos=[2, -2, 3], dir=[-1, 1, -1.2], diffuse=[0.3, 0.3, 0.3], castshadow=0)
    spec.visual.headlight.ambient = [0.35, 0.35, 0.35]
    spec.visual.headlight.diffuse = [0.4, 0.4, 0.4]
    spec.visual.global_.offwidth = 1920
    spec.visual.global_.offheight = 1080


def add_band(spec: mujoco.MjSpec, body: str) -> None:
    """Мягкая подвеска: weld тела к миру в исходной позе с податливым solref (как elastic band в unitree_mujoco).

    Включается/выключается галочкой "band" в правой панели viewer'а, раздел Equality.
    """
    # data = anchor(3), relpose(7), torquescale: якорь в начале тела, нулевой quat => поза из qpos0.
    # По умолчанию MjSpec кладёт сюда polycoef [0 1 0 ...], т.е. якорь в 1 м от тела — подвеска «уплывает».
    spec.add_equality(name="band", type=mujoco.mjtEq.mjEQ_WELD, objtype=mujoco.mjtObj.mjOBJ_BODY,
                      name1=body, solref=BAND_SOLREF, data=[0] * 10 + [1])


def add_gravcomp(spec: mujoco.MjSpec) -> None:
    """Компенсация силы тяжести всех звеньев моментами приводов (в пределах их момента). Только для закреплённой базы:
    со свободной базой часть компенсации пришлась бы на free joint и робот стал бы невесомым."""
    for b in spec.bodies:
        if b.name != "world":
            b.gravcomp = 1
    for j in spec.joints:
        if j.type == mujoco.mjtJoint.mjJNT_HINGE:
            j.actgravcomp = True


def build_spec(urdf: Path, free: bool, band: bool, unitree_kd: bool, gravcomp: bool = True) -> mujoco.MjSpec:
    spec = mujoco.MjSpec.from_file(str(urdf))
    spec.meshdir = str(urdf.parent)        # в URDF пути "meshes/x.stl" + meshdir="meshes" дали бы meshes/meshes/
    spec.compiler.fusestatic = False       # иначе таз без сустава сливается с миром и теряет имя
    spec.option.timestep = 0.002
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST  # демпфирование сервопривода считается неявно
    add_scene(spec)

    pelvis = spec.body("pelvis")
    if free or band:
        pelvis.pos = [0, 0, STAND_HEIGHT - (BAND_DROP if band else 0)]
        pelvis.add_freejoint(name="floating_base")
    else:
        pelvis.pos = [0, 0, STAND_HEIGHT + HANG_HEIGHT]  # таз жёстко прикреплён к миру — робот «висит на стенде»
        if gravcomp:
            add_gravcomp(spec)
    if band:
        add_band(spec, "pelvis")

    # Геомы робота: коллизии (group 0) скрыть в group 3 и сталкивать только с полом (conaffinity=0 => не друг с другом).
    for g in spec.geoms:
        if g.name == "floor":
            continue
        if g.contype:
            g.group = 3
            g.conaffinity = 0

    # Стартовая камера viewer'а: спереди-сбоку, робот на весь кадр.
    spec.visual.global_.azimuth = 150
    spec.visual.global_.elevation = -12
    spec.stat.center = [0, 0, pelvis.pos[2] - 0.15]
    spec.stat.extent = 1.5

    joints = [j for j in spec.joints if j.type == mujoco.mjtJoint.mjJNT_HINGE]
    for j in joints:
        effort = j.actfrcrange[1]
        j.armature = armature_for(effort)
        j.damping = [0.5, 0, 0]  # в MuJoCo 3.15 damping — вектор (линейный коэффициент первым)
        j.frictionloss = 0.1
        kp, kd = UNITREE_GAINS[j.name]
        act = spec.add_actuator(name=j.name.removesuffix("_joint"), target=j.name,
                                trntype=mujoco.mjtTrn.mjTRN_JOINT)
        act.set_to_position(kp=kp, kv=kd)
        act.ctrllimited = mujoco.mjtLimited.mjLIMITED_TRUE
        act.ctrlrange = j.range
        act.forcelimited = mujoco.mjtLimited.mjLIMITED_TRUE
        act.forcerange = [-effort, effort]

    model = spec.compile()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    if not unitree_kd:
        # kv = 2·sqrt(kp·M_ii): M_ii — диагональ матрицы масс в исходной позе (с armature), т.е. ≈ критическое демпфирование.
        mass = np.zeros((model.nv, model.nv))
        mujoco.mj_fullM(model, data, mass)  # в 3.15 сигнатура (m, d, dst), data.qM больше нет
        for act in spec.actuators:
            dof = model.joint(act.target).dofadr[0]
            m_ii = mass[dof, dof]
            kp = UNITREE_GAINS[act.target][0]
            act.set_to_position(kp=kp, kv=max(UNITREE_GAINS[act.target][1], 2.0 * np.sqrt(kp * m_ii)))

    ctrl0 = np.clip(np.zeros(model.nu), model.actuator_ctrlrange[:, 0], model.actuator_ctrlrange[:, 1])
    spec.add_key(name="home", qpos=model.qpos0.copy(), ctrl=ctrl0)
    return spec


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--urdf", type=Path, default=DEFAULT_URDF, help="H2.urdf (рядом папка meshes/)")
    ap.add_argument("--free", action="store_true", help="свободная база: робот на полу (без контроллера баланса падает)")
    ap.add_argument("--band", action="store_true", help="на полу, но таз держит мягкая подвеска (панель Equality)")
    ap.add_argument("--unitree-kd", action="store_true", help="штатные Kd Unitree вместо критического демпфирования")
    ap.add_argument("--no-gravcomp", action="store_true", help="без компенсации силы тяжести в режиме подвеса")
    ap.add_argument("--save-xml", type=Path, default=None, help="сохранить собранную сцену в MJCF и выйти")
    args = ap.parse_args()

    spec = build_spec(args.urdf.resolve(), args.free, args.band, args.unitree_kd, not args.no_gravcomp)
    model = spec.compile()
    if args.save_xml:
        args.save_xml.write_text(spec.to_xml())  # пути мешей абсолютные (meshdir), файл можно открыть откуда угодно
        print("saved", args.save_xml)
        return

    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)

    print(f"H2: {model.nu} суставов, {model.body_subtreemass[0]:.1f} кг. "
          "Ползунки: правая панель -> Control. Backspace — сброс позы, пробел — пауза.")
    for i in range(model.nu):
        lo, hi = np.degrees(model.actuator_ctrlrange[i])
        print(f"  {model.actuator(i).name:<22} {lo:7.1f}° .. {hi:6.1f}°   kp {model.actuator_gainprm[i, 0]:5.0f}"
              f"  kv {-model.actuator_biasprm[i, 2]:5.1f}   max {model.actuator_forcerange[i, 1]:6.1f} Н·м")

    mujoco.viewer.launch(model, data)


if __name__ == "__main__":
    main()
