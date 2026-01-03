import argparse

import numpy as np
import warp as wp
from pxr import Gf, Usd, UsdGeom, UsdPhysics


def apply_collision_api(prim):
    type_name = str(prim.GetTypeName()).lower()

    if "proxy" in str(prim.GetPath()):
        return

    if type_name in ("mesh", "capsule", "sphere", "box", "cylinder", "cone"):
        print(f"Applying CollisionAPI to {prim}")
        collisionAPI = UsdPhysics.CollisionAPI.Apply(prim)
        collisionAPI.CreateCollisionEnabledAttr(True)

    for child in prim.GetChildren():
        apply_collision_api(child)


def parse_xform(prim, local=True):
    xform = UsdGeom.Xform(prim)
    if local:
        mat = np.array(xform.GetLocalTransformation(), dtype=np.float32)
    else:
        mat = np.array(xform.ComputeLocalToWorldTransform(Usd.TimeCode.Default()), dtype=np.float32)
    rot = wp.quat_from_matrix(wp.mat33(mat[:3, :3].T.flatten()))
    pos = mat[3, :3]
    return wp.transform(pos, rot)


def add_chainlink_joints(stage):
    chains = {
        "HangingLanternA_01": "HangingLanternChainA_05",
        "HangingLanternA_02": "HangingLanternChainA_06",
        "HangingLanternA_03": "HangingLanternChainA_04",
        "HangingLanternC_01": "HangingLanternChainA_01",
        "HangingLanternC_02": "HangingLanternChainA_07",
        "HangingLanternD_01": "HangingLanternChainA_03",
        "HangingLanternE_01": "HangingLanternChainA_02",
    }

    bad_links = ["16", "40"]

    world = stage.GetPrimAtPath("/World")

    for lantern, chain in chains.items():
        lantern_prim = world.GetPrimAtPath(f"/World/{lantern}")
        UsdPhysics.RigidBodyAPI.Apply(lantern_prim)

        apply_collision_api(lantern_prim)

        chain_geo_prim = world.GetPrimAtPath(f"/World/{chain}/geo")
        chain_joints_prim = world.GetPrimAtPath(f"/World/{chain}/geo/joints")

        chainlinks = [
            child
            for child in chain_geo_prim.GetChildren()
            if "chainlink" in str(child.GetPath()) and not any(xx in str(child.GetPath()) for xx in bad_links)
        ]

        for chainlink in chainlinks:
            UsdPhysics.RigidBodyAPI.Apply(chainlink)
            mass_api = UsdPhysics.MassAPI.Apply(chainlink)
            mass_api.CreateMassAttr().Set(0.1)
            print(f"Applied RigidBodyAPI to {chainlink}")
            # apply_collision_api(lantern_prim)

        # chainlink joints
        joints = [UsdPhysics.SphericalJoint(joint_prim) for joint_prim in chain_joints_prim.GetChildren()]

        for k, joint in enumerate(joints):
            if k >= len(chainlinks):
                continue

            if k > 0:
                joint.GetBody0Rel().AddTarget(chainlinks[k - 1].GetPrimPath())
                joint.GetBody1Rel().AddTarget(chainlinks[k].GetPrimPath())

                b0_xform = parse_xform(chainlinks[k - 1])
                b1_xform = parse_xform(chainlinks[k])
                local_pos_1 = 0.5 * wp.transform_point(
                    wp.transform_inverse(b1_xform), wp.transform_get_translation(b0_xform)
                )
                local_pos_0 = 0.5 * wp.transform_point(
                    wp.transform_inverse(b0_xform), wp.transform_get_translation(b1_xform)
                )

                joint.GetLocalPos1Attr().Set(Gf.Vec3f(local_pos_1[0], local_pos_1[1], local_pos_1[2]))
                joint.GetLocalPos0Attr().Set(Gf.Vec3f(local_pos_0[0], local_pos_0[1], local_pos_0[2]))
            else:
                joint.GetBody0Rel().AddTarget(chainlinks[k].GetPrimPath())
                b0_xform = parse_xform(chainlinks[k])
                joint.GetLocalPos0Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
                joint.GetLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
                joint.GetBody1Rel().AddTarget("/World")

            # print(
            #     f"Joint from {joint.GetBody0Rel().GetTargets()[0]} to {joint.GetBody1Rel().GetTargets()[0]} local pos: {joint.GetLocalPos1Attr().Get()}"
            # )

        UsdPhysics.ArticulationRootAPI.Apply(chainlinks[0])

        # chain/lantern
        lantern_joint = UsdPhysics.SphericalJoint.Define(stage, f"/World/{lantern}_joint")
        lantern_joint.GetBody0Rel().AddTarget(chainlinks[-1].GetPrimPath())
        lantern_joint.GetBody1Rel().AddTarget(lantern_prim.GetPrimPath())

        b0_xform = parse_xform(chainlinks[-1], local=False)
        b1_xform = parse_xform(lantern_prim, local=False)
        local_pos = wp.transform_point(wp.transform_inverse(b1_xform), wp.transform_get_translation(b0_xform))
        lantern_joint.GetLocalPos1Attr().Set(Gf.Vec3f(local_pos[0], local_pos[1], local_pos[2]))

        print(f"Added lanterns for chain {chain} [{lantern}]")


def add_lantern_joints(stage):
    chains = [
        [
            "HangingLanternChainA_08",
            "HangingLanternChainA_05",
            "HangingLanternA_01",
        ],
        [
            "HangingLanternChainA_13",
            "HangingLanternChainA_06",
            "HangingLanternA_02",
        ],
        [
            "HangingLanternChainA_04",
            "HangingLanternA_03",
        ],
        [
            "HangingLanternChainA_12",
            "HangingLanternChainA_01",
            "HangingLanternC_01",
        ],
        [
            "HangingLanternChainA_11",
            "HangingLanternChainA_07",
            "HangingLanternC_02",
        ],
        [
            "HangingLanternChainA_09",
            "HangingLanternChainA_02",
            "HangingLanternE_01",
        ],
        [
            "HangingLanternChainA_10",
            "HangingLanternChainA_03",
            "HangingLanternD_01",
        ],
    ]

    chain_link_length = 0.6

    world = stage.GetPrimAtPath("/World")

    for chain_idx, chain in enumerate(chains):
        # register links as rigid bodies
        bodies = []
        for i, link in enumerate(chain):
            chain_prim = world.GetPrimAtPath(f"/World/{link}")
            UsdPhysics.RigidBodyAPI.Apply(chain_prim)
            apply_collision_api(world.GetPrimAtPath(f"/World/{link}/geo"))
            bodies.append(chain_prim)

            if i > 0 and i < len(chain) - 1:
                # connect links with fixed joints
                fixed_joint = UsdPhysics.FixedJoint.Define(stage, f"/World/fixed_joint_{chain_idx}_{i}")
                fixed_joint.GetBody0Rel().AddTarget(bodies[i - 1].GetPrimPath())
                fixed_joint.GetBody1Rel().AddTarget(bodies[i].GetPrimPath())
                fixed_joint.GetLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, chain_link_length))

        UsdPhysics.ArticulationRootAPI.Apply(bodies[0])

        anchor_joint = UsdPhysics.SphericalJoint.Define(stage, f"/World/anchor_joint_{chain_idx}")
        anchor_joint.GetBody0Rel().AddTarget(bodies[0].GetPrimPath())
        anchor_joint.GetBody1Rel().AddTarget("/World")
        anchor_joint.GetLocalPos0Attr().Set(Gf.Vec3f(0.0, 0.0, chain_link_length))
        anchor_joint.GetLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, chain_link_length))

        lantern_joint = UsdPhysics.SphericalJoint.Define(stage, f"/World/lantern_joint_{chain_idx}")
        lantern_joint.GetBody0Rel().AddTarget(bodies[-2].GetPrimPath())
        lantern_joint.GetBody1Rel().AddTarget(bodies[-1].GetPrimPath())

        # figure out world pose of the lantern to compute the right offset from the last chain link
        lantern_xform = parse_xform(bodies[-1])
        chain_xform = parse_xform(bodies[-2])
        lantern_offset = lantern_xform.p - chain_xform.p
        lantern_joint.GetLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, -lantern_offset[2]))

        print(f"Added lanterns for chain {chain_idx} [{', '.join(chain)}]")


def add_rod_joints(stage, use_fixed_joints=False):
    # name of rod parent prim, name of lantern prim, number of rod links
    rods = [
        ("HangingLanternChainA_01", "HangingLanternC_01"),  # 3
        ("HangingLanternChainA_02", "HangingLanternE_01"),  # 2
        ("HangingLanternChainA_03", "HangingLanternD_01"),  # 1
        ("HangingLanternChainA_04", "HangingLanternA_03"),  # 1
        ("HangingLanternChainA_05", "HangingLanternA_01"),  # 1
        ("HangingLanternChainA_06", "HangingLanternA_02"),  # 2
        ("HangingLanternChainA_07", "HangingLanternC_02"),  # 2
    ]

    joints = []
    world = stage.GetPrimAtPath("/World")
    for rod_idx, (chain_name, lantern_name) in enumerate(rods):
        jp_prim = stage.GetPrimAtPath(f"/World/{chain_name}/geo/jointpositions")
        joint_positions: list[wp.vec3] = [parse_xform(child, local=False).p for child in jp_prim.GetChildren()]
        # register links as rigid bodies
        geo_prim = stage.GetPrimAtPath(f"/World/{chain_name}/geo")
        rod_link_prims = geo_prim.GetChildren()[1:]  # skip "jointpositions" prim
        assert len(joint_positions) == len(rod_link_prims) + 1
        assert len(rod_link_prims) > 0
        UsdPhysics.ArticulationRootAPI.Apply(rod_link_prims[0])
        for i, rod_prim in enumerate(rod_link_prims):
            UsdPhysics.RigidBodyAPI.Apply(rod_prim)
            apply_collision_api(rod_prim)

            if i > 0:
                # connect rod links
                if use_fixed_joints:
                    joint = UsdPhysics.FixedJoint.Define(stage, f"/World/rod_joint_{rod_idx}_{i}")
                else:
                    joint = UsdPhysics.SphericalJoint.Define(stage, f"/World/rod_joint_{rod_idx}_{i}")
                joint.GetBody0Rel().AddTarget(rod_link_prims[i - 1].GetPrimPath())
                joint.GetBody1Rel().AddTarget(rod_link_prims[i].GetPrimPath())
                diff = joint_positions[i - 1] - joint_positions[i]
                joint.GetLocalPos1Attr().Set(Gf.Vec3f(*diff))
                joints.append(joint)

        anchor_joint = UsdPhysics.SphericalJoint.Define(stage, f"/World/anchor_joint_{rod_idx}")
        anchor_joint.GetBody0Rel().AddTarget(rod_link_prims[0].GetPrimPath())
        anchor_joint.GetBody1Rel().AddTarget("/World")
        p0 = joint_positions[0] - joint_positions[1]
        anchor_joint.GetLocalPos0Attr().Set(Gf.Vec3f(*p0))
        p1 = joint_positions[0] - parse_xform(rod_link_prims[0], local=False).p
        anchor_joint.GetLocalPos1Attr().Set(Gf.Vec3f(*p1))
        joints.append(anchor_joint)

        lantern_prim = stage.GetPrimAtPath(f"/World/{lantern_name}")
        UsdPhysics.RigidBodyAPI.Apply(lantern_prim)
        apply_collision_api(lantern_prim)

        lantern_joint = UsdPhysics.SphericalJoint.Define(stage, f"/World/lantern_joint_{rod_idx}")
        lantern_joint.GetBody0Rel().AddTarget(rod_link_prims[-1].GetPrimPath())
        lantern_joint.GetBody1Rel().AddTarget(lantern_prim.GetPrimPath())

        # figure out world pose of the lantern to compute the right offset from the last rod link
        lantern_xform = parse_xform(lantern_prim)
        lantern_offset = lantern_xform.p - joint_positions[-1]
        lantern_joint.GetLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, -lantern_offset[2]))
        joints.append(lantern_joint)

        print(f"Added joints for chain {chain_name} [{', '.join(str(p.GetPrimPath()) for p in rod_link_prims)}]")


def add_lantern_rod_joints(stage):
    chains = {
        "HangingLanternA_01": "HangingLanternChainA_05",
        "HangingLanternA_02": "HangingLanternChainA_06",
        "HangingLanternA_03": "HangingLanternChainA_04",
        "HangingLanternC_01": "HangingLanternChainA_01",
        "HangingLanternC_02": "HangingLanternChainA_07",
        "HangingLanternD_01": "HangingLanternChainA_03",
        "HangingLanternE_01": "HangingLanternChainA_02",
    }

    world = stage.GetPrimAtPath("/World")

    for lantern, chain in chains.items():
        lantern_prim = world.GetPrimAtPath(f"/World/{lantern}")
        UsdPhysics.RigidBodyAPI.Apply(lantern_prim)

        apply_collision_api(lantern_prim)

        chain_geo_prim = world.GetPrimAtPath(f"/World/{chain}/geo")
        chain_joints_prim = world.GetPrimAtPath(f"/World/{chain}/geo/jointpositions")

        chainlinks = [child for child in chain_geo_prim.GetChildren() if "hangingLanternChain" in str(child.GetPath())]

        for chainlink in chainlinks:
            UsdPhysics.RigidBodyAPI.Apply(chainlink)
            # mass_api = UsdPhysics.MassAPI.Apply(chainlink)
            # mass_api.CreateMassAttr().Set(2.0)
            print(f"Applied RigidBodyAPI to {chainlink}")
            apply_collision_api(chainlink)

        # chainlink joints
        joints = list(chain_joints_prim.GetChildren())

        chainlinks = list(reversed(chainlinks))

        for k, joint_xf_prim in enumerate(joints):
            joint_xform = parse_xform(joint_xf_prim, local=False)

            joint = UsdPhysics.SphericalJoint.Define(stage, f"{joint_xf_prim.GetPath()}/joint")

            if k == 0:
                b0 = chainlinks[k]
                b1 = stage.GetPrimAtPath("/World")
            elif k == len(joints) - 1:
                b0 = chainlinks[k - 1]
                b1 = lantern_prim
            else:
                b0 = chainlinks[k - 1]
                b1 = chainlinks[k]

            joint.GetBody0Rel().AddTarget(b0.GetPrimPath())
            joint.GetBody1Rel().AddTarget(b1.GetPrimPath())

            b0_xform = parse_xform(b0, local=False)
            local_pos_0 = wp.transform_point(wp.transform_inverse(b0_xform), wp.transform_get_translation(joint_xform))
            joint.GetLocalPos0Attr().Set(Gf.Vec3f(local_pos_0[0], local_pos_0[1], local_pos_0[2]))

            if k > 0:
                b1_xform = parse_xform(b1, local=False)
            else:
                b1_xform = b0_xform

            local_pos_1 = wp.transform_point(wp.transform_inverse(b1_xform), wp.transform_get_translation(joint_xform))
            joint.GetLocalPos1Attr().Set(Gf.Vec3f(local_pos_1[0], local_pos_1[1], local_pos_1[2]))

            print(
                f"Joint from {joint.GetBody0Rel().GetTargets()[0]} to {joint.GetBody1Rel().GetTargets()[0]} local pos: {joint.GetLocalPos1Attr().Get()}"
            )
            # break

        UsdPhysics.ArticulationRootAPI.Apply(chainlinks[0])

        print(f"Added lanterns for chain {chain} [{lantern}]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_path", type=str)
    args = parser.parse_args()

    output_path = args.input_path.replace(".usd", "_physics.usd")

    stage = Usd.Stage.Open(args.input_path)

    for prim in stage.Traverse():
        if "proxy" in str(prim.GetPath()):
            continue
        path = str(prim.GetPath()).split("/")

        # ROBOT
        if any(name in path[-1] for name in ("HEAD", "HIP", "KNEE", "PELVIS", "NECK", "FOOT", "ANTENNA")):
            print(f"Applying RigidBodyAPI to {prim}")
            rigidBodyAPI = UsdPhysics.RigidBodyAPI.Apply(prim)
            rigidBodyAPI.CreateKinematicEnabledAttr(True)

            for child in prim.GetChildren():
                apply_collision_api(child)

        # TERRAIN (adjust)
        elif any(name in path[-1] for name in ("terrainMaincol",)):
            print(f"Applying CollisionAPI to {prim}")
            collisionAPI = UsdPhysics.CollisionAPI.Apply(prim)
            collisionAPI.CreateCollisionEnabledAttr(True)

        # RIGID BODIES (adjust)
        elif len(path) == 5 and any(name in path[-1] for name in ("gear", "piece", "piston")):
            print(f"Applying RigidBodyAPI and MassAPI to {prim}")
            rigidBodyAPI = UsdPhysics.RigidBodyAPI.Apply(prim)
            massAPI = UsdPhysics.MassAPI.Apply(prim)

            for child in prim.GetChildren():
                apply_collision_api(child)

    # check if lanterns are present
    if stage.GetPrimAtPath("/World/HangingLanternChainA_01/geo/joints").IsValid():
        print("Adding chain links")
        add_chainlink_joints(stage)
    elif stage.GetPrimAtPath("/World/HangingLanternChainA_01/geo/jointpositions").IsValid():
        print("Adding rod links")
        add_lantern_rod_joints(stage)  # V1 (Gilles)
        # add_rod_joints(stage) # V2 (Eric)
    elif stage.GetPrimAtPath("/World/HangingLanternChainA_08").IsValid():
        print("Adding lanterns")
        add_lantern_joints(stage)

    stage.Export(output_path)
    print(f"Saved to {output_path}")
