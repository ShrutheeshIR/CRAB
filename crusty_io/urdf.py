import yourdfpy
from crusty_model.link import Link, Sphere
from crusty_model.joint import Joint
from crusty_model.robot import Robot
import symforce.symbolic as sf


def load_urdf(path: str) -> yourdfpy.URDF:
    """Load a URDF file and return a yourdfpy.URDF object."""
    return yourdfpy.URDF.load(path)

def urdf_to_robot(urdf: yourdfpy.URDF) -> Robot:
    """Convert a yourdfpy.URDF object to a Robot object."""
    robot = Robot(name="robot")

    for i, (link_name, link) in enumerate(urdf.link_map.items()):
        collision_spheres = []
        for collision in link.collisions:
            if collision.geometry.sphere:
                collision_spheres.append(Sphere(
                    pose = sf.Pose3(R = sf.Rot3.from_rotation_matrix(collision.origin[:3, :3]), t = sf.V3(collision.origin[:3, 3])),
                    radius = collision.geometry.sphere.radius,
                ))
        robot.links[link.name] = Link(
            name=link.name,
            id=i,
            primitives=collision_spheres,
        )


        # (TODO) -- extract out the collision spheres from the urdf here and add them to the link
        # robot.links[link.name] = Link(name=link.name, id=i)

    for i, (joint_name, joint) in enumerate(urdf.joint_map.items()):
        parent_name = robot.links[joint.parent].name
        child_name = robot.links[joint.child].name

        robot.joints[joint.name] = Joint(
            name=joint.name,
            id=i,
            parent=parent_name,
            child=child_name,
            origin= sf.Pose3(R = sf.Rot3.from_rotation_matrix(joint.origin[:3, :3]), t = sf.V3(joint.origin[:3, 3])),
            axis=sf.V3(joint.axis),
            type=joint.type,
            limits=(joint.limit.lower, joint.limit.upper) if joint.limit else None,
            velocity_limits=(joint.limit.velocity, -joint.limit.velocity) if joint.limit else None,
            is_actuated=joint.type != "fixed",
        )

        # update the parent and child links with the joint information
        robot.links[joint.parent].child_joints.append(joint.name)
        robot.links[joint.child].parent_joint = joint.name

    # needed to trigger the post init function to compute the traversal order
    # this is stupid i know yeah but if you have a better way to do be my guest
    new_robot = Robot(
        name=robot.name,
        links=robot.links,
        joints=robot.joints,
        allowed_collision_pairs=robot.allowed_collision_pairs,
        end_effectors=robot.end_effectors,
    )

    return new_robot