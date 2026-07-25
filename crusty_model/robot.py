import symforce.symbolic as sf
from dataclasses import dataclass, field
from .link import Link
from .joint import Joint


def _joint_limits_by_index(robot) -> list[tuple[float, float]]:
    """(lower, upper) per q-index. Unlimited (e.g. continuous) or unspecified joints
    get a wide-open range so the penalty below never activates for them."""
    limits = [(-1e3, 1e3)] * robot.nq
    for joint in robot.joints.values():
        if joint.is_actuated and joint.limits is not None:
            limits[joint.id] = joint.limits
    return limits


@dataclass(slots=True)
class Robot:
    """Immutable description of a robot."""

    name: str

    links: dict[str, Link] = field(default_factory=dict)
    joints: dict[str, Joint] = field(default_factory=dict)
    allowed_collision_pairs: set[tuple[int, int]] = field(default_factory=set)

    end_effectors: list[str] = field(default_factory=list)

    # we need to maintain a mapping between links and index, so that we can use the index to access the link in the collision checker
    link_name_to_index: dict[str, int] = field(init=False)


    traversal_order: list[int] = field(init=False)

    _frozen: bool = field(default=False, init=False, repr=False)

    @property
    def nq(self) -> int:
        """Number of generalized coordinates.
        This should be number of actuated joints only
        """
        return sum(1 for joint in self.joints.values() if joint.is_actuated)

    @property
    def nv(self) -> int:
        """Number of generalized velocities."""
        return sum(1 for joint in self.joints.values() if joint.is_actuated)

    @property
    def root(self) -> Link:
        for link in self.links.values():
            if link.parent_joint is None:
                return link
        return Link(name="DUMMY_ROOT", id=-1)


    def compute_allowed_collision_pairs(self):
        """
        Contrary to the name, this actually computes the pairs for which we should check for collision, which is the complement of the allowed collision pairs.
        For now, we will just check for collisions between all pairs of links that are not adjacent in the kinematic tree. This is a conservative approximation, but it should be sufficient for our purposes.
        Later we will load the srdf file and use the allowed collision pairs specified there, which will allow us to ignore collisions between links that are close to each other in the kinematic tree but are not actually adjacent (e.g. the upper arm and the forearm of a humanoid robot).
        """
        adjacent_pairs = set()
        for joint in self.joints.values():
            adjacent_pairs.add((joint.parent, joint.child))
            adjacent_pairs.add((joint.child, joint.parent))

        all_pairs = set()
        link_names = list(self.links.keys())
        for i in range(len(link_names)):
            for j in range(i + 1, len(link_names)):
                all_pairs.add((link_names[i], link_names[j]))

        self.allowed_collision_pairs = all_pairs - adjacent_pairs


    def __post_init__(self):
        children: dict[str, list[str]] = {name: [] for name in self.links}

        for joint in self.joints.values():
            children[joint.parent].append(joint.name)

        self.traversal_order = []

        def dfs(link_name: str):
            # fix for dummy root
            if link_name == "DUMMY_ROOT":
                return

            for joint_name in children[link_name]:
                self.traversal_order.append(joint_name)
                child_link = self.joints[joint_name].child
                dfs(child_link)

        dfs(self.root.name)

        # Index-based alternative to keying by link name: SymForce's symbolic tree
        # walking (StorageOps) only knows how to recurse into list/tuple/Matrix/Values
        # containers, not dict, so anything symbolic keyed by link name (e.g.
        # crusty_kinematics.fk's link_poses) has to be addressed by index instead.
        # Root gets index 0; every other link gets the index of the joint that first
        # visits it as a child in traversal_order, so this stays consistent with the
        # ordering already used elsewhere (e.g. crab_codegen/sphere_order.py).
        self.link_name_to_index = {self.root.name: 0}
        for i, joint_name in enumerate(self.traversal_order):
            child_link = self.joints[joint_name].child
            self.link_name_to_index[child_link] = i + 1


    def finalize(self) -> "Robot":
        """
        This ensures all sorts of checks to ensure the robot is watertight
         - all joints have valid parent and child links
            - all links are connected to the root
         - the parent of a child link is the link itself
         - all joints have unique names
         - all links have unique names
        """
        assert len(set(self.joints.keys())) == len(self.joints), "Duplicate joint names found"
        assert len(set(self.links.keys())) == len(self.links), "Duplicate link names found"

        for joint_name, joint in self.joints.items():
            assert joint.parent in self.links, f"Joint {joint_name} has invalid parent link {joint.parent}"
            assert joint.child in self.links, f"Joint {joint_name} has invalid child link {joint.child}"
            assert self.links[joint.parent].child_joints.count(joint_name) == 1, f"Joint {joint_name} is not properly connected to its parent link {joint.parent}"
            assert self.links[joint.child].parent_joint == joint_name, f"Joint {joint_name} is not properly connected to its child link {joint.child}"


        roots = [link for link in self.links.values() if link.parent_joint is None]
        assert len(roots) == 1, "Multiple root links found"
        root = roots[0]


        # Check that all links are connected to the root
        visited = set()
        def dfs(link_name: str):
            visited.add(link_name)
            for joint_name in self.links[link_name].child_joints:
                child_link = self.joints[joint_name].child
                dfs(child_link)
        dfs(root.name)
        assert len(visited) == len(self.links), "Not all links are connected to the root"


        # Cycle detection
        WHITE, GRAY, BLACK = 0, 1, 2
        color: dict[str, int] = {link_name: WHITE for link_name in self.links}

        def dfs_cycle(link_name: str) -> bool:
            color[link_name] = GRAY
            for joint_name in self.links[link_name].child_joints:
                child_link = self.joints[joint_name].child
                if color[child_link] == GRAY:
                    return True
                if color[child_link] == WHITE and dfs_cycle(child_link):
                    return True
            color[link_name] = BLACK
            return False
        assert not dfs_cycle(root.name), "Cycle detected in the robot kinematic tree"


        self._frozen = True
        return self