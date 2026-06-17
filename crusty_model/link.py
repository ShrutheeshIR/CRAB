from dataclasses import dataclass, field
import symforce.symbolic as sf
from ._bounding_sphere import minimum_enclosing_sphere_of_spheres

@dataclass(slots=True)
class Sphere:
    pose: sf.Pose3
    radius: float

@dataclass(slots=True)
class Link:
    name: str
    id: int

    parent_joint: str | None = None
    child_joints: list[str] = field(default_factory=list)

    # for now i will assume that each link is just a list of spheres
    primitives: list[Sphere] = field(default_factory=list)
    bounding_primitive: Sphere | None = None

    def compute_bounding_primitive(self):
        if not self.primitives:
            return None        
        # compute the Minimum_enclosing_sphere_of_spheres_d algorithm to find the bounding sphere of the link
        # https://www.sciencedirect.com/science/article/pii/S0925772115000867
        spheres = [(primitive.pose.t, primitive.radius) for primitive in self.primitives]
        center, radius = minimum_enclosing_sphere_of_spheres(spheres)
        self.bounding_primitive = Sphere(pose=sf.Pose3(t=sf.V3(*center)), radius=radius)

    def __post_init__(self):
        self.compute_bounding_primitive()