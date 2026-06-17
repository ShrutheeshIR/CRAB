from dataclasses import dataclass, field
import symforce.symbolic as sf

@dataclass(slots=True)
class Joint:
    name: str
    id: int

    parent: str
    child: str

    origin: sf.Pose3
    axis: sf.Vector3 | None = None

    type: str = "revolute"
    limits: tuple[float, float] | None = None
    velocity_limits: tuple[float, float] | None = None

    is_actuated: bool = True

    @staticmethod
    def from_dict(name: str, id: int, data: dict) -> "Joint":
        return Joint(
            name=name,
            id=id,
            parent=data["parent"],
            child=data["child"],
            origin=sf.Pose3.from_dict(data["origin"]),
            axis=sf.Vector3.from_dict(data["axis"]) if "axis" in data else None,
            type=data.get("type", "revolute"),
            limits=tuple(data["limits"]) if "limits" in data else None,
            velocity_limits=tuple(data["velocity_limits"]) if "velocity_limits" in data else None,
        )
    
    @staticmethod
    def dummy(name: str, id: int, parent: str, child: str) -> "Joint":
        return Joint(
            name=name,
            id=id,
            parent=parent,
            child=child,
            origin=sf.Pose3(),
            is_actuated=False,
        )
    
    def __str__(self) -> str:
        return f"Joint(name={self.name}, id={self.id}, parent={self.parent}, child={self.child}, origin={self.origin}, axis={self.axis}, type={self.type}, limits={self.limits}, velocity_limits={self.velocity_limits}, is_actuated={self.is_actuated})"

