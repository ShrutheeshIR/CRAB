import symforce
symforce.set_epsilon_to_symbol()
symforce.set_symbolic_api("symengine")
symforce.set_log_level("warning")

import symforce.symbolic as sf
from symforce import codegen
from symforce.codegen import codegen_util


from crusty_kinematics.fk import forward_kinematics
from crusty_io.urdf import load_urdf, urdf_to_robot
from crusty_kinematics.collision_checker import collision_checker

from typing import List, Tuple

def codegen_for_fk():
    
    urdf = load_urdf("robots/panda/panda_spherized.urdf")
    robot = urdf_to_robot(urdf)
    robot = robot.finalize()


    def fk(q: sf.Vector7) -> Tuple[List[List[float]], List[List[float]], List[tuple[str, sf.Pose3]]]:
        return forward_kinematics(robot, q)    

    fk_codegen = codegen.Codegen.function(
        func=fk,
        config=codegen.CppConfig(use_eigen_types=False),
        name="forward_kinematics",
    )
    fk_codegen_data = fk_codegen.generate_function()

    return fk_codegen_data


def codegen_for_collision_checker():
    
    urdf = load_urdf("robots/panda/panda_spherized.urdf")
    robot = urdf_to_robot(urdf)
    robot = robot.finalize()


    def collision_check(q: sf.Vector7) -> Tuple[List[List[float]], List[List[float]]]:
        return collision_checker(robot, q)    

    collision_codegen = codegen.Codegen.function(
        func=collision_check,
        config=codegen.CppConfig(use_eigen_types=False),
        name="collision_checker",
    )
    collision_codegen_data = collision_codegen.generate_function()

    return collision_codegen_data

if __name__ == "__main__":
    fk_codegen_data = codegen_for_fk()
    print(fk_codegen_data.function_dir, fk_codegen_data.generated_files)
    print("Files generated in {}:\n".format(fk_codegen_data.output_dir))
    # print("\nGenerated code:\n"
    #       "----------------\n"
    #       "{}\n"
    #       "----------------".format(fk_codegen_data.generated_files[0].read_text()))


    # collision_codegen_data = codegen_for_collision_checker()
    # print("Files generated in {}:\n".format(collision_codegen_data.output_dir))
    # print("\nGenerated code:\n"
    #       "----------------\n"
    #       "{}\n"
    #       "----------------".format(collision_codegen_data.generated_files[0].read_text()))