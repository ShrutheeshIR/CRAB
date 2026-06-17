import symforce
symforce.set_epsilon_to_symbol()
symforce.set_symbolic_api("symengine")
symforce.set_log_level("warning")

import symforce.symbolic as sf
from symforce import codegen
from symforce.codegen import codegen_util


from crusty_kinematics.fk import forward_kinematics
from crusty_io.urdf import load_urdf, urdf_to_robot

from typing import List, Tuple

def codegen_for_fk():
    
    urdf = load_urdf("robots/panda/panda_spherized.urdf")
    robot = urdf_to_robot(urdf)
    robot = robot.finalize()


    def fk(q: sf.Vector7) -> Tuple[List[List[float]], List[List[float]]]:
        return forward_kinematics(robot, q)    

    fk_codegen = codegen.Codegen.function(
        func=fk,
        config=codegen.CppConfig(use_eigen_types=False),
        name="forward_kinematics",
    )
    fk_codegen_data = fk_codegen.generate_function()

    return fk_codegen_data

if __name__ == "__main__":
    fk_codegen_data = codegen_for_fk()
    print("Files generated in {}:\n".format(fk_codegen_data.output_dir))
    print("\nGenerated code:\n"
          "----------------\n"
          "{}\n"
          "----------------".format(fk_codegen_data.generated_files[0].read_text()))