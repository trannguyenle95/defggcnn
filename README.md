Note: This code is developed based on Generative Grasping CNN (GG-CNN) (https://github.com/dougsm/ggcnn)
# Deformation-Aware Data-Driven Grasp Synthesis

Grasp synthesis for 3-D deformable objects remains a little-explored topic, most works aiming to minimize deformations. However, deformations are not necessarily harmful—humans are, for example, able to exploit deformations to generate new potential grasps. How to achieve that on a robot is though an open question. This letter proposes an approach that uses object stiffness information in addition to depth images for synthesizing high-quality grasps. We achieve this by incorporating object stiffness as an additional input to a state-of-the-art deep grasp planning network. We also curate a new synthetic dataset of grasps on objects of varying stiffness using the Isaac Gym simulator for training the network. We experimentally validate and compare our proposed approach against the case where we do not incorporate object stiffness on a total of 2800 grasps in simulation and 560 grasps on a real Franka Emika Panda. The experimental results show significant improvement in grasp success rate using the proposed approach on a wide range of objects with varying shapes, sizes, and stiffnesses. Furthermore, we demonstrate that the approach can generate different grasping strategies for different stiffness values. Together, the results clearly show the value of incorporating stiffness information when grasping objects of varying stiffness. Code and video are available at: https://irobotics.aalto.fi/defggcnn/ .

If you use this work, please cite:

```text
@ARTICLE{tran_defggcnn,
  author={Le, Tran Nguyen and Lundell, Jens and Abu-Dakka, Fares J. and Kyrki, Ville},
  journal={IEEE Robotics and Automation Letters}, 
  title={Deformation-Aware Data-Driven Grasp Synthesis}, 
  year={2022},
  volume={7},
  number={2},
  pages={3038-3045},
  keywords={Grasping;Strain;Robots;Deformable models;Three-dimensional displays;Solid modeling;Grippers;Deep Learning in grasping and manipulation;grasping},
  doi={10.1109/LRA.2022.3146551}}

```

**Contact**

Any questions or comments contact [Tran Nguyen Le]

## Installation

This code was developed with Python 3.6 on Ubuntu 16.04.  Python requirements can installed by:

```bash
pip install -r requirements.txt
```
