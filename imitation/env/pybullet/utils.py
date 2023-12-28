import pybullet as p


def add_obstacle(obs_state, color=[1.0, 0.0, 0.0, 1], size=0.05):
    ### DRAW OBSTACLE ###
    obst_radius = size
    obst_collision = -1#p.createCollisionShape(p.GEOM_SPHERE, radius=obst_radius,
                                                   #height=0.02, collisionFramePosition=[0., 0., 0.0075])
    obst_visual = p.createVisualShape(p.GEOM_SPHERE, radius=obst_radius, length=0.005,
                                             rgbaColor=color)
    obst = p.createMultiBody(baseMass=0.0,
                                    baseCollisionShapeIndex=obst_collision,
                                    baseVisualShapeIndex=obst_visual,
                                    basePosition=obs_state)
    return obst

def add_goal(goal_state, p, color=[1.0, 1.0, 0.0, 1], size=0.05):
    goal_radius = size
    goal_collision = -1
    goal_visual = p.createVisualShape(p.GEOM_SPHERE, radius=goal_radius, length=0.005,
                                             rgbaColor=color)
    goal = p.createMultiBody(baseMass=0.1,
                                    baseCollisionShapeIndex=goal_collision,
                                    baseVisualShapeIndex=goal_visual,
                                    basePosition=goal_state)
    return goal


class Debug_Joint_Slider():
    def __init__(self, limits, p, JOINT_ID, robot):
        self.q_ids = []
        self.p = p
        self.JOINT_ID = JOINT_ID
        self.robot = robot

        for i, limit in enumerate(limits):
            self.q_ids.append(p.addUserDebugParameter(paramName='Joint'+str(i), rangeMin=limit[0], rangeMax=limit[1], startValue=0))

    def read(self):
        q = []
        for q_id in self.q_ids:
            q.append(self.p.readUserDebugParameter(q_id))
        return q

    def set(self):
        q = self.read()
        for i, q_i in enumerate(q):
            self.p.resetJointState(self.robot, self.JOINT_ID[i], q_i)
