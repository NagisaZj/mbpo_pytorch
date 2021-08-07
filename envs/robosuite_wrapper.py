import numpy as np
from gym import Wrapper
from gym import spaces

def makeEnvWrapper(env, num_joints=7, has_gripper=True):
    """return wrapped gym environment for parallel sample collection (vectorized environments)"""
    def helper():
        return RobosuiteWrapper(env, num_joints, has_gripper)
    return helper

def makeMetaEnvWrapper(env, num_joints=7, has_gripper=True):
    """return wrapped gym environment for parallel sample collection (vectorized environments)"""
    def helper():
        return MetaWrapper(env, num_joints, has_gripper)
    return helper

class RobosuiteWrapper(Wrapper):
    def __init__(self,env,num_joints,has_gripper=True):
        self.env = env

        self.action_space = spaces.Box(low=self.env.action_spec[0], high=self.env.action_spec[1], dtype=np.float32)
        obs = self.env.reset()
        if has_gripper:
            self.obs_dim = (obs['object-state'].shape[0]+15)*(num_joints+1)   # endeffector+gripper+sign
        else:
            self.obs_dim = (obs['object-state'].shape[0]+10) *num_joints  #endeffector+jointobservation

        high = np.inf * np.ones(self.obs_dim)
        low = -high
        self.observation_space = spaces.Box(low=low.astype(np.float32), high=high.astype(np.float32))
        low, high = self.env.action_spec
        self.action_space = spaces.Box(low=low.astype(np.float32), high=high.astype(np.float32))
        # self.observation_space = spaces.Box(low=np.ones(self.obs_dim)*-1, high=np.ones(self.obs_dim), dtype=np.float32)
        self.reward_range = (-np.inf,np.inf)
        self.metadata = None
        self.num_joints = num_joints
        self.has_gripper = has_gripper

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        return self._get_obs(observation), reward, done, info

    def get_spaces(self, **kwargs):
        return self.observation_space,self.action_space

    def reset(self, **kwargs):
        observation = self.env.reset(**kwargs)
        return self._get_obs(observation)

    def _get_obs(self,obs):
        joint_obs = []
        for i in range(self.num_joints):
            if self.has_gripper:
                joint_obs.append(np.concatenate(
                                       [obs['robot0_eef_pos'],
                                       obs['robot0_eef_quat'],
                                       obs['object-state'],
                                       obs['robot0_joint_pos_cos'][i:i + 1],
                                       obs['robot0_joint_pos_sin'][i:i + 1],
                                       obs['robot0_joint_vel'][i:i + 1],
                                        np.zeros(5)
                                        ]))
            else:
                joint_obs.append(np.concatenate(
                    [obs['robot0_eef_pos'],
                     obs['robot0_eef_quat'],
                     obs['object-state'],
                     obs['robot0_joint_pos_cos'][i:i + 1],
                     obs['robot0_joint_pos_sin'][i:i + 1],
                     obs['robot0_joint_vel'][i:i + 1]
                     ]))
        if self.has_gripper:
            joint_obs.append(np.concatenate(
                [obs['robot0_eef_pos'],
                 obs['robot0_eef_quat'],
                 obs['object-state'],
                 np.zeros(3),
                 obs['robot0_gripper_qpos'],
                 obs['robot0_gripper_qvel'],
                 np.ones(1)
                 ]))
        joint_obs = np.concatenate(joint_obs)

        return joint_obs


class MetaWrapper(Wrapper):
    def __init__(self,env,num_joints,has_gripper=True):
        self.env = env

        self.action_space = spaces.Box(low=self.env.action_spec[0], high=self.env.action_spec[1], dtype=np.float32)
        obs = self.env.reset()
        if has_gripper:
            self.obs_dim = obs['object-state'].shape[0]+15   # endeffector+gripper+sign
        else:
            self.obs_dim = obs['object-state'].shape[0]+10   #endeffector+jointobservation
        self.obs_dim = self.obs_dim * 2 + 1 + 1
        # print(self.obs_dim)
        self.observation_space = spaces.Box(low=np.ones(self.obs_dim)*-1, high=np.ones(self.obs_dim), dtype=np.float32)
        self.reward_range = (-np.inf,np.inf)
        self.metadata = None
        self.num_joints = num_joints
        self.has_gripper = has_gripper
        self.previous_a = np.zeros((self.env.action_spec[0].shape[0]))
        self.previous_r = np.zeros((1))
        self.previous_obs = obs

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        return_obs = self._get_obs(observation)
        self.previous_a = action
        self.previous_r = np.ones((1)) * reward
        # print(self.previous_r.shape)
        self.previous_obs = observation
        return return_obs, reward, done, info

    def get_spaces(self, **kwargs):
        return self.observation_space,self.action_space

    def reset(self, **kwargs):
        observation = self.env.reset(**kwargs)
        self.previous_obs = observation
        self.previous_a = np.zeros((self.env.action_spec[0].shape[0]))
        self.previous_r = np.zeros((1))
        return self._get_obs(observation)

    def _get_obs(self,obs):
        joint_obs = []
        for i in range(self.num_joints):
            if self.has_gripper:
                joint_obs.append(np.concatenate(
                                       [obs['robot0_eef_pos'],
                                       obs['robot0_eef_quat'],
                                       obs['object-state'],
                                       obs['robot0_joint_pos_cos'][i:i + 1],
                                       obs['robot0_joint_pos_sin'][i:i + 1],
                                       obs['robot0_joint_vel'][i:i + 1],
                                        np.zeros(5),
                                        self.previous_obs['robot0_eef_pos'],
                                        self.previous_obs['robot0_eef_quat'],
                                        self.previous_obs['object-state'],
                                        self.previous_obs['robot0_joint_pos_cos'][i:i + 1],
                                        self.previous_obs['robot0_joint_pos_sin'][i:i + 1],
                                        self.previous_obs['robot0_joint_vel'][i:i + 1],
                                        np.zeros(5),
                                        self.previous_a[i:i+1],
                                        self.previous_r
                                        ]))
            else:
                joint_obs.append(np.concatenate(
                    [obs['robot0_eef_pos'],
                     obs['robot0_eef_quat'],
                     obs['object-state'],
                     obs['robot0_joint_pos_cos'][i:i + 1],
                     obs['robot0_joint_pos_sin'][i:i + 1],
                     obs['robot0_joint_vel'][i:i + 1],
                     self.previous_obs['robot0_eef_pos'],
                     self.previous_obs['robot0_eef_quat'],
                     self.previous_obs['object-state'],
                     self.previous_obs['robot0_joint_pos_cos'][i:i + 1],
                     self.previous_obs['robot0_joint_pos_sin'][i:i + 1],
                     self.previous_obs['robot0_joint_vel'][i:i + 1],
                                        self.previous_a[i:i+1],
                                        self.previous_r
                     ]))
        if self.has_gripper:
            joint_obs.append(np.concatenate(
                [obs['robot0_eef_pos'],
                 obs['robot0_eef_quat'],
                 obs['object-state'],
                 np.zeros(3),
                 obs['robot0_gripper_qpos'],
                 obs['robot0_gripper_qvel'],
                 np.ones(1),
                 self.previous_obs['robot0_eef_pos'],
                 self.previous_obs['robot0_eef_quat'],
                 self.previous_obs['object-state'],
                 np.zeros(3),
                 self.previous_obs['robot0_gripper_qpos'],
                 self.previous_obs['robot0_gripper_qvel'],
                 np.ones(1),
                self.previous_a[-2:-1],
                self.previous_r
                 ]))
        joint_obs = np.concatenate(joint_obs)

        return joint_obs

if __name__=='__main__':
    import robosuite
    from robosuite.controllers import load_controller_config

    # load default controller parameters for Operational Space Control (OSC)
    controller_config = load_controller_config(default_controller="OSC_POSE")

    env = robosuite.make(
        "TwoArmLift",
        robots=["Sawyer", "Panda"],  # load a Sawyer robot and a Panda robot
        gripper_types="default",  # use default grippers per robot arm
        controller_configs=controller_config,  # each arm is controlled using OSC
        env_configuration="single-arm-opposed",  # (two-arm envs only) arms face each other
        has_renderer=False,  # no on-screen rendering
        has_offscreen_renderer=False,  # no off-screen rendering
        control_freq=20,  # 20 hz control for applied actions
        horizon=200,  # each episode terminates after 200 steps
        use_object_obs=True,  # provide object observations to agent
        use_camera_obs=False,  # don't provide image observations to agent
        reward_shaping=True,  # use a dense reward signal for learning
    )

    env = RobosuiteWrapper(env,7,True)
    obs = env.reset()
    print(obs)