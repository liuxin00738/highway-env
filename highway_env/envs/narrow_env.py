import numpy as np
from gym.envs.registration import register

from highway_env import utils
from highway_env.envs.common.abstract import AbstractEnv
from highway_env.road.lane import LineType, StraightLane
from highway_env.road.road import Road, RoadNetwork
from highway_env.vehicle.kinematics import Vehicle
from highway_env.vehicle.controller import MDPVehicle


class NarrowEnv(AbstractEnv):

    """
    A risk management task: the agent is driving on a two-way lane with one oncoming vehicle.

    There are vehicles parked on the side of the road. It must balance making progress by
    overtaking and ensuring safety.

    These conflicting objectives are implemented by a reward signal and a constraint signal,
    in the CMDP/BMDP framework.
    """

    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update({
            "observation": {
                "type": "Kinematics",
                "features": ["presence", "x", "y", "vx", "vy"],
                "vehicles_count": 3,
                "features_range": {
                    "x": [-100, 100],
                    "y": [-100, 100],
                    "vx": [-20, 20],
                    "vy": [-20, 20]
                },
                "absolute": True,
                "order": "sorted",
                "normalize": False,
            },
            "action": {
                "type": "DiscreteMetaAction",
            },
            "collision_reward": -1,
            "left_lane_constraint": 1,
            "left_lane_reward": 0.2,
            "high_speed_reward": 0.8,
        })

        return config

    def _reward(self, action: int) -> float:
        """
        The vehicle is rewarded for driving with high speed
        :param action: the action performed
        :return: the reward of the state-action transition
        """

        # TODO(xin): implement reward
        return 0.0
        # neighbours = self.road.network.all_side_lanes(self.vehicle.lane_index)

        # reward = self.config["high_speed_reward"] * self.vehicle.speed_index / (self.vehicle.target_speeds.size - 1) \
        #     + self.config["left_lane_reward"] \
        #         * (len(neighbours) - 1 - self.vehicle.target_lane_index[2]) / (len(neighbours) - 1)
        # return reward

    def _is_terminal(self) -> bool:
        """The episode is over if the ego vehicle crashed or the time is out."""
        return self.vehicle.crashed

    def _cost(self, action: int) -> float:
        """The constraint signal is the time spent driving on the opposite lane, and occurrence of collisions."""
        return float(0.0)
        # TODO(xin): implement cost
        # return float(self.vehicle.crashed) + float(self.vehicle.lane_index[2] == 0)/15

    def _reset(self) -> np.ndarray:
        self._make_road()
        self._make_vehicles()

    def _make_road(self, length=200):
        """
        Make a road composed of a two-way road.

        :return: the road
        """
        net = RoadNetwork()

        # For narrow scenario, there is only one lane that overlap with each other. The road is set to be 3.5 times
        # the width of a vehicle: there will be vehilecs on both sides which can reduce the lane to be unpassable if
        # both vehicles try to pass at the same time.
        net.add_lane("a", "b", StraightLane([0, 0], [length, 0],
                                            line_types=(LineType.CONTINUOUS_LINE, LineType.CONTINUOUS_LINE),
                                            width=Vehicle.WIDTH*3.5, speed_limit=11.176))
        net.add_lane("b", "a", StraightLane([length, 0], [0, 0],
                                            line_types=(LineType.NONE, LineType.NONE),
                                            width=Vehicle.WIDTH*3.5, speed_limit=11.176))

        road = Road(network=net, np_random=self.np_random, record_history=self.config["show_trajectories"])
        self.road = road

    def _make_vehicles(self) -> None:
        """
        Populate a road with several vehicles on the road

        :return: the ego-vehicle
        """
        road = self.road
        ego_vehicle = self.action_type.vehicle_class(road,
                                                     road.network.get_lane(("a", "b", 0)).position(50, 0),
                                                     speed=3,
                                                     target_speeds=[0,2.5,5,7.5,10,12.5], name="ego_vehicle")
        road.vehicles.append(ego_vehicle)
        self.vehicle = ego_vehicle

        vehicles_type = utils.class_from_path(self.config["other_vehicles_type"])

        # Add stationary vehicles on the left and right of the lane with zero speed.
        for i in range(2):
            v = Vehicle(road, position=road.network.get_lane(("b", "a", 0))
                .position(50 + 100*self.np_random.uniform(), 2.5),
                heading=road.network.get_lane(("b", "a", 0)).heading_at(0),
                speed=0, name="left_vehicle_"+str(i))
            v.target_lane_index = ("b", "a", 0)
            self.road.vehicles.append(v)

        # Add stationary vehicles on the left and right of the lane with zero speed.
        for i in range(2):
            v = Vehicle(road, position=road.network.get_lane(("a", "b", 0))
                .position(50 + 100*self.np_random.uniform(), 2.5),
                heading=road.network.get_lane(("a", "b", 0)).heading_at(0),
                speed=0, name="right_vehicle_"+str(i))
            v.target_lane_index = ("a", "b", 0)
            self.road.vehicles.append(v)

        # Add one oncoming vehicle, which is simply following the lane
        for i in range(1):
            v = vehicles_type(road,
                              position=road.network.get_lane(("b", "a", 0))
                              .position(50 + 10*self.np_random.uniform(), 0),
                              heading=road.network.get_lane(("b", "a", 0)).heading_at(0),
                              speed=1 + 5*self.np_random.uniform(),
                              enable_lane_change=False, name="oncoming")
            v.target_lane_index = ("b", "a", 0)
            self.road.vehicles.append(v)


register(
    id='narrow-v0',
    entry_point='highway_env.envs:NarrowEnv',
    max_episode_steps=50
)
