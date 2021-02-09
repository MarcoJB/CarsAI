import numpy as np
from shapely import affinity
from shapely.geometry import Polygon, LineString, LinearRing, MultiLineString, Point, MultiPoint

from NeuralNetwork import NeuralNetwork


class Car:
    velocity = 0
    angle_velocity = 0
    alive = True
    round = 0
    progress = 0
    current_step = 0
    rays = []
    randomized_rank = 0

    def __init__(self, route: LinearRing, route_2d: Polygon, neuralNetwork: NeuralNetwork, x: float = 0, y: float= 0, rotation: float = 0):
        self.route = route
        self.route_2d = route_2d
        self.neuralNetwork = neuralNetwork
        self.scores = []

        self.initial_position = {
            "x": x,
            "y": y
        }
        self.position = {
            "x": x,
            "y": y
        }
        self.initial_rotation = rotation
        self.rotation = rotation

    def step(self, time):
        if not self.alive:
            return

        self.calcRays()
        acceleration, angle_velocity = self.neuralNetwork.calc([
            self.velocity / 2,
            self.rays[0].length / 50,
            self.rays[1].length / 50,
            self.rays[2].length / 50,
            # self.rays[3].length / 500,
            # self.rays[4].length / 500,
            self.velocity / 2,
        ])

        self.velocity += acceleration / 2 * time
        self.rotation += angle_velocity / 5 * time

        self.position["x"] += np.cos(self.rotation) * self.velocity * time
        self.position["y"] += np.sin(self.rotation) * self.velocity * time
        # self.rotation += self.angle_velocity * time

        progress = self.getProgress()
        if progress < self.progress - 0.5:
            self.round += 1
        elif progress > self.progress + 0.5:
            self.round -= 1
        self.progress = progress

        # if self.round == 5:
        #     self.die()
        #     self.progress = 3000 / self.current_step

        self.scores.append(self.getScore())
        if len(self.scores) > 50:
            if self.scores[-1] - self.scores[0] < self.current_step / 5000:
                self.die()
            self.scores = self.scores[1:]

        if not self.route_2d.contains(self.getShape()):
            self.die()

        self.current_step += 1

    def die(self):
        if not self.alive:
            return

        self.alive = False

    def getProgress(self):
        position = Point((self.position["x"], self.position["y"]))
        # return (1 - self.route_2d.exterior.project(position, True) + self.route_2d.interiors[0].project(position, True)) / 2
        # print(self.route.project(position, True), 1-self.route_2d.exterior.project(position, True), self.route_2d.interiors[0].project(position, True))

        return 1-self.route_2d.exterior.project(position, True)

    def getShape(self):
        shape = Polygon([(-15, -10), (15, -10), (15, 10), (-15, 10)])
        shape = affinity.rotate(shape, self.rotation, use_radians=True)
        shape = affinity.translate(shape, xoff=self.position["x"], yoff=self.position["y"])

        return shape

    def calcRays(self):
        # rays = [
        #     LineString([(0, 0), (1000, 0)]),
        #     LineString([(0, 0), (1000, 0)]),
        #     LineString([(0, 0), (1000, 0)]),
        #     LineString([(0, 0), (1000, 0)]),
        #     LineString([(0, 0), (1000, 0)])
        # ]
        #
        # rays[0] = affinity.rotate(rays[0], -60, origin=(0, 0))
        # rays[1] = affinity.rotate(rays[1], -30, origin=(0, 0))
        # rays[3] = affinity.rotate(rays[3], 30, origin=(0, 0))
        # rays[4] = affinity.rotate(rays[4], 60, origin=(0, 0))

        rays = []

        for ray in range(-1, 2):
            rays.append(LineString([(self.position["x"], self.position["y"]), (
                self.position["x"] + 1000 * np.cos(self.rotation + ray*np.pi/3),
                self.position["y"] + 1000 * np.sin(self.rotation + ray*np.pi/3)
            )]))
            rays[-1] = rays[-1].intersection(self.route_2d)
            if type(rays[-1]) == MultiLineString:
                rays[-1] = rays[-1][0]


            # rays[ray] = affinity.rotate(rays[ray], self.rotation, origin=(0, 0), use_radians=True)
            # rays[ray] = affinity.translate(rays[ray], xoff=self.position["x"], yoff=self.position["y"])
            # outer_crossing = long_ray.intersection(self.route_2d.exterior)
            # if type(outer_crossing) == MultiPoint:
            #     outer_crossing = outer_crossing[0]
            #
            # inner_crossing = long_ray.intersection(self.route_2d.interiors[0])
            # if type(inner_crossing) == MultiPoint:
            #     inner_crossing = inner_crossing[0]
            #
            # if inner_crossing.is_empty and outer_crossing.is_empty:
            #     rays.append(inner_crossing)
            # elif inner_crossing.is_empty:
            #     rays.append(LineString([(self.position["x"], self.position["y"]), outer_crossing]))
            # elif outer_crossing.is_empty:
            #     rays.append(LineString([(self.position["x"], self.position["y"]), inner_crossing]))
            # else:
            #     outer_ray = LineString([(self.position["x"], self.position["y"]), outer_crossing])
            #     inner_ray = LineString([(self.position["x"], self.position["y"]), inner_crossing])
            #
            #     if outer_ray.length < inner_ray.length:
            #         rays.append(outer_ray)
            #     else:
            #         rays.append(inner_ray)



        self.rays = rays

    def getScore(self):
        return self.round + self.progress

    def reset(self):
        self.alive = True
        self.progress = 0
        self.round = 0
        self.velocity = 0
        self.angle_velocity = 0
        self.rotation = self.initial_rotation
        self.position["x"] = self.initial_position["x"]
        self.position["y"] = self.initial_position["y"]
        self.scores = []
        self.current_step = 0

