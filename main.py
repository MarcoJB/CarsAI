import time

import numpy as np

from tkinter import *

from shapely.geometry import LinearRing, Point

from NeuralNetwork import NeuralNetwork
from Car import Car

population = 12

route = LinearRing([(100, 100), (700, 100), (900, 300), (900, 600), (800, 600), (500, 300), (400, 300), (100, 600)])
route_2d = route.buffer(50).simplify(10)
print(len(route_2d.exterior.coords.xy[0]))

key_states = {
    "Down": False,
    "Up": False,
    "Right": False,
    "Left": False
}

cars = []
configuration = [5, 3, 2]
for k in range(population):
    network = NeuralNetwork(configuration[0], configuration[-1], configuration[1:-1])
    # network.weights_matrices[0] = np.array([
    #     [-0.5, -0.5, 0, 1, 0, -0.5, 0],
    #     [0, -1, -0.25, 0, 0.25, 1, 0]
    # ])
    cars.append(Car(route, route_2d, network, 200, 100))

# def keypress(e):
#     key_states[e.keysym] = True
#
#
# def keyrelease(e):
#     key_states[e.keysym] = False

positions = []
for layer in range(len(configuration)):
    positions.append([])

    x = 25 + layer * 100
    offset_y = 25 + (np.max(configuration) - configuration[layer]) * 25

    for neuron in range(configuration[layer] + (layer < len(configuration) - 1)):
        y = offset_y + neuron * 50

        positions[layer].append((x, y))


def drawNetwork(network):
    network_canvas.delete("all")

    for layer_index, layer in enumerate(positions):
        for neuron_index, neuron in enumerate(layer):
            if neuron_index == len(layer) - 1 and layer_index != len(positions) - 1:
                neuron_value = 1
            else:
                neuron_value = network.neuron_values[layer_index][neuron_index]
                neuron_value = np.min([neuron_value, 1])
                neuron_value = np.max([-1, neuron_value])

            if neuron_value > 0:
                color = "#{:02X}{:02X}{:02X}".format(int((1 - neuron_value) * 255), 255, int((1 - neuron_value) * 255))
            else:
                color = "#{:02X}{:02X}{:02X}".format(255, int((1 + neuron_value) * 255), int((1 + neuron_value) * 255))

            network_canvas.create_oval(
                neuron[0] - 10,
                neuron[1] - 10,
                neuron[0] + 10,
                neuron[1] + 10,
                fill=color
            )

    for layer, weights in enumerate(network.weights_matrices):
        for neuron_in_next_layer in range(len(weights)):
            for neuron_in_current_layer in range(len(weights[neuron_in_next_layer])):
                weight = np.tanh(weights[neuron_in_next_layer][neuron_in_current_layer])

                if weight > 0:
                    line_color = "#{:02X}{:02X}{:02X}".format(int((1-weight)*255), 255, int((1-weight)*255))
                else:
                    line_color = "#{:02X}{:02X}{:02X}".format(255, int((1+weight)*255), int((1+weight)*255))

                network_canvas.create_line(
                    positions[layer][neuron_in_current_layer][0],
                    positions[layer][neuron_in_current_layer][1],
                    positions[layer + 1][neuron_in_next_layer][0],
                    positions[layer + 1][neuron_in_next_layer][1],
                    width=4*np.abs(weight),
                    fill=line_color
                )


master = Tk()
master.title("CarsAI")
canvas = Canvas(master, width=1000, height=700)
canvas.grid(row=0, column=0)

diagram = Canvas(master, width=1000, height=200)
diagram.grid(row=1, column=0)

network_canvas = Canvas(master, width=(len(configuration) - 1) * 100 + 50, height=(np.max(configuration)) * 50 + 50)
network_canvas.grid(row=0, column=1)

label = Label(master, text="Generation 1")
label.grid(row=3, column=0)

highscores = [0]
avg_best_scores = [0]
avg_scores = [0]

# master.bind("<KeyPress>", keypress)
# master.bind("<KeyRelease>", keyrelease)

old_time = time.time()


def getScore(car):
    return car.getScore()


def getRandomizedRank(car):
    return car.randomized_rank


generation = 1
counter = 0

while True:
    # if key_states["Up"]:
    #     car.acceleration = 1
    # elif key_states["Down"]:
    #     car.acceleration = -1
    # else:
    #     car.acceleration = 0
    #
    # if key_states["Right"]:
    #     car.omega = 0.1
    # elif key_states["Left"]:
    #     car.omega = -0.1
    # else:
    #     car.omega = 0

    canvas.delete("all")
    canvas.create_polygon([int(n) for n in np.array(route_2d.exterior.coords.xy).transpose().flatten()], fill="#ccc")
    canvas.create_polygon([int(n) for n in np.array(route_2d.interiors[0].coords.xy).transpose().flatten()],
                          fill="#fff")

    dead_counter = 0
    best_alive = None

    old_time = time.time()
    for car in cars:
        car.step(1)

        colors = np.array([0, 0, 0])
        if car.round == 1:
            colors = np.array([150, 0, 0])
        elif car.round == 2:
            colors = np.array([150, 150, 0])
        elif car.round == 3:
            colors = np.array([0, 150, 0])
        elif car.round == 4:
            colors = np.array([0, 150, 150])
        elif car.round >= 5:
            colors = np.array([0, 0, 150])

        if not car.alive:
            dead_counter += 1
            colors = 255 - (255 - colors) * 0.4

        canvas.create_polygon([int(n) for n in np.array(car.getShape().exterior.coords.xy).transpose().flatten()],
                              fill="#{:02X}{:02X}{:02X}".format(int(colors[0]), int(colors[1]), int(colors[2])))
        if car.alive:
            for ray in car.rays:
                if not ray.is_empty:
                    canvas.create_line([int(n) for n in np.array(ray.coords.xy).transpose().flatten()])

        if car.alive and (best_alive is None or car.getScore() > best_alive.getScore()):
            best_alive = car

    if best_alive is not None:
        drawNetwork(best_alive.neuralNetwork)

    # print(time.time()-old_time)

    # if counter > 100 + 5*generation:
    #     for car in cars:
    #         car.die()

    if dead_counter == population:
        # break

        cars.sort(key=getScore, reverse=True)

        print(cars[0].neuralNetwork.weights_matrices[0])

        for i, car in enumerate(cars):
            car.randomized_rank = i * np.random.rand()
        cars.sort(key=getRandomizedRank)

        # drawNetwork(cars[0].neuralNetwork)

        avg_best = 0
        avg_score = 0
        for i, car in enumerate(cars):
            if i < int(population / 3):
                avg_best += car.getScore()
            avg_score += car.getScore()
        avg_best /= int(population / 3)
        avg_score /= population
        avg_best_scores.append(avg_best)
        avg_scores.append(avg_score)
        highscores.append(cars[0].getScore())

        cars = cars[0:int(population / 3)]

        print("Generation {}: Best rating: {}, Average rating: {}".format(generation, highscores[-1], avg_scores[-1]))

        for i in range(len(cars)):
            # cars[i].reset()

            network = cars[i].neuralNetwork.clone()
            network.mutate(0.1, 0.1)
            cars.append(Car(route, route_2d, network, 200, 100))

            network = cars[i].neuralNetwork.clone()
            network.mutate(0.1, 0.2)
            cars.append(Car(route, route_2d, network, 200, 100))

        highscore_points = []
        avg_best_points = []
        avg_score_points = []

        for i in range(generation + 1):
            highscore_points.append(i / generation * 1000)
            highscore_points.append(195 - highscores[i] / highscores[-1] * 190)
            avg_best_points.append(i / generation * 1000)
            avg_best_points.append(195 - avg_best_scores[i] / highscores[-1] * 190)
            avg_score_points.append(i / generation * 1000)
            avg_score_points.append(195 - avg_scores[i] / highscores[-1] * 190)

        diagram.delete("all")

        for i in range(int(highscores[-1] * 4) + 1):
            diagram.create_line([0, 195 - i / 4 / highscores[-1] * 190, 1000, 195 - i / 4 / highscores[-1] * 190],
                                fill="#ddd")
        for i in range(int(highscores[-1]) + 1):
            diagram.create_line([0, 195 - i / highscores[-1] * 190, 1000, 195 - i / highscores[-1] * 190], fill="#bbb")
        for i in range(int(highscores[-1] / 5) + 1):
            diagram.create_line([0, 195 - i * 5 / highscores[-1] * 190, 1000, 195 - i * 5 / highscores[-1] * 190],
                                fill="#999")

        diagram.create_line(highscore_points, fill="#b00")
        diagram.create_line(avg_best_points, fill="#55c")
        diagram.create_line(avg_score_points, fill="#5c5")

        generation += 1
        label.config(text="Generation {}".format(generation))

        counter = 0
    else:
        counter += 1

    master.update_idletasks()
    master.update()

    time.sleep(0.01)
