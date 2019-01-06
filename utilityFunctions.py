import copy
import csv
import os
import errno
import matplotlib.pyplot as plt


# define user directories
from Genome import *
from graphs import *
from Species import *

def visualize_population(population, generation, population_size):
    """
    creates visual representation of all networks in population
    :param population: current population
    :param generation: the current generation
    :param population_size : size of population
    :return none
    """
    for i in range(population_size):
        create_graph(population[i], generation, i)


def print_in_files(gen, population):
    """
    stores all the population data in files
    :param gen: current generation number
    :param population: current population
    :return: none
    """

    # set os path
    filename = 'C:/Users/Sravan Kumar/PycharmProjects/NEAT for forecasting/saved_data/gen'+str(gen)+'/data.csv'
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    lines = []    # create a line array
    number = 1    # create a number to keep track of which neural network we are using

    # append data to lines
    for genome in population:
        lines.append(["MEMBER "+str(number)])
        number += 1
        for node in genome.return_nodes():
            lines.append(node)
        lines.extend(genome.return_connections())
        lines.append(["fitness="+str(genome.fitness)])
        lines.append(["\n\n\n"])

    # add data into our data.csv stored in that generation file
    with open(filename, 'w') as writeFile:
        writer = csv.writer(writeFile)
        writer.writerows(lines)
    writeFile.close()


def visualize_species(species_list , generation):
    number = 1
    for sp in species_list:
        create_graph_best(sp.best_genome,generation,number)
        number+=1


def print_in_files_species(gen, species_list):
    # set os path
    filename = 'C:/Users/Sravan Kumar/PycharmProjects/NEAT for forecasting/saved_data/gen'+str(gen)+'/species1.csv'
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    lines = []    # create a line array
    number = 1    # create a number to keep track of which neural network we are using

    # append data to lines
    for sp in species_list:
        for genome in sp.members:
            lines.append(["MEMBER "+str(number)])
            number += 1
            for node in genome.return_nodes():
                lines.append(node)
            lines.extend(genome.return_connections())
            lines.append(["fitness="+str(genome.fitness)])

    # add data into our data.csv stored in that generation file
    with open(filename, 'w') as writeFile:
        writer = csv.writer(writeFile)
        writer.writerows(lines)
    writeFile.close()

def required_best_genome(genome , name=None):
    if name ==None:
        filename = 'C:/Users/Sravan Kumar/PycharmProjects/NEAT for forecasting/saved_data/best.csv'
    else:
        filename = 'C:/Users/Sravan Kumar/PycharmProjects/NEAT for forecasting/saved_data/'+str(name)+'.csv'
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    lines = []
    lines.extend(genome.return_nodes())
    lines.extend(genome.return_connections())
    lines.append(str(genome.fitness))
    # add data into our data.csv stored in that generation file
    with open(filename, 'w') as writeFile:
        writer = csv.writer(writeFile)
        writer.writerows(lines)
    writeFile.close()

def graph_data(fitness_list):
    plt.plot(fitness_list)
    plt.ylabel('fitness')
    plt.show()

