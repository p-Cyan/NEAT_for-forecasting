# define user directories
from Species import *
from geneticAlgorithm import *
from utilityFunctions import *

# required global variables
input_size = 7
output_size = 1
population_size = 20
no_of_iterations = 20
compatibility_threshold = 7.0
no_of_ways = 3
best_genome = None
fitness_list = []
elitesize = 1

train_X, train_Y, dev_X, dev_Y, test_X, test_Y = getInputs2()

def main():
    """
    Implementation of NEAT to evaluate forecasting
    """
    global population_size,best_genome
    # initialize population
    population = initialize_population()
    for genome in population:
        evaluation(train_X,train_Y,genome)
        if best_genome == None:
            best_genome = copy.deepcopy(genome)
        elif best_genome.fitness < genome.fitness:
            best_genome = copy.deepcopy(genome)
    required_best_genome(best_genome,"first")
    species = speciate(population, [])

    # print , store data and visualize population
    # print_in_files(0, population)
    # visualize_population(population, 0, population_size)
    print_in_files_species(0,species)
    visualize_species(species,0)

    # start generations
    for generation in range(1, no_of_iterations + 1):
        print("generation = ", generation)
        # evaluate genome and calculate required values
        number = 1
        total_average = 0
        for sp in species:
            N = len(sp.members)
            fitness_sum = 0
            for genome in sp.members:
                number += 1
                genome.adjusted_fitness = genome.fitness / N
                fitness_sum += genome.fitness
            sp.average_fitness = fitness_sum / N
            total_average += sp.average_fitness

        # find no.of spawns required for each species
        for sp in species:
            print("member:")
            sp.print_species()
            print("\tspecies spawns:")
            sp.spawns_required = int(round(population_size * sp.average_fitness / total_average))
            print("\t\t", sp.spawns_required)

        population = []

        for sp in species:
            newlist = sorted(sp.members, key=lambda x: x.fitness, reverse=True)
            if elitesize <= len(sp.members) and elitesize <= sp.spawns_required:
                population.extend(newlist[:elitesize+1])
                sp.spawns_required -= elitesize

        offspring_species = []
        for sp in species:
            new_species = None
            for i in range(sp.spawns_required):
                gene1 = tournament_selection(sp, no_of_ways)
                gene2 = tournament_selection(sp, no_of_ways)
                child = crossover(copy.deepcopy(gene1), copy.deepcopy(gene2))
                mutation(child)
                evaluation(train_X,train_Y,child)
                if best_genome.fitness < child.fitness:
                    best_genome = child
                if new_species is None:
                    new_species = Species(copy.deepcopy(sp.leader),child)
                else:
                    new_species.add(child)
            if new_species is not None:
                offspring_species.append(new_species)

        # for sp in offspring_species:
        #     print("osp")
        #     sp.print_species()
        # for sp in offspring_species:
        #     print("sp")
        #     for member in sp.members:
        #         print(member.connection_map.keys(), member.fitness)
        species = offspring_species
        print_in_files_species(generation,species)
        visualize_species(species,generation)

        # convert species into population
        fitness_values = []
        for sp in species:
            for genome in sp.members:
                population.append(genome)
                fitness_values.append(genome.fitness)
        population_size = len(population)
        fitness_list.append(max(fitness_values))
        # print_in_files(generation, population)
        # visualize_population(population, generation, len(population))
        print("POPULATION_SIZE: ",population_size)

        # speciate

        species = speciate(population, species)


    print(best_genome.fitness)
    required_best_genome(best_genome)
    evaluation(train_X,train_Y,best_genome,check=True)
    graph_data(fitness_list)



def initialize_population():
    """
    creates a new population containing
    :return: a list of genome objects containing only input and output nodes with dense connections among them
    """
    # initialize nodes
    temp_list = [Genome(input_size, output_size)]
    for i in range(population_size - 1):
        temp_list.append(copy.deepcopy(temp_list[0]))

    # initialize connections
    for genome in temp_list:
        genome.initialize_connections()

    return temp_list


def speciate(population, prev_species=None):
    if prev_species is None:
        species_list = []
    else:
        species_list = prev_species
    for species in species_list:
        species.reset_species()
    for genome in population:
        flag = False
        for species in species_list:
            if species.compatibility_score(genome) < compatibility_threshold:
                species.add(genome)
                flag = True
                break
        if flag is False:
            species_list.append(Species(copy.deepcopy(genome)))
    index = 0
    copyspecies = copy.deepcopy(species_list)
    for species in copyspecies:
        if len(species.members) == 0:
            species_list.pop(index)
        else:
            index += 1

    # number = 1
    # for species in species_list:
    #     print("species ",number,":")
    #     number+=1
    #     for member in species.members:
    #         print("\t", member.connection_map.keys(), member.fitness)
    #     print(species.leader.fitness, species.best_fitness, species.best_genome.connection_map.keys())
    #     print("\tleader:\t", species.leader.connection_map.keys())
    return species_list


if __name__ == "__main__":
    print(main.__doc__)
    main()
