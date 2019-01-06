import random
import copy

c1 = 1.0
c2 = 1.0
c3 = 0.6


class Species:
    def __init__(self,genome1,genome2 = None):
        self.compatibility_list = []
        if genome2 == None:
            self.leader = genome1
            self.fitness_list = [genome1.fitness]
            self.best_fitness = genome1.fitness
            self.best_genome = genome1
            self.members = [genome1]
            self.compatibility_list.append(self.compatibility_score(genome1))
        else:
            self.leader = genome1
            self.fitness_list = [genome2.fitness]
            self.best_fitness = genome2.fitness
            self.best_genome = genome2
            self.members = [genome2]
            self.compatibility_list.append(self.compatibility_score(genome2))
        self.average_fitness = -999.99
        self.spawns_required = 0

    def compatibility_score(self,genome):
        # print(self.leader.fitness,genome.fitness)
        if self.leader.fitness > genome.fitness:
            gene1 = sorted(self.leader.connection_map.keys())
            gene1_weightsum = self.leader.weight_sum
            gene2 = sorted(genome.connection_map.keys())
            gene2_weightsum = genome.weight_sum
        else:
            gene1 = sorted(genome.connection_map.keys())
            gene1_weightsum = genome.weight_sum
            gene2 = sorted(self.leader.connection_map.keys())
            gene2_weightsum = self.leader.weight_sum
        # print("\t",gene1,gene1_weightsum)
        # print("\t",gene2,gene2_weightsum)
        # print("\n")
        max_inn = min(max(gene1),max(gene2))
        N = max(len(gene1),len(gene2))
        # print(max_inn)
        disjoint = 0
        excess = 0
        for inn in gene2:
            if inn>max_inn:
                excess +=1
            elif inn not in gene1:
                disjoint += 1

        for inn in gene1:
            if inn>max_inn:
                excess +=1
            elif inn not in gene2:
                disjoint += 1
        # print(excess,disjoint,abs(gene1_weightsum-gene2_weightsum))
        delta = (( (c1*excess) + (c2*disjoint) )/N ) + c3 * abs(gene1_weightsum-gene2_weightsum)
        # print(delta)
        return delta

    def add(self,genome):
        self.members.append(genome)
        self.compatibility_list.append(self.compatibility_score(genome))
        self.fitness_list.append(genome.fitness)
        if self.best_fitness < genome.fitness:
            self.best_fitness = genome.fitness
            self.best_genome = genome

    def reset_species(self):
        self.leader = copy.deepcopy(random.choice(self.members))
        self.best_fitness = -999.99
        self.members= []
        self.fitness_list = []

    def print_species(self):
        print("\tleader:",self.leader.return_nodes())
        print("\tleader fitness:",self.leader.fitness)
        print("\tfitness list :",self.fitness_list)
        print("\tcompatability list: ", self.compatibility_list)
        print("\tbest fitness : ", self.best_fitness)
        print("\tbest genome : ", self.best_genome.return_nodes())

        # print("\tmembers:")
        # for member in self.members:
        #     print("\t\t",member.return_nodes(),member.fitness)
