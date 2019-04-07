import multiprocessing

import numpy
import GA
from PIL import Image


def create_image(array, index, name):
    num_picture = [(i[0], i[1], i[2]) for i in array[index]]
    new_image = Image.new('RGB', (512, 512))
    new_image.putdata(num_picture)
    new_image.save(name)


def main(cores):
    sol_per_pop = 20
    num_parents_mating = 10
    colors = 3  # RGB

    input_image = Image.open("i2.jpg")
    width, height = input_image.size
    num_weights = width * height

    new_image = Image.new('RGB', (width, height), 'WHITE')
    input_image = numpy.array(input_image.getdata())

    # Defining the population size.
    pop_size = (sol_per_pop,
                num_weights,
                colors)

    # Creating the initial population.
    new_num_population = numpy.random.randint(low=0, high=256, size=pop_size)

    # for i in new_num_population:
    #     for j in range(len(i)):
    #         i[j][0] += input_image[j][0] / 4
    #         i[j][1] += input_image[j][1] / 4
    #         i[j][2] += input_image[j][2] / 4

    create_image(new_num_population, 0, 'generations/rand_generation.jpg')

    num_generations = 10

    p = multiprocessing.Pool(cores)

    for generation in range(num_generations):
        print("Generation : ", generation)


        # Measure the fitness of each chromosome in the population.
        fitness = GA.multi_fitness(input_image, new_num_population, p)


        # Selecting the best parents in the population for mating.
        parents = GA.multi_selection(new_num_population, fitness,
                                     num_parents_mating, p)

        # Generating next generation using crossover.
        offspring_crossover = GA.multi_crossover(parents,
                                                 (pop_size[0] - parents.shape[0], num_weights, colors), input_image,
                                                 new_image, p)

        # Adding some variations to the offsrping using mutation.
        offspring_mutation = GA.multi_mutation(offspring_crossover, p)

        # Creating the new population based on the parents and offspring.
        new_num_population[0:parents.shape[0], :, :] = parents
        new_num_population[parents.shape[0]:, :, :] = offspring_mutation

        # The best result in the current iteration.
        create_image(new_num_population, 0, 'generations/gen' + str(generation) + '.jpg')

    # Getting the best solution after iterating finishing all generations.
    # At first, the fitness is calculated for each solution in the final generation.
    fitness = GA.cal_pop_fitness(input_image, new_num_population)
    # Then return the index of that solution corresponding to the best fitness.
    best_match_idx = numpy.where(fitness == numpy.max(fitness))

    create_image(new_num_population, best_match_idx[0][0], 'generations/final_gen.jpg')


if __name__ == "__main__":
    cores = max(1, multiprocessing.cpu_count())
    multiprocessing.freeze_support()
    main(cores)
