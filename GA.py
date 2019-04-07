import numpy
import multiprocessing
from PIL import Image, ImageDraw


def cal_pop_fitness(input_image_array, pop):
    # Calculating the fitness value of each solution in the current population.
    # The fitness function caulcuates the sum of products between each input and its corresponding weight.

    # fitness = numpy.sum(pop * equation_inputs, axis=1)
    fitness = []

    for i in pop:
        diff = 0
        for j in range(input_image_array.size // 3):
            if input_image_array[j][0] - i[j][0] < 50:
                diff += 255 - input_image_array[j][0] - i[j][0]
            else:
                diff += 1000
            if input_image_array[j][1] - i[j][1] < 50:
                diff += 255 - input_image_array[j][1] - i[j][1]
            else:
                diff += 1000
            if input_image_array[j][2] - i[j][2] < 50:
                diff += 255 - input_image_array[j][2] - i[j][2]
            else:
                diff += 1000
        fitness.append(diff)
    return fitness
    # for i in pop:
    #     diff_r = 0
    #     diff_g = 0
    #     diff_b = 0
    #     for j in zip(equation_inputs, i):
    #         diff_r += j[0][0] - j[1][0]
    #         diff_g += j[0][1] - j[1][1]
    #         diff_b += j[0][2] - j[1][2]
    #     fitness.append([diff_r, diff_g, diff_b])
    # return fitness


def find_nearest(array, value):
    array = numpy.asarray(array)
    idx = (numpy.abs(array - value)).argmin()
    return idx


def select_mating_pool(pop, fitness, num_parents):
    # Selecting the best individuals in the current generation as parents for producing the offspring of the next generation.
    parents = numpy.empty((num_parents, pop.shape[1], pop.shape[2]))
    for parent_num in range(num_parents):
        max_fitness_idx = find_nearest(array=fitness, value=0)
        parents[parent_num, :, :] = pop[max_fitness_idx, :, :]
        fitness[max_fitness_idx] = -99999999999
    return parents


def crossover(parents, offspring_size, input_image, new_image):
    offspring = numpy.empty(offspring_size)
    # The point at which crossover takes place between two parents. Usually it is at the center.
    crossover_point = offspring_size[1] // 2

    image = ImageDraw.Draw(new_image)

    for k in range(offspring_size[0]):
        # Index of the first parent to mate.
        parent1_idx = k % parents.shape[0]
        # Index of the second parent to mate.
        parent2_idx = (k + 1) % parents.shape[0]
        # The new offspring will have its first half of its genes taken from the first parent.

        for i in range(input_image.shape[0]):
            y = i // 512
            x = i % 512
            radius_x = 3
            radius_y = 2
            dif_1 = 0
            dif_2 = 0
            dif_1 += abs(input_image[i][0] - parents[parent1_idx][i][0])
            dif_1 += abs(input_image[i][1] - parents[parent1_idx][i][1])
            dif_1 += abs(input_image[i][2] - parents[parent1_idx][i][2])
            dif_2 += abs(input_image[i][0] - parents[parent2_idx][i][0])
            dif_2 += abs(input_image[i][1] - parents[parent2_idx][i][1])
            dif_2 += abs(input_image[i][2] - parents[parent2_idx][i][2])
            if dif_1 < dif_2 and dif_1 < 15:
                image.ellipse((x - radius_x, y - radius_y, x + radius_x, y + radius_y),
                              fill=(int(parents[parent1_idx][i][0]), int(parents[parent1_idx][i][1]),
                                    int(parents[parent1_idx][i][2])))

            elif dif_2 < 15:
                image.ellipse((x - radius_y, y - radius_x, x + radius_y, y + radius_x), fill=(
                    int(parents[parent2_idx][i][0]), int(parents[parent2_idx][i][1]), int(parents[parent2_idx][i][2])))

        offspring[k] = numpy.array(new_image.getdata())
        # offspring = numpy.array(image.Image)
        # offspring[k, 0:crossover_point] = parents[parent1_idx, 0:crossover_point]
        # The new offspring will have its second half of its genes taken from the second parent.
        # offspring[k, crossover_point:] = parents[parent2_idx, crossover_point:]
    return offspring


def mutation(offspring_crossover):
    # Mutation changes a single gene in each offspring randomly.
    for idx in range(offspring_crossover.shape[0]):
        # The random value to be added to the gene.
        random_amount_gen = numpy.random.randint(0, offspring_crossover.shape[1] // 10)
        for i in range(random_amount_gen):
            random_value = numpy.random.uniform(-255.0, 256.0, 3)
            random_gen = numpy.random.randint(0, offspring_crossover.shape[1])
            offspring_crossover[idx, random_gen] = offspring_crossover[idx, random_gen] + random_value
    return offspring_crossover


def multi_fitness(input_image_array, pop, p):
    results = p.apply(cal_pop_fitness, (input_image_array, pop,))
    return results


def multi_selection(pop, fitness, num_parents, p):
    results = p.apply(select_mating_pool, (pop, fitness, num_parents,))
    return results


def multi_crossover(parents, offspring_size, input_image, new_image, p):
    results = p.apply(crossover, (parents, offspring_size, input_image, new_image,))
    return results


def multi_mutation(offspring_crossover, p):
    results = p.apply(mutation, (offspring_crossover,))
    return results
