import numpy
from PIL import ImageDraw, Image
from main import create_image
from main import DIR


def cal_pop_fitness(input_image_array, pop):
    # Calculating the fitness value of each solution in the current population.
    # The fitness function caulcuates the sum of products between each input and its corresponding weight.

    fitness = []
    for i in pop:
        diff = 0
        for j in range(input_image_array.shape[0]):
            if input_image_array[j][0] - i[j][0] < 50:
                diff += abs(input_image_array[j][0] - i[j][0])
            else:
                diff += 1000
            if input_image_array[j][1] - i[j][1] < 50:
                diff += abs(input_image_array[j][1] - i[j][1])
            else:
                diff += 1000
            if input_image_array[j][2] - i[j][2] < 50:
                diff += abs(input_image_array[j][2] - i[j][2])
            else:
                diff += 1000
        fitness.append(diff)
    return fitness


def find_nearest(array, value):
    array = numpy.asarray(array)
    idx = (numpy.abs(array - value)).argmin()
    return idx


def select_mating_pool(pop, fitness, num_parents, generation):
    # Selecting the best individuals in the current generation as parents for producing the offspring of the next generation.
    parents = numpy.empty((num_parents, pop.shape[1], pop.shape[2]))
    max_fitness_idx = find_nearest(array=fitness, value=0)
    create_image(pop, max_fitness_idx, DIR + '/gen' + str(generation) + '.jpg')
    for parent_num in range(num_parents):
        max_fitness_idx = find_nearest(array=fitness, value=0)
        parents[parent_num, :, :] = pop[max_fitness_idx, :, :]
        fitness[max_fitness_idx] = -99999999999
    return parents


def crossover(parents, offspring_size, input_image, new_image):
    offspring = numpy.empty(offspring_size)
    image = ImageDraw.Draw(new_image)
    crossover_point = offspring_size[0] // 2

    for k in range(offspring_size[0]):
        parent1_idx = k % parents.shape[0]
        parent2_idx = (k + 1) % parents.shape[0]

        new_image_original = Image.new('RGB', (512, 512), 'WHITE')
        image_original = ImageDraw.Draw(new_image_original)
        # rand_step = numpy.random.randint(1, 5)
        for i in range(input_image.shape[0]):
            y = i // 512
            x = i % 512
            radius_x = 10
            radius_y = 7
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
                image_original.ellipse((x - radius_x, y - radius_y, x + radius_x, y + radius_y),
                                       fill=(int(parents[parent1_idx][i][0]), int(parents[parent1_idx][i][1]),
                                             int(parents[parent1_idx][i][2])))

            elif dif_2 < 15:
                image.ellipse((x - radius_y, y - radius_x, x + radius_y, y + radius_x), fill=(
                    int(parents[parent2_idx][i][0]), int(parents[parent2_idx][i][1]), int(parents[parent2_idx][i][2])))
                image_original.ellipse((x - radius_y, y - radius_x, x + radius_y, y + radius_x), fill=(
                    int(parents[parent2_idx][i][0]), int(parents[parent2_idx][i][1]), int(parents[parent2_idx][i][2])))

        if numpy.random.randint(0, 2) > 0.5:
            offspring[k, 0:crossover_point] = parents[parent1_idx, 0:crossover_point]
            offspring[k, crossover_point:] = parents[parent2_idx, crossover_point:]
        else:
            offspring[k, 0:crossover_point] = parents[parent2_idx, 0:crossover_point]
            offspring[k, crossover_point:] = parents[parent1_idx, crossover_point:]
        new_image_original = numpy.array(new_image_original.getdata())
        for i in range(offspring_size[1]):
            if sum(new_image_original[i]) > 0:
                offspring[k][i] = new_image_original[i]
        # offspring = numpy.array(image.Image)
        # The new offspring will have its second half of its genes taken from the second parent.
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
    data = numpy.column_stack([input_image_array, pop.T])
    results = p.starmap(cal_pop_fitness, data)
    return results


def multi_selection(pop, fitness, num_parents, p):
    data = numpy.column_stack([pop, fitness, num_parents.T])
    results = p.starmap(select_mating_pool, data)
    return results


def multi_crossover(parents, offspring_size, input_image, new_image, p):
    data = numpy.column_stack([parents, offspring_size, input_image, new_image.T])
    results = p.starmap(crossover, data)
    return results


def multi_mutation(offspring_crossover, p):
    results = p.starmap(mutation, offspring_crossover)
    return results
