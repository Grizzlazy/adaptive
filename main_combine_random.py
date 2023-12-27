import Data
import Function
import Neighborhood
import time
import random
import Cuongs_nei
global LOOP
global tabu_tenure
global best_sol
global best_fitness
global Tabu_Structure
global current_neighborhood

# Set up chỉ số -------------------------------------------------------------------
tabu_tenure = 8
LOOP = 70
ITE = 3
epsilon = (-1)*0.00001
BREAKLOOP = 30
# Set up chỉ số -------------------------------------------------------------------

# Hai mẫu bị lỗi khi chạy bộ 10 C101 1 that vào ở initial solution để thử 
check = [[[[0, [1, 2, 5]], [1, [3]], [2, [4, 10, 7]], [5, []], [3, []], [4, [6, 8]], [10, []], [7, []], [6, []], [8, [9]], [9, []]]], [[[1, [3]]], [[2, [4, 10, 7]]], [[4, [6, 8]]], [[8, [9]]]]]
check2 = [[[[0, [1, 5]], [1, [2, 3]], [2, []], [5, [4, 7, 6]], [3, []], [4, [8]], [10, [10, 9]], [7, []], [6, []], [8, []], [9, []]]], [[[1, [2, 3]]], [[5, [4, 7, 6]]], [[4, [8]]], [[10, [10, 9]]]]]
# Hai mẫu bị lỗi khi chạy bộ 10 C101 1 that vào ở initial solution để thử 


def Tabu_search_for_CVRP():
    global current_neighborhood
    Tabu_Structure = [tabu_tenure * (-1)] * Data.number_of_cities
    Tabu_Structure1 = []
    for i in range(Data.number_of_cities):
        row = [tabu_tenure * (-1)] * Data.number_of_cities
        Tabu_Structure1.append(row)
    current_sol1 = Function.initial_solution1()
    current_sol2 = Function.initial_solution()
    fitness1 = Function.fitness(current_sol1)
    fitness2 = Function.fitness(current_sol2)
    if fitness1[0] > fitness2[0]:
        current_sol = current_sol2
    else:
        current_sol = current_sol1
    
    # Initial solution thay ở đây ------------->
    #current_sol = check2     # Để dòng này làm comment để tìm initial solution theo tham lam
    # <------------- Initial solution thay ở đây 
    
    
    current_fitness, current_truck_time = Function.fitness(current_sol)
    best_sol = current_sol
    best_fitness = current_fitness
    print(best_sol) 
    print(best_fitness)
    print(Function.Check_if_feasible(best_sol))
    for i in range(LOOP):
        cuong = random.random()
        cuong = int(2*cuong)
        print("----------",i,"------------")
        if cuong == 0 and i % BREAKLOOP > 0:
            current_neighborhood = Cuongs_nei.two_swap(current_sol)
            best_nei = -1 # Chỉ số để xem chạy neighborhood nào
            index = -1
            min_nei = 1000000 
            for j in range(len(current_neighborhood)):
                cfnode = current_neighborhood[j][1][0]
                if cfnode - best_fitness < epsilon:
                    min_nei = cfnode
                    index = j
                    best_fitness = cfnode
                    best_sol = current_neighborhood[j][0]

                elif cfnode - min_nei < epsilon and Tabu_Structure1[current_neighborhood[j][2][0]][current_neighborhood[j][2][1]] + tabu_tenure <= i:
                    min_nei = cfnode
                    index = j
            #print(i)
            current_sol = current_neighborhood[index][0]
            current_fitness = current_neighborhood[index][1][0]
            Tabu_Structure1[current_neighborhood[index][2][0]][current_neighborhood[index][2][1]] = i
            Tabu_Structure1[current_neighborhood[index][2][1]][current_neighborhood[index][2][0]] = i
            #print("change in: ",current_neighborhood[index][2][0]," and ",current_neighborhood[index][2][1])
        else:
            #print("----------",i,"------------")
            current_neighborhood = []
            nei_set = [0, 1, 2, 3]
            choose = random.random()
            choose = int(choose*4)
            if i % BREAKLOOP == 0:
                current_neighborhood5 = Neighborhood.swap_two_array(current_sol)
                current_neighborhood.append(current_neighborhood5)
            else:
                if choose == 0:
                    current_neighborhood1 = Neighborhood.Neighborhood_one_otp(current_sol, current_truck_time)
                    current_neighborhood.append(current_neighborhood1)
                elif choose == 1:
                    current_neighborhood2 = Neighborhood.Neighborhood_group_trip(current_sol, current_fitness)
                    current_neighborhood.append(current_neighborhood2)
                elif choose == 2:
                    current_neighborhood3 = Neighborhood.Neighborghood_change_drone_route(current_sol)
                    current_neighborhood.append(current_neighborhood3)
                elif choose == 3:
                    current_neighborhood4 = Neighborhood.Neighborhood_one_otp_plus(current_sol, current_truck_time)
                    current_neighborhood.append(current_neighborhood4)
            index = [-1] * len(current_neighborhood)
            min_nei = [100000] * len(current_neighborhood)
            for j in range(len(current_neighborhood)):
                for k in range(len(current_neighborhood[j])):
                    if i % BREAKLOOP == 0:
                        cfnode = current_neighborhood[j][k][1][0]
                        if cfnode - best_fitness < epsilon:
                            min_nei[j] = cfnode
                            index[j] = k
                            best_fitness = cfnode
                            best_sol = current_neighborhood[j][k][0]
                        elif cfnode - min_nei[j] < epsilon:
                            min_nei[j] = cfnode
                            index[j] = k
                    else:
                        if choose == 0 or choose == 3:
                            cfnode = current_neighborhood[j][k][1][0]
                            if cfnode - best_fitness < epsilon:
                                min_nei[j] = cfnode
                                index[j] = k
                                best_fitness = cfnode
                                best_sol = current_neighborhood[j][k][0]

                            elif cfnode - min_nei[j] < epsilon and Tabu_Structure[current_neighborhood[j][k][2]]+ tabu_tenure <= i:

                                min_nei[j] = cfnode
                                index[j] = k
                        else:
                            cfnode = current_neighborhood[j][k][1][0]
                            if cfnode - best_fitness < epsilon:
                                min_nei[j] = cfnode
                                index[j] = k
                                best_fitness = cfnode
                                best_sol = current_neighborhood[j][k][0]
                            elif cfnode - min_nei[j] < epsilon:
                                min_nei[j] = cfnode
                                index[j] = k
            temp = []
            for j in range(len(current_neighborhood)):
                temp.append([min_nei[j],j])
            temp.sort(key=lambda x:x[0])
            best_nei = temp[0][1]
            if len(current_neighborhood[best_nei]) == 0:
                continue
            current_sol = current_neighborhood[best_nei][index[best_nei]][0]
            current_fitness = current_neighborhood[best_nei][index[best_nei]][1][0]
            current_truck_time = current_neighborhood[best_nei][index[best_nei]][1][1]
            if (choose == 0 or choose == 3 ) and i % BREAKLOOP > 0:
                Tabu_Structure[current_neighborhood[best_nei][index[best_nei]][2]] = i
        '''if cuong == 0:
            print("Neighborhood cua Cuong: two otp")
        elif choose == 0:
            print ("Neighborhood cua Luyen: one otp")
            print("Thành phố được chuyển là thành phố", current_neighborhood[best_nei][index[best_nei]][2])
        elif choose == 1:
            print ("Neighborhood cua Luyen: group trip to another drone trip")
        elif choose == 2:
            print ("Neighborhood cua Luyen: change drone route")
        elif choose == 3:
            print ("Neighborhood cua Luyen: one otp plus")
            print("Thành phố được chuyển là thành phố", current_neighborhood[best_nei][index[best_nei]][2])
'''
        for j in range(Data.number_of_trucks):
            print(current_sol[0][j])
        print(current_sol[1])
        print(current_fitness)
        print(Function.Check_if_feasible(current_sol))
        # print(current_sol)
        # print(current_fitness)
        # print(Function.Check_if_feasible(current_sol))
    return best_fitness, best_sol



result = []
run_time = []
for i in range(ITE):
    print("------------------------",i,"------------------------")
    start_time = time.time()
    best_fitness, best_sol = Tabu_search_for_CVRP()
    print("---------- RESULT ----------")
    print(best_sol)
    print(best_fitness)
    result.append(best_fitness)
    print(Function.Check_if_feasible(best_sol))
    end_time = time.time()
    run = end_time - start_time
    run_time.append(run)
print(result)
print(run_time)
