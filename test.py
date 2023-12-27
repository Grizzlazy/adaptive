import random
import Function
import Neighborhood
import Neighborhood10
import Neighborhood_drone
import math
import Data

# a = Function.initial_solution5()

# print(a)
# print("distance_truck1: ", Function.cal_distance_of_truck(a, 0))
# print("truck1: ", Function.cal_truck_time(a)[0])
# print("distance_truck2: ", Function.cal_distance_of_truck(a, 1))
# print("truck2: ", Function.cal_truck_time(a)[1])

current_sol5 = [[[[0, [13]], [8, [8, 15, 6, 14]], [9, [9, 12]], [12, []], [14, []], [15, []], [13, []], [6, []]], 
                 [[0, [4, 7, 3, 1]], [10, [10, 11, 2, 5]], [11, []], [2, []], [5, []], [4, []], [7, []], [3, []], [1, []]]], 
                [[[8, [8, 15, 6, 14]]], [[9, [9, 12]]], [[10, [10, 11, 2, 5]]]]]
# print(current_sol5)
a = [5, 4, 11, 1, 3, 2, 10, 13]
print(Function.max_release_date(a))
print(Function.min_release_date(a))
print(Data.standard_deviation)

