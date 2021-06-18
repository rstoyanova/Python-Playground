import matplotlib.pyplot as plt
import numpy as np
x = 5
# print(x)

f_list = ['a', 'b', 'c', 1, 2, 3, 4, 5]
# print(f_list)

nested_list = [1, 2, ["icko"]]
# print(nested_list)

last_el = nested_list[-1]
# print(last_el)

# print(f_list[2:5])


f_dictionary = {"liubov": "icko", "chushka": "ronche"}
# print(f_dictionary["liubov"])
"""
for i in range(5,10):
    #print(i)

ingredients = ["flour", "sugar", "eggs", "oil", "baking soda"]
for ingredient in ingredients:
    print(ingredient)


for i in range(10):
    if i % 2 == 0:
        print("even")
    elif i == 5:
        print("it five")
    else:
        print("odd")
"""
"""
even_list = [2, 4, 6, 8]
odd_list = [even + 1 for even in even_list]
print(odd_list)

x = np.array([2, 3, 4])
print(x)


# We'll start with a parabola
# Compute the parabola's x and y coordinates
x = np.arange(-5, 5, 0.1)
y = np.square(x)

# Use matplotlib for the plot
plt.plot(x, y, 'b')  # specify the color blue for the line

# for i in [1, 2, 3]:
#    print(i)

#name = input("Enter your name: ")
"""


def is_even(num):
    return num % 2 == 0


def chanche_num(num):
    num = 5


def increment(x):
    print(f"Initial address of x: {id(x)}")
    x += 1
    print(f"  Final address of x: {id(x)}")


def mult_returns(arr):
    a = arr[-1]
    b = arr[-2]
    return a, b


class Student:
    def __init__(self, name, age, major):
        self.name = name
        self.age = age
        self.major = major


def main():
    """
    num1 = 7
    num2 = 8
    chanche_num(num1)
    increment(num2)
    print(num1)
    print(num2)
    nums = [1, 2, 3, 4]
    t = mult_returns(nums)
    print(t)
    x, y = mult_returns(nums)
    print(x)
    """

    renka = Student("Renka", 22, "IT")

    renka.name = "Ne renka"
    print(renka.name)


main()
