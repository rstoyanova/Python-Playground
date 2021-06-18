
# intersection of sets
def intersect(set1, set2):
    i = 0
    j = 0
    res = []
    sort_s1 = sorted(set1)
    sort_s2 = sorted(set2)
    while i < len(sort_s1) and j < len(sort_s2):
        if sort_s1[i] == sort_s2[j]:
            res.append(sort_s1[i])
        elif i < j:
            i += 1
        else:
            j += 1

    return res


# intersection of all lists in a list
def intersect_lists(list_of_lists):
    if not list_of_lists:
        return []
    res = list_of_lists[0]
    sorted_list = sorted(list_of_lists, key=lambda x: len(x))
    for i in range(1, len(list_of_lists)):
        res = intersect(res, sorted_list[i])

    return res


icko = intersect([1, 2, 3], [2, 3])
a = intersect_lists([[1, 2, 3, 4], [12, 34, 2, 34, 3], [1, 2, 3]])
print(icko)
