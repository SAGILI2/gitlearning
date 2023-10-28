def verify_summation(sum_val, addends):
    return sum(addends) == sum(sum_val)


def verify_percents(numerator, denominator, percentages):
    for p in percentages:
        if 0.01 * p * denominator == numerator:
            return True
    return False


def numBitsChange(str1, str2):
    st1 = str(str1)
    st2 = str(str2)
    rng = min(len(st1), len(st2))
    numB = 0
    print("COMING INTO numBitsChange ,", st1, st2, rng)
    rev1 = [a for a in st1]
    rev2 = [a for a in st2]
    rev1.reverse()
    rev2.reverse()
    for ctr in range(rng):
        if rev1[ctr] != rev2[ctr]:
            numB += 1
    numB += abs(len(st1) - len(st2))
    print("ITSY BITSY - to change from ", str1, " to ", str2, " = ", numB)
    return numB


def get_bits_change_count(lhs_addends, rhs_addends):
    """

    :param lhs_addends: list of lhs nums
    :param rhs_addends: list of rhs nums
    :return:
    [minimum num of digits to be changed, changed number]
    """
    min_change_reqd = 1000
    changed_num = {}
    for num in lhs_addends:
        calc_num = sum(rhs_addends) - (sum(lhs_addends) - num)
        bitsChange = numBitsChange(num, calc_num)
        if bitsChange < min_change_reqd:
            min_change_reqd = bitsChange
            changed_num["old"] = num
            changed_num["new"] = calc_num
    for num in rhs_addends:
        calc_num = sum(lhs_addends) - (sum(rhs_addends) - num)
        bitsChange = numBitsChange(num, calc_num)
        if bitsChange < min_change_reqd:
            min_change_reqd = bitsChange
            changed_num["old"] = num
            changed_num["new"] = calc_num
    return [min_change_reqd, changed_num]


# print(get_bits_change_count([1,2,3],[1,23,6]))
