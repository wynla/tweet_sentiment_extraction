def permute(nums: int):
    def dfs(nums, size, depth, path, used, res):
        if depth == size:
            res.append(path)
            return

        for i in range(size):
            if not used[i]:
                used[i] = True
                path.append(nums[i])

                dfs(nums, size, depth + 1, path, used, res)

                used[i] = False
                path.pop()

    size = len(nums)
    if len(nums) == 0:
        return []

    used = [False for _ in range(size)]
    res = []
    dfs(nums, size, 0, [], used, res)
    return res


def dfs(nums, size, path, res, used):
    if len(path) == size:
        res.append(path.copy())
        return
    for i in range(size):
        if used[i]: continue
        if i > 0 and nums[i] == nums[i - 1] and used[i - 1]: break
        used[i] = True
        path.append(nums[i])
        dfs(nums, size, path, res, used)
        used[i] = False
        path.pop()


phone = {'2': ['a', 'b', 'c'],
         '3': ['d', 'e', 'f'],
         '4': ['g', 'h', 'i'],
         '5': ['j', 'k', 'l'],
         '6': ['m', 'n', 'o'],
         '7': ['p', 'q', 'r', 's'],
         '8': ['t', 'u', 'v'],
         '9': ['w', 'x', 'y', 'z']}


def dfs(nums, size, path, res, phone):
    if len(path) == size:
        res.append(''.join(path.copy()))
        return
    for j in phone[nums[0]]:
        path.append(j)
        dfs(nums[1:], size, path, res, phone)
        path.pop()


# def letterCombinations(digits):
#     """
#     :type digits: str
#     :rtype: List[str]
#     """
#     phone = {'2': ['a', 'b', 'c'],
#              '3': ['d', 'e', 'f'],
#              '4': ['g', 'h', 'i'],
#              '5': ['j', 'k', 'l'],
#              '6': ['m', 'n', 'o'],
#              '7': ['p', 'q', 'r', 's'],
#              '8': ['t', 'u', 'v'],
#              '9': ['w', 'x', 'y', 'z']}
#
#     def backtrack(combination, next_digits):
#         # if there is no more digits to check
#         if len(next_digits) == 0:
#             # the combination is done
#             output.append(combination)
#             return
#         # if there are still digits to check
#          # iterate over all letters which map
#         # the next available digit
#         for letter in phone[next_digits[0]]:
#             # append the current letter to the combination
#             # and proceed to the next digits
#             backtrack(combination + letter, next_digits[1:])
#
#
#     output = []
#     if digits:
#         backtrack("", digits)
#     return output
# print(letterCombinations('234'))
def dfs(candidates, begin, target, res, path):
    if sum(path) == target:
        res.append(path.copy())
        return
    elif sum(path) > target:
        return
    for index in range(begin, len(candidates)):
        if index > begin and candidates[index] == candidates[index - 1]: continue
        path.append(candidates[index])
        dfs(candidates, index + 1, target, res, path)
        path.pop()


# def add_solution():
#     solution = []
#         for _, col in sorted(queens):
#             solution.append('.' * col + 'Q' + '.' * (n - col - 1))
#         output.append(solution)


def solveNQueens(n: int):
    output = []

    def add_solution(res):
        solution = []
        for _, col in sorted(res):
            solution.append('.' * col + 'Q' + '.' * (n - col - 1))
        output.append(solution)

    def dfs(nums, row, path):

        for col in nums:
            if visit[col]: continue
            if hill_diagonals[row - col] or dale_diagobals[row + col]: continue
            visit[col] = True
            hill_diagonals[row - col] = True
            dale_diagobals[row + col] = True
            path.append((row, col))
            if row + 1 == n:
                add_solution(path)
            else:
                dfs(nums, row + 1, path)
            visit[col] = False
            hill_diagonals[row - col] = False
            dale_diagobals[row + col] = False
            path.pop()

    hill_diagonals = [False] * (2 * n - 1)  # 主对角线
    dale_diagobals = [False] * (2 * n - 1)  # 副对角线
    visit = [False] * n
    nums = [i for i in range(n)]
    dfs(nums, 0, [])
    print(output)


def getPermutation(n: int, k: int):
    def jiecheng(level):
        sum = 1
        for i in range(1, level + 1):
            sum *= i
        return sum

    def dfs(nums, index, path, k, visit):
        current_result = jiecheng(n - index)
        for i in nums:
            if visit[i - 1]: continue
            if k > current_result:
                k = k - current_result
                continue
            path.append(str(i))
            visit[i - 1] = True
            if len(path) == n:
                if k == 1: break
            else:
                dfs(nums, index + 1, path, k, visit)

    path = []
    nums = [i for i in range(1, n + 1)]
    visit = [False] * len(nums)
    dfs(nums, 1, path, k, visit)
    print(path)


def subsetsWithDup(nums) :
    res = []
    def dfs(index, path, visited):
        res.append(path.copy())
        for i in range(index, len(nums)):
            if i>0 and nums[i] == nums[i - 1] and visited[i - 1] == False: continue
            visited[i] = True
            dfs(i+1, path+[nums[i]], visited)
            visited[i] = False


    # for k in range(len(nums)+1):
    visited = [False] * len(nums)
    dfs(0, [], visited)
    print(res)


def restoreIpAddresses(s: str):
    res = []

    def dfs(count, ip, s=''):
        if count == 4:
            print(ip)
            if s == '':
                res.append(ip)
            return
        if len(s) > 0:
            dfs(count + 1, ip + s[0] + '.', s[1:])
        if len(s) > 1 and s[0] != '0':
            dfs(count + 1, ip + s[:2] + '.', s[2:])
        if len(s) > 2 and s[0] != '0' and int(s[0:3]) < 256:
            dfs(count + 1, ip + s[:3] + '.', s[3:])

    dfs(0, '', s)
    return res


def validPalindrome(s: str) -> bool:
    def ispalindrom(s1, time):
        print(s1)
        i, j = 0, len(s1) - 1
        while i < j:
            if s1[i] != s1[j]:
                if time == 0:
                    return False
                return ispalindrom(s1[i + 1:j + 1], time - 1) or ispalindrom(s1[i:j], time - 1)
            i = i + 1
            j = j - 1
        return True

    return ispalindrom(s, 1)
from typing import List
import random
from collections import Counter
import collections
class Solution:
    def findKthLargest(self, nums: List[int], k: int) -> int:
        size = len(nums)

        target = size - k
        left = 0
        right = size - 1
        while True:
            index = self.__partition(nums, left, right)
            if index == target:
                return nums#nums[index]
            elif index < target:
                # 下一轮在 [index + 1, right] 里找
                left = index + 1
            else:
                right = index - 1

    #  循环不变量：[left + 1, j] < pivot
    #  (j, i) >= pivot
    def __partition(self, nums, left, right):
        # 随机化切分元素
        # randint 是包括左右区间的
        random_index = random.randint(left, right)
        nums[random_index], nums[left] = nums[left], nums[random_index]

        pivot = nums[left]
        j = left
        for i in range(left + 1, right + 1):
            if nums[i] < pivot:
                j += 1
                nums[i], nums[j] = nums[j], nums[i]

        nums[left], nums[j] = nums[j], nums[left]
        return j

    def q_sort(self,L, left, right):
        if left < right:
            pivot = self.partition(L, left, right)

            self.q_sort(L, left, pivot - 1)
            self.q_sort(L, pivot + 1, right)
        return L

    def partition(self, nums, left, right):
        index = random.randint(left, right)
        nums[left], nums[index] = nums[index], nums[left]
        mid = nums[left]
        while left < right:
            while left < right and nums[right] > mid:
                right = right - 1
            nums[left] = nums[right]
            while left < right and nums[left] < mid:
                left = left + 1
            nums[right] = nums[left]
        nums[left] = mid
        return left

    def topKFrequent(self, nums:str, k: int) -> List[int]:
        # buck = [0] * len(nums)#桶排序
        res = []
        lookup = Counter(nums)
        for item in lookup.items():
            num,frequence = item
            res.append((frequence,num))
        print(res)
        self.q_sort(res,0,len(res)-1)
        print(res)

    def frequencySort(self, s: str) -> str:
        # 桶排序
        ret = []
        countFrequency = collections.defaultdict(int)
        for i in s:
            countFrequency[i] += 1
        buckets = [[] for _ in range(len(s) + 1)]
        for i in countFrequency:
            print(i,i * countFrequency[i])
            buckets[countFrequency[i]].append(i * countFrequency[i])
        for i in buckets[::-1]:
            if (i):
                ret.extend(i)
        return ''.join(ret)



s=Solution()
c= 'tree'
print(s.frequencySort(c))
# print((2,2)>(1,2))
# lookup=Counter('tree')
# print(lookup.items())