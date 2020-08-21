
# s = [1,2,3]
# sum = pow(2,len(s))
# res = []
# nth_bit = 1 << len(s)
# for i in range(1,sum):
#     # bin_len = len(bin(i)[2:])
#     # c = len(s) - bin_len#如1--》001 实际输出为1,前面少了2位0
#     sub = []
#     # for b,v in zip(bin(i)[2:],s[c:]):
#     #     # print(bin(i)[2:],b,v)
#     #     if int(b) == 1:
#     #         sub.append(v)
#     # res.append(sub)
#
#     bitmask = bin(i | nth_bit)[3:]
#     for i in range(len(s)):
#         if int(bitmask[i]):
#             sub.append(s[i])
#     res.append(sub)
# print(res)
# inorderList =[2, 3, 3, 3, 4, 4, 5, 5, 6]
# record = []
# count = 1
# max_count = 1
# for i in range(1,len(inorderList)):
#     print(i,record,count,max_count)
#     if inorderList[i] != inorderList[i-1]:
#         if count==max_count:
#             max_count = count
#             record.append(inorderList[i-1])
#         elif count>max_count:
#             max_count = count
#             record.clear()
#             record.append(inorderList[i-1])
#         count=1
#     else:count += 1
# print(record)
# import heapq
# matrix = [
#    [ 1,  5,  9],
#    [10, 11, 13],
#    [12, 13, 15]
# ]
# n = len(matrix) #注：题目中这个矩阵是n*n的，所以长宽都是n
# pq = [(matrix[i][0], i, 0) for i in range(n)] #取出第一列候选人#matrix[i][0]是具体的值，后面的(i,0)是在记录候选人在矩阵中的位置，方便每次右移添加下一个候选人
# heapq.heapify(pq) #变成一个heap
# for i in range(5 - 1):#一共弹k次：这里k-1次，return的时候1次
#     num, x, y = heapq.heappop(pq) #弹出候选人里最小一个
#     # print(num, x, y,matrix[x][y + 1], x, y + 1)
#     if y != n - 1: #如果这一行还没被弹完
#         heapq.heappush(pq, (matrix[x][y + 1], x, y + 1)) #加入这一行的下一个候选人
# print(heapq.heappop(pq)[0])

print(-1&0xFFFFFFFF)
