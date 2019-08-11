import random
# 创建三行四列列表
matrix = []
for row in range(3):
    matrix.append([])
    for column in range(4):
        matrix[row].append(column)
print(matrix)

matrix1 = []
matrix1.append(3 *[1])
print(3 * [1])
print(matrix1)
matrix1.append(3 *[1])
print(matrix1)
matrix1.append(3 *[1])
print(matrix1)
matrix1[0][0] = 2
print(matrix1)


