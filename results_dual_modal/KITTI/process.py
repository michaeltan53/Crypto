with open('geometry.txt', 'r') as f:
    numbers = f.readlines()

numbers = [str(-float(num.strip())) + '\n' for num in numbers]

with open('geometry.txt', 'w') as f:
    f.writelines(numbers)

print("处理完成，结果保存在 TCIA.txt")
