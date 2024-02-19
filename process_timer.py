file = open('text.txt')
total = 0
num_lines = 0
while True:
    content = file.readline()
    if not content:
        break
    num_lines += 1
    content = content.split(',')
    content = content[-1]
    content = content.split('m')
    content = float(content[0])
    total += content

print(total)
print(num_lines)
print(total/num_lines)