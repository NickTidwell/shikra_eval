import matplotlib.pyplot as plt

data = {
    '1': 0.33643977217670995,
    '2': 0.2884404155293563,
    '3': 0.24499710785670084,
    '4': 0.27332765846534124,
    '5': 0.21520631619034958,
    '6-10': 0.22239107116240833,
    '11-20': 0.19447533953963755,
    '20+': 0.16501156288996235
}

# Create a table
print("Number Range\tPercentage")
print("---------------------------")
for key, value in data.items():
    print(f"{key}\t\t{value}")

# Create a plot
keys = list(data.keys())
values = list(data.values())

plt.figure(figsize=(10, 6))

# Draw a line through the bars
plt.plot(keys, values, marker='o', color='red', linestyle='-')

# Assign each bar a different color
colors = ['skyblue', 'orange', 'green', 'purple', 'pink', 'yellow', 'brown', 'gray']
for i in range(len(keys)):
    plt.bar(keys[i], values[i], color=colors[i])

plt.xlabel('Number of Images')
plt.ylabel('Percentage')
plt.title('Percentage Distribution by Number of Images')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


import matplotlib.pyplot as plt

data = {'common': 0.25286440731835632, 'rare': 0.21184566624273992}
# Extract keys and values
keys = list(data.keys())
values = list(data.values())
plt.figure(figsize=(6, 4))
# Plotting the bar graph
plt.bar(keys, values, color=['blue', 'green'])
# Adding labels and title
plt.xlabel('Categories')
plt.ylabel('Percentage')
plt.title('Percentage Distribution by Category')
plt.show()