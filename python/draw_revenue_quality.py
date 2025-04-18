import matplotlib.pyplot as plt
import numpy as np

folder_path = 'loop-experiment/output/quality-revenue-relationship/'
id = 3476

def read_data(filepath, column1, column2):
    data1 = []
    data2 = []

    with open(filepath, 'r') as f:
        next(f)
        for line in f:
            parts = [p for p in line.split(' ') if p]
            data1.append(float(parts[column1]))
            data2.append(float(parts[column2]))
    f.close()
    return data1, data2

def draw_scatter_plot(xs, ys, xlabel, ylabel):
    plt.figure(figsize=(10, 6))
    plt.scatter(xs, ys, alpha=0.6)

    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)

    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()

def main():
    filepath = folder_path + str(id) + '.txt'
    revenues, winning_scores = read_data(filepath, 1, 7)
    draw_scatter_plot(revenues, winning_scores, 'revenue', 'winning score')

if __name__ == '__main__':
    main()
