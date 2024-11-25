import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import random
import math
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Global variable to store Ns
Ns_value = None
plot_canvas = None  # Global variable to store the canvas for the plot


def clean_table(mas_y, mas_ni, mas_n, Ns, Ms, rs):
    i = 0
    while i < len(mas_ni):
        if mas_y[i] < Ms - (Ns - rs) or mas_y[i] > rs:
            del mas_y[i]
            del mas_ni[i]
            del mas_n[i]
        else:
            i += 1


def update_table(mas_y, mas_ni, mas_n):
    for i in tree.get_children():
        tree.delete(i)

    for i in range(len(mas_y)):
        tree.insert('', 'end', values=(mas_y[i], mas_ni[i], mas_n[i]))


def update_second_table(mas_el):
    for i in tree2.get_children():
        tree2.delete(i)

    tree2.insert('', 'end', values=(
    mas_el[0], mas_el[1], round(abs(mas_el[0] - mas_el[1]), 12), mas_el[2], round(mas_el[3], 12),
    round(abs(mas_el[2] - mas_el[3]), 12), mas_el[4], mas_el[5]))


def run_experiment():
    global Ns_value

    N = int(entry_N.get())
    M = int(entry_M.get())
    r = int(entry_r.get())
    n = int(entry_kol.get())

    Ns_value = N
    Ns = N
    Ms = M
    rs = r

    mas_y = []
    mas_ni = []
    mas_n = []
    mas_x = []

    for i in range(M + 1):
        mas_y.append(i)
        mas_ni.append(0)
        mas_n.append(0)

    for i in range(n):
        sv = 0
        M = Ms
        N = Ns
        r = rs
        while r > 0:
            P = M / N
            number = random.random()
            if number < P:
                M -= 1
                sv += 1
            N -= 1
            r -= 1

        mas_x.append(sv)
        index = mas_y.index(sv)
        mas_ni[index] += 1

    mas_x.sort()

    mas_x_only = list(set(mas_x))

    for i in mas_x:
        if i not in mas_x:
            mas_x_only.append(i)

    for i in range(len(mas_ni)):
        mas_n[i] = mas_ni[i] / n

    clean_table(mas_y, mas_ni, mas_n, Ns, Ms, rs)

    mas_p = []

    for i in range(0, len(mas_y)):
        ni = mas_y[i]
        mas_p.append(math.comb(Ms, ni) * math.comb(Ns - Ms, abs(rs - ni)) / math.comb(Ns, rs))

    update_table(mas_y, mas_p, mas_n)

    mas_el = []

    Eη = 0

    for i in range(0, len(mas_y)):
        Eη += mas_y[i] * mas_p[i]

    Eη = rs * Ms / Ns

    mas_el.append(Eη)

    x = sum(mas_x) / len(mas_x)

    mas_el.append(x)

    D = 0
    for i in range(0, len(mas_y)):
        D += mas_y[i] * mas_y[i] * mas_p[i]

    D = Ms * (Ns - Ms) * (Ns - rs) * rs / (Ns * Ns * (Ns - 1))

    mas_el.append(D)

    S2 = 0
    for i in range(0, len(mas_x)):
        S2 += (mas_x[i] - x) ** 2
    S2 /= len(mas_x)
    mas_el.append(S2)

    Me = 0
    if len(mas_x) % 2 == 0:
        Me = (mas_x[len(mas_x) // 2 - 1] + mas_x[len(mas_x) // 2]) / 2
    else:
        Me = mas_x[len(mas_x) // 2]
    mas_el.append(Me)

    mas_el.append(mas_x[len(mas_x) - 1] - mas_x[0])

    update_second_table(mas_el)

    max_otkl = 0
    for i in range(0, len(mas_y)):
        if abs(mas_n[i] - mas_p[i]) > max_otkl:
            max_otkl = abs(mas_n[i] - mas_p[i])

    tk.Label(root, text=f"Максимально отклонение: {max_otkl}").grid(row=6, column=0)
    # tk.Label(root, text=f"массив занчений: {mas_x_only}").grid(row=7, column=0)
    plot_data(mas_y, mas_p, mas_x_only, mas_n)


def show_image():
    image_path = "text.png"
    image = Image.open(image_path)

    image_window = tk.Toplevel(root)
    image_window.title("Image Viewer")

    tk_image = ImageTk.PhotoImage(image)

    label = tk.Label(image_window, image=tk_image)
    label.image = tk_image
    label.pack()


root = tk.Tk()
root.title("Experiment Results")

# Input fields
entry_N = tk.Entry(root)
entry_N.grid(row=0, column=1)
entry_N.config(font=('Arial', 12))  # Increase font size

entry_M = tk.Entry(root)
entry_M.grid(row=1, column=1)
entry_M.config(font=('Arial', 12))  # Increase font size

entry_r = tk.Entry(root)
entry_r.grid(row=2, column=1)
entry_r.config(font=('Arial', 12))  # Increase font size

entry_kol = tk.Entry(root)
entry_kol.grid(row=3, column=1)
entry_kol.config(font=('Arial', 12))  # Increase font size

# Labels
tk.Label(root, text="Общее число лампочек:").grid(row=0, column=0)
tk.Label(root, text="Число перегоревших лампочек:").grid(row=1, column=0)
tk.Label(root, text="Число выбранных лампочек:").grid(row=2, column=0)
tk.Label(root, text="Число эксперементов:").grid(row=3, column=0)

# Run button
btn_run = tk.Button(root, text="Начать", command=run_experiment)
btn_run.grid(row=4, column=0, columnspan=2)
btn_run.config(font=('Arial', 12))  # Increase font size

# Table
tree = ttk.Treeview(root, columns=('Value', 'Occurrences', 'Frequency'), height=10)
tree.heading('#0', text='', anchor='center')  # Hide the header
tree.column('#0', width=0)  # Hide the column
tree.heading('#1', text='Значения с.в.')
tree.heading('#2', text='Вероятность')
tree.heading('#3', text='Частота')
tree.grid(row=5, column=0, columnspan=2)

# Second Table
tree2 = ttk.Treeview(root, columns=('Eη', 'x', '|Eη − x|', 'Dη', 'S^2', '|Dη − S^2|', 'Me', 'R'), height=3)
tree2.heading('#0', text='', anchor='center')  # Hide the header
tree2.column('#0', width=0)  # Hide the column
tree2.heading('#1', text='Eη')
tree2.heading('#2', text='x')
tree2.heading('#3', text='|Eη − x|')
tree2.heading('#4', text='Dη')
tree2.heading('#5', text='S2')
tree2.heading('#6', text='|Dη − S2|')
tree2.heading('#7', text='Me')
tree2.heading('#8', text='R')
tree2.grid(row=8, column=0, columnspan=2)

# Make each column twice as narrow
for col in tree2['columns']:
    tree2.column(col, width=tree2.column(col, width=None) // 2)

# Space for Plots
plot_frame = tk.Frame(root)
plot_frame.grid(row=0, column=3, rowspan=6, padx=10, pady=10)


def repeat_and_adjust(arr):
    result = []
    result.append(arr[0] - 1)
    for num in arr:
        result.extend([num, num])
    result.append(arr[-1] + 1)
    return result


def sum_and_repeat(arr):
    result = [0, 0]
    for num in arr:
        total = num + result[-1]
        result.extend([total, total])
    return result


def transform_array(arr):
    length = len(arr)
    result = []
    fractions = [i / length for i in range(1, length + 1)]
    for fraction in fractions:
        result.extend([fraction, fraction])
    result = [0, 0] + result
    return result


# Function to plot data
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


def plot_data(mas_y, mas_p, mas_x, mas_n):
    global plot_canvas

    # Clear previous plot if exists
    if plot_canvas:
        plot_canvas.get_tk_widget().destroy()

    mas_n = [x for x in mas_n if x != 0]

    for elem in mas_y:
        if elem not in mas_x:
            index = mas_y.index(elem)
            mas_x.insert(index, 0)
            mas_n.insert(index, 0)

    while len(mas_x) < len(mas_y):
        mas_x.append(0)
        mas_n.append(0)

    '''
    tk.Label(root, text=f"x1 - mas y: {mas_y}").grid(row=10, column=0)
    tk.Label(root, text=f"x2 - mas x: {mas_x}").grid(row=11, column=0)
    tk.Label(root, text=f"y1 - mas p: {mas_p}").grid(row=12, column=0)
    tk.Label(root, text=f"y2 - mas n: {mas_n}").grid(row=13, column=0)
    '''

    # Generate example data
    x1 = repeat_and_adjust(mas_y)
    x2 = repeat_and_adjust(mas_x)

    y1 = sum_and_repeat(mas_p)
    y2 = sum_and_repeat(mas_n)

    # Create a figure and plot the data
    fig, ax = plt.subplots()
    ax.plot(x1, y1, label='Теоретическая фун-ия рапсред.')
    ax.plot(x1, y2, label='Выборочная фун-ия рапсред.')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('')

    D = 0
    for i in range(0, len(y2)):
        if abs(y1[i] - y2[i]) > D:
            D = abs(y1[i] - y2[i])
            X = x1[i]

    tk.Label(root, text=f"Мера расхождения: {round(D, 12)}").grid(row=6, column=3)
    tk.Label(root, text=f"При X = {X}").grid(row=7, column=3)

    '''
    tk.Label(root, text=f"Y1: {y1}").grid(row=10, column=0)
    tk.Label(root, text=f"X2: {x1}").grid(row=11, column=0)
    tk.Label(root, text=f"Y2: {y2}").grid(row=12, column=0)
    tk.Label(root, text=f"X2: {x2}").grid(row=13, column=0)
    '''

    # Draw dashed lines for y=0 and y=1 in green color
    ax.axhline(y=0, color='green', linestyle='--')
    ax.axhline(y=1, color='green', linestyle='--')

    # Add legend
    ax.legend()

    # Embed the plot into tkinter window
    canvas = FigureCanvasTkAgg(fig, master=plot_frame)
    canvas.draw()
    canvas.get_tk_widget().pack()

    # Store the canvas reference
    plot_canvas = canvas


# Button for showing image
btn_show_image = tk.Button(root, text="Вариант №15", command=show_image)
btn_show_image.grid(row=0, column=2, columnspan=2)
btn_show_image.config(font=('Arial', 12))  # Increase font size

root.mainloop()