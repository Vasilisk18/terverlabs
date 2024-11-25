import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import random

def run_experiment():
    N = int(entry_N.get())
    M = int(entry_M.get())
    r = int(entry_r.get())
    n = int(entry_kol.get())

    Ns = N
    Ms = M
    rs = r

    mas_y = []
    mas_ni = []
    mas_n = []

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

        index = mas_y.index(sv)
        mas_ni[index] += 1

    for i in range(len(mas_ni)):
        mas_n[i] = mas_ni[i] / n

    update_table(mas_y, mas_ni, mas_n)

def update_table(mas_y, mas_ni, mas_n):
    for i in tree.get_children():
        tree.delete(i)

    for i in range(len(mas_y)):
        tree.insert('', 'end', values=(mas_y[i], mas_ni[i], mas_n[i]))

def show_image():
    # Load the image
    image_path = "text.png"  # исправлено
    image = Image.open(image_path)

    # Create a new window for the image
    image_window = tk.Toplevel(root)
    image_window.title("Image Viewer")  # исправлено

    # Convert the image to a format that tkinter can display
    tk_image = ImageTk.PhotoImage(image)

    # Display the image in a label
    label = tk.Label(image_window, image=tk_image)
    label.image = tk_image  # Keep a reference to the image to prevent garbage collection
    label.pack()

# Create main window
root = tk.Tk()
root.title("Experiment Results")  # исправлено

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
tk.Label(root, text='Общее число лампочек').grid(row=0, column=0)  # исправлено
tk.Label(root, text='Число перегоревших лампочек').grid(row=1, column=0)  # исправлено
tk.Label(root, text='Число выбранных лампочек').grid(row=2, column=0)  # исправлено
tk.Label(root, text='Число экспериментов').grid(row=3, column=0)  # исправлено

# Run button
btn_run = tk.Button(root, text='Начать', command=run_experiment)  # исправлено
btn_run.grid(row=4, column=0, columnspan=2)
btn_run.config(font=('Arial', 12))  # Increase font size

# Table
tree = ttk.Treeview(root, columns=('Value', 'Occurrences', 'Frequency'), height=20)
tree.heading('#0', text='', anchor='center')  # Hide the header
tree.column('#0', width=0)  # Hide the column
tree.heading('#1', text='Значения с.в.')
tree.heading('#2', text='Число выпадений')
tree.heading('#3', text='Частота')
tree.grid(row=5, column=0, columnspan=2)

# Button for showing image
btn_show_image = tk.Button(root, text='Вариант №15', command=show_image)  # исправлено
btn_show_image.grid(row=0, column=2, columnspan=2)
btn_show_image.config(font=('Arial', 12))  # Increase font size

root.mainloop()
