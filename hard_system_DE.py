import matplotlib.pyplot as plt
import numpy as np
import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from PIL import Image, ImageTk
import math

class RungeKuttaSystem:
    def __init__(self, stepSize, initialX, initialU1, initialU2, maxCount, epsilonG):
        self.gamma = None
        self.stepSize = stepSize
        self.initialX = initialX
        self.initialU1 = initialU1
        self.initialU2 = initialU2
        self.maxCount = maxCount
        self.epsilonG = epsilonG
        self.V2 = []
        self.OLP = []
        self.Hi = []
        self.C1 = []
        self.C2 = []

    def calculateNextU(self, x, u1, u2, h):
        drob = (3 + math.sqrt(3)) / 6
        tmp = u1 + h * drob * (-500.005 * u1 + 499.995 * u2) / (1 + 500.005 * h * drob)

        k12 = (499.995 * tmp - 500.005 * u2) / (
                1 - ((499.995 * h * drob) ** 2) / (1 + 500.005 * h * drob) + 500.005 * h * drob)

        k11 = (-500.005 * u1 + 499.995 * (u2 + h * drob * k12)) / (1 + 500.005 * h * drob)

        k22 = (499.995 * (u1 + h * (-math.sqrt(3) / 3 * k11 + drob * (
                -500.005 * (u1 - h * math.sqrt(3) / 3 * k11) + 499.995 * (u2 - h * math.sqrt(3) / 3 * k12)) / (
                                            1 + 500.005 * h * drob))) - 500.005 * (
                       u2 - h * math.sqrt(3) / 3 * k12)) / (
                      1 - (drob * 499.995 * h) ** 2 / (1 + 500.005 * h * drob) + 500.005 * h * drob)

        k21 = (-500.005 * (u1 - h * math.sqrt(3) / 3 * k11) + 499.995 * (
                u2 + h * (-math.sqrt(3) / 3 * k12 + drob * k22))) / (1 + 500.005 * h * drob)

        nextU1 = u1 + h / 2 * (k11 + k21)

        nextU2 = u2 + h / 2 * (k12 + k22)

        return (nextU1, nextU2)

    def fixecStep(self, xMax):
        numSteps = int((xMax - self.initialX) / self.stepSize) + 1
        steps = []
        x, u1, u2 = self.initialX, self.initialU1, self.initialU2
        steps.append([x, u1, u2])
        for i in range(1, numSteps):
            x += self.stepSize
            u1, u2 = self.calculateNextU(x - self.stepSize, u1, u2, self.stepSize)
            steps.append([x, u1, u2])
        return steps

    def variableSteps(self, xMax, maxError):
        currentStepSize = self.stepSize
        steps = []
        C1 = 0
        C2 = 0

        currentStep = [self.initialX, self.initialU1, self.initialU2]
        steps.append(currentStep)
        while True:
            nextX = currentStep[0] + currentStepSize
            if nextX > xMax:
                currentStepSize = xMax - currentStep[0]
                continue
            nextU1, nextU2 = self.calculateNextU(currentStep[0], currentStep[1], currentStep[2], currentStepSize)
            nextU1HalfStep, nextU2HalfStep = self.calculateNextU(currentStep[0], currentStep[1], currentStep[2], currentStepSize / 2)
            nextU1HalfStep, nextU2HalfStep = self.calculateNextU(currentStep[0] + currentStepSize / 2, nextU1HalfStep, nextU2HalfStep, currentStepSize / 2)
            errorEstimate1 = abs(nextU1HalfStep - nextU1) / 15
            errorEstimate2 = abs(nextU2HalfStep - nextU2) / 15
            errorEstimate = max(errorEstimate1,errorEstimate2)
            if (errorEstimate <= maxError) and (errorEstimate >= (maxError / 32)):
                nextStep = [nextX, nextU1, nextU2]
                steps.append(nextStep)
                currentStep = nextStep
                self.V2.append([nextU1HalfStep,nextU2HalfStep])
                self.OLP.append(errorEstimate * 16)
                self.Hi.append(currentStepSize)
                self.C1.append(C1)
                self.C2.append(C2)

                if nextX <=xMax and nextX >= xMax-self.epsilonG:
                    break

            elif errorEstimate < (maxError / 32):
                C2 += 1
                nextStep = [nextX, nextU1, nextU2]
                steps.append(nextStep)
                currentStep = nextStep
                self.V2.append([nextU1HalfStep, nextU2HalfStep])
                self.OLP.append(errorEstimate * 16)
                self.Hi.append(currentStepSize)
                self.C1.append(C1)
                self.C2.append(C2)
                currentStepSize *= 2

                if nextX <=xMax and nextX >= xMax-self.epsilonG:
                    break

            else:
                currentStepSize /= 2
                C1 += 1

            if len(steps) != self.maxCount + 1:
                continue
            break

        return steps

    def goToZero(self, zeroController, maxError):
        currentStepSize = self.stepSize
        steps = []
        C1 = 0
        C2 = 0

        currentStep = [self.initialX, self.initialU1, self.initialU2]
        steps.append(currentStep)
        while currentStep[1] > zeroController or currentStep[2] > zeroController:
            nextX = currentStep[0] + currentStepSize

            nextU1, nextU2 = self.calculateNextU(currentStep[0], currentStep[1], currentStep[2], currentStepSize)
            nextU1HalfStep, nextU2HalfStep = self.calculateNextU(currentStep[0], currentStep[1], currentStep[2],
                                                                 currentStepSize / 2)
            nextU1HalfStep, nextU2HalfStep = self.calculateNextU(currentStep[0] + currentStepSize / 2,
                                                                 nextU1HalfStep, nextU2HalfStep,
                                                                 currentStepSize / 2)
            errorEstimate1 = abs(nextU1HalfStep - nextU1) / 15
            errorEstimate2 = abs(nextU2HalfStep - nextU2) / 15
            errorEstimate = max(errorEstimate1, errorEstimate2)
            if (errorEstimate <= maxError) and (errorEstimate >= (maxError / 32)):
                nextStep = [nextX, nextU1, nextU2]
                steps.append(nextStep)
                currentStep = nextStep
                self.V2.append([nextU1HalfStep, nextU2HalfStep])
                self.OLP.append(errorEstimate * 16)
                self.Hi.append(currentStepSize)
                self.C1.append(C1)
                self.C2.append(C2)

            elif errorEstimate < (maxError / 32):
                C2 += 1
                nextStep = [nextX, nextU1, nextU2]
                steps.append(nextStep)
                currentStep = nextStep
                self.V2.append([nextU1HalfStep, nextU2HalfStep])
                self.OLP.append(errorEstimate * 16)
                self.Hi.append(currentStepSize)
                self.C1.append(C1)
                self.C2.append(C2)
                currentStepSize *= 2

            else:
                currentStepSize /= 2
                C1 += 1

            if len(steps) != self.maxCount + 1:
                continue
            break

        return steps



def update_plot_zero():
    epsilonG = float(epsilonG_entry.get())
    maxCount = float(maxCount_entry.get())
    maxError = float(maxError_entry.get())
    h0 = float(h0_entry.get())
    xMax = float(xMax_entry.get())
    x_0 = float(x0_entry.get())
    U1_0 = float(u1_0_entry.get())
    U2_0 = float(u2_0_entry.get())
    zero = float(zeroController_entry.get())
    step_type = step_type_var.get()

    func1 = RungeKuttaSystem(h0, x_0, U1_0, U2_0, maxCount, epsilonG)
    data = np.array([func1.goToZero(zero, maxError)])

    x, u1, u2 = data.T
    trueU1, trueU2 = trueSolve(x)

    V2 = func1.V2
    OLP = func1.OLP
    Hi = func1.Hi
    C1 = func1.C1
    C2 = func1.C2
    differenceU1 = u1 - trueU1
    differenceU2 = u2 - trueU2

    max_diff = np.max(np.abs(differenceU1))
    max_diff_index = np.argmax(np.abs(differenceU1))
    max_diff_x = x[max_diff_index]

    tree.delete(*tree.get_children())

    Data = []
    Data.append((0, x[0][0], (u1[0][0], u2[0][0])))
    for i in range(1, len(x)):
        Data.append((i, x[i][0], (u1[i][0], u2[i][0]), V2[i - 1] if i - 1 < len(V2) else "",
                     (V2[i - 1][0] - u1[i][0], V2[i - 1][1] - u2[i][0]) if i - 1 < len(V2) else "",
                     OLP[i - 1] if i - 1 < len(OLP) else "" if i > 1 and i - 1 < len(OLP) else "",
                     Hi[i - 1] if i - 1 < len(Hi) else "",
                     C1[i - 1] if i - 1 < len(C1) else "",
                     C2[i - 1] if i - 1 < len(C2) else "", differenceU1[i][0], differenceU2[i][0]))

    for inf in Data:
        tree.insert("", "end", values=inf)

    number_of_iterations = len(x) - 1
    difference = xMax - x[number_of_iterations]
    maxOLP = max(OLP) if len(OLP) != 0 else 0
    C_1 = max(C1) if len(C1) != 0 else 0
    C_2 = max(C2) if len(C2) != 0 else 0
    max_h = max(Hi) if len(Hi) != 0 else h0
    min_h = min(Hi) if len(Hi) != 0 else h0
    max_h_index = Hi.index(max_h) + 1 if len(Hi) != 0 else 0
    min_h_index = Hi.index(min_h) + 1 if len(Hi) != 0 else 0
    max_h_x = x[max_h_index] if len(Hi) != 0 else 0
    min_h_x = x[min_h_index] if len(Hi) != 0 else 0

    results_window = tk.Toplevel(root)
    results_window.title("Результаты")

    tk.Label(results_window, text="Количество итераций:").grid(row=0, column=0)
    tk.Label(results_window, text=number_of_iterations).grid(row=0, column=1)

    tk.Label(results_window, text="Разница между правой границей и последним вычисленным значением:").grid(row=1,
                                                                                                           column=0)
    tk.Label(results_window, text=difference[0]).grid(row=1, column=1)

    tk.Label(results_window, text="Максимальное значение OLP:").grid(row=2, column=0)
    tk.Label(results_window, text=maxOLP).grid(row=2, column=1)

    tk.Label(results_window, text="Количество делений:").grid(row=3, column=0)
    tk.Label(results_window, text=C_1).grid(row=3, column=1)

    tk.Label(results_window, text="Количество удвоений:").grid(row=4, column=0)
    tk.Label(results_window, text=C_2).grid(row=4, column=1)

    tk.Label(results_window, text="Максимальное значение Hi:").grid(row=5, column=0)
    tk.Label(results_window, text=max_h).grid(row=5, column=1)

    tk.Label(results_window, text="Минимальное значение Hi:").grid(row=6, column=0)
    tk.Label(results_window, text=min_h).grid(row=6, column=1)

    tk.Label(results_window, text="Значение x для максимального Hi:").grid(row=9, column=0)
    tk.Label(results_window, text=max_h_x[0]).grid(row=9, column=1)

    tk.Label(results_window, text="Значение x для минимального Hi:").grid(row=10, column=0)
    tk.Label(results_window, text=min_h_x[0]).grid(row=10, column=1)

    tk.Label(results_window, text="Максимальное отклонение:").grid(row=11, column=0)
    tk.Label(results_window, text=max_diff).grid(row=11, column=1)

    tk.Label(results_window, text="x для максимального отклонения:").grid(row=12, column=0)
    tk.Label(results_window, text=max_diff_x[0]).grid(row=12, column=1)

    plt.cla()

    ax.clear()

    ax.plot(x, u1, label='U1(x)')
    ax.plot(x, u2, label='U2(x)')
    ax.plot(x, trueU1, label='Истинное U1(x)', linestyle='dashed')
    ax.plot(x, trueU2, label='Истинное U2(x)', linestyle='dashed')
    ax.set_xlabel("x")
    ax.set_ylabel("u")
    ax.grid(True)
    ax.legend()

    canvas.draw()
    show_difference_plot(x, differenceU1, differenceU2)

def update_plot():
    epsilonG = float(epsilonG_entry.get())
    maxCount = float(maxCount_entry.get())
    maxError = float(maxError_entry.get())
    h0 = float(h0_entry.get())
    xMax = float(xMax_entry.get())
    x_0 = float(x0_entry.get())
    U1_0 = float(u1_0_entry.get())
    U2_0 = float(u2_0_entry.get())
    step_type = step_type_var.get()

    func1 = RungeKuttaSystem(h0, x_0, U1_0, U2_0, maxCount, epsilonG )
    data = np.array([func1.variableSteps(xMax, maxError)]) if step_type == "Переменный" else np.array(
        [func1.fixecStep(xMax)])
    x, u1, u2 = data.T
    trueU1, trueU2 = trueSolve(x)

    V2 = func1.V2
    OLP = func1.OLP
    Hi = func1.Hi
    C1 = func1.C1
    C2 = func1.C2
    differenceU1 = u1 - trueU1
    differenceU2 = u2 - trueU2

    max_diff = np.max(np.abs(differenceU1))
    max_diff_index = np.argmax(np.abs(differenceU1))
    max_diff_x = x[max_diff_index]

    tree.delete(*tree.get_children())

    Data = []
    Data.append((0, x[0][0], (u1[0][0], u2[0][0])))
    for i in range(1, len(x)):
        Data.append((i, x[i][0], (u1[i][0], u2[i][0]), V2[i - 1] if i - 1 < len(V2) else "",
                     (V2[i - 1][0] - u1[i][0], V2[i - 1][1] - u2[i][0]) if i - 1 < len(V2) else "",
                     OLP[i - 1] if i - 1 < len(OLP) else "" if i>1 and i-1 < len(OLP) else "", Hi[i - 1] if i - 1 < len(Hi) else "",
                     C1[i - 1] if i - 1 < len(C1) else "",
                     C2[i - 1] if i - 1 < len(C2) else "",differenceU1[i][0], differenceU2[i][0]))

    for inf in Data:
        tree.insert("", "end", values=inf)

    number_of_iterations = len(x) - 1
    difference = xMax - x[number_of_iterations]
    maxOLP = max(OLP) if len(OLP) != 0 else 0
    C_1 = max(C1) if len(C1) != 0 else 0
    C_2 = max(C2) if len(C2) != 0 else 0
    max_h = max(Hi) if len(Hi) != 0 else h0
    min_h = min(Hi) if len(Hi) != 0 else h0
    max_h_index = Hi.index(max_h) + 1 if len(Hi) != 0 else 0
    min_h_index = Hi.index(min_h) + 1 if len(Hi) != 0 else 0
    max_h_x = x[max_h_index] if len(Hi) != 0 else 0
    min_h_x = x[min_h_index] if len(Hi) != 0 else 0

    results_window = tk.Toplevel(root)
    results_window.title("Результаты")

    tk.Label(results_window, text="Количество итераций:").grid(row=0, column=0)
    tk.Label(results_window, text=number_of_iterations).grid(row=0, column=1)

    tk.Label(results_window, text="Разница между правой границей и последним вычисленным значением:").grid(row=1,
                                                                                                           column=0)
    tk.Label(results_window, text=difference[0]).grid(row=1, column=1)

    tk.Label(results_window, text="Максимальное значение OLP:").grid(row=2, column=0)
    tk.Label(results_window, text=maxOLP).grid(row=2, column=1)

    tk.Label(results_window, text="Количество делений:").grid(row=3, column=0)
    tk.Label(results_window, text=C_1).grid(row=3, column=1)

    tk.Label(results_window, text="Количество удвоений:").grid(row=4, column=0)
    tk.Label(results_window, text=C_2).grid(row=4, column=1)

    tk.Label(results_window, text="Максимальное значение Hi:").grid(row=5, column=0)
    tk.Label(results_window, text=max_h).grid(row=5, column=1)

    tk.Label(results_window, text="Минимальное значение Hi:").grid(row=6, column=0)
    tk.Label(results_window, text=min_h).grid(row=6, column=1)

    tk.Label(results_window, text="Значение x для максимального Hi:").grid(row=9, column=0)
    tk.Label(results_window, text=max_h_x[0]).grid(row=9, column=1)

    tk.Label(results_window, text="Значение x для минимального Hi:").grid(row=10, column=0)
    tk.Label(results_window, text=min_h_x[0]).grid(row=10, column=1)

    tk.Label(results_window, text="Максимальное отклонение:").grid(row=11, column=0)
    tk.Label(results_window, text=max_diff).grid(row=11, column=1)

    tk.Label(results_window, text="x для максимального отклонения:").grid(row=12, column=0)
    tk.Label(results_window, text=max_diff_x[0]).grid(row=12, column=1)

    plt.cla()

    ax.clear()

    ax.plot(x, u1, label='U1(x)')
    ax.plot(x, u2, label='U2(x)')
    ax.plot(x, trueU1, label='Истинное U1(x)', linestyle='dashed')
    ax.plot(x, trueU2, label='Истинное U2(x)', linestyle='dashed')
    ax.set_xlabel("x")
    ax.set_ylabel("u")
    ax.grid(True)
    ax.legend()

    canvas.draw()
    show_difference_plot(x, differenceU1, differenceU2)

def trueSolve(x):
    return (-3 * np.exp(-1000 * x) + 10 * np.exp(-0.01 * x),
            3 * np.exp(-1000 * x) + 10 * np.exp(-0.01 * x))

def show_difference_plot(x, differenceU1, differenceU2):
    # Создание нового окна
    diff_window = tk.Toplevel(root)
    diff_window.title("Глобальная погрешность")

    # Создание фигуры для графика
    diff_fig, diff_ax = plt.subplots()
    diff_canvas = FigureCanvasTkAgg(diff_fig, master=diff_window)
    diff_canvas_widget = diff_canvas.get_tk_widget()
    diff_canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    # Отображение графика разности
    diff_ax.plot(x, differenceU1, label='Разница U1')
    diff_ax.plot(x, differenceU2, label='Разница U2')
    diff_ax.set_xlabel("x")
    diff_ax.set_ylabel("Разница")
    diff_ax.grid(True)
    diff_ax.legend()

    diff_canvas.draw()



root = tk.Tk()
root.title("Маилян Эрик, 3822Б1ПМмм1, Жесткая система, метод 16.63")

frame = ttk.Frame(root)
frame.pack(padx=10, pady=10)

u1_0_label = ttk.Label(frame, text="U_1:")
u1_0_label.grid(row=1, column=3)
u1_0_entry = ttk.Entry(frame)
u1_0_entry.grid(row=1, column=4)
u1_0_entry.insert(0,"7")

u2_0_label = ttk.Label(frame, text="U_2:")
u2_0_label.grid(row=2, column=3)
u2_0_entry = ttk.Entry(frame)
u2_0_entry.grid(row=2, column=4)
u2_0_entry.insert(0,"13")

maxCount_label = ttk.Label(frame, text="Максимальное количество итераций:")
maxCount_label.grid(row=1, column=0)
maxCount_entry = ttk.Entry(frame)
maxCount_entry.grid(row=1, column=1)
maxCount_entry.insert(0, "100000")

maxError_label = ttk.Label(frame, text="Параметр контроля ошибки:")
maxError_label.grid(row=2, column=0)
maxError_entry = ttk.Entry(frame)
maxError_entry.grid(row=2, column=1)
maxError_entry.insert(0, "0.000000001")

h0_label = ttk.Label(frame, text="Начальный шаг:")
h0_label.grid(row=3, column=0)
h0_entry = ttk.Entry(frame)
h0_entry.grid(row=3, column=1)
h0_entry.insert(0, "0.001")

xMax_label = ttk.Label(frame, text="Правая граница:")
xMax_label.grid(row=4, column=0)
xMax_entry = ttk.Entry(frame)
xMax_entry.grid(row=4, column=1)
xMax_entry.insert(0, "0.01")

x0_label = ttk.Label(frame, text="x0:")
x0_label.grid(row=5, column=0)
x0_entry = ttk.Entry(frame)
x0_entry.grid(row=5, column=1)
x0_entry.insert(0, "0")

epsilonG_label = ttk.Label(frame, text="Точность выхода за границу:")
epsilonG_label.grid(row=6, column=0)
epsilonG_entry = ttk.Entry(frame)
epsilonG_entry.grid(row=6, column=1)
epsilonG_entry.insert(0, "0.0001")

zeroController_label = ttk.Label(frame, text="Точность спуска до 0")
zeroController_label.grid(row=7, column=0)
zeroController_entry = ttk.Entry(frame)
zeroController_entry.grid(row=7, column=1)
zeroController_entry.insert(0, "0.0001")

zero_button = ttk.Button(frame, text="Посчитать спуск до 0", command=update_plot_zero)
zero_button.grid(row=7, column=2, columnspan=2)


step_type_label = ttk.Label(frame, text="Выберите шаг:")
step_type_label.grid(row=8, column=0)
step_type_var = tk.StringVar()
step_type_var.set("Переменный")
step_type_option = ttk.OptionMenu(frame, step_type_var, "Переменный","Фиксированный", "Переменный")
step_type_option.grid(row=8, column=1)


update_button = ttk.Button(frame, text="Обновить", command=update_plot)
update_button.grid(row=9, columnspan=2)

columns = ("i", "xi", "vi", "v2i", "vi-v2i", "Бесконечная норма ОЛП", "hi", "Количество делений", "Количество умножений", "Разница между иситнным и численным")
tree = ttk.Treeview(columns=columns, show="headings")

vsb = ttk.Scrollbar(root, orient="vertical", command=tree.yview)
vsb.pack(side="right", fill="y")
tree.configure(yscrollcommand=vsb.set)

tree.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=10)

for col in columns:
    tree.heading(col, text=col)

fig, ax = plt.subplots()
canvas = FigureCanvasTkAgg(fig, master=root)
canvas_widget = canvas.get_tk_widget()
canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=True)


update_plot()

root.mainloop()