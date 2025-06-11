import math

import matplotlib.pyplot as plt
import numpy as np
import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import warnings

warnings.filterwarnings("ignore")

class RungeKutta:
    def __init__(self, stepSize, initialX, initialY, maxCount, epsilonG, function):
        self.epsilonG = epsilonG
        self.function = function
        self.maxCount = maxCount
        self.stepSize = stepSize
        self.initialX = initialX
        self.initialY = initialY
        self.V2 = []
        self.OLP = []
        self.Hi = []
        self.C1 = []
        self.C2 = []

    def calculateNextY(self, x, y, stepSize):
        k1 = self.function(x, y)
        k2 = self.function(x + stepSize / 2, y + k1 * stepSize / 2)
        k3 = self.function(x + stepSize / 2, y + k2 * stepSize / 2)
        k4 = self.function(x + stepSize, y + k3 * stepSize)
        nextY = y + stepSize * (k1 + 2 * k2 + 2 * k3 + k4) / 6
        if (nextY > y + 1 ):
            nextY = y+self.epsilonG
        elif (nextY < y - 1):
            nextY = y - self.epsilonG
        return nextY

    def fixedStep(self, xMax):
        numSteps = int((xMax - self.initialX) / self.stepSize) + 1
        steps = []
        x, y = self.initialX, self.initialY
        steps.append([x, y])
        for i in range(1, numSteps):
            x = x + self.stepSize
            y = self.calculateNextY(x - self.stepSize, y, self.stepSize)
            steps.append([x, y])

        return steps

    def variableStep(self, xMax, maxError):
        currentStepSize = self.stepSize
        steps = []
        C1 = 0
        C2 = 0

        currentStep = [self.initialX, self.initialY]
        steps.append(currentStep)
        while True:
            nextX = currentStep[0] + currentStepSize
            if nextX > xMax:
                currentStepSize = xMax - currentStep[0]
                continue
            nextY = self.calculateNextY(currentStep[0], currentStep[1], currentStepSize)
            nextYHalfStep = self.calculateNextY(currentStep[0], currentStep[1], currentStepSize / 2)
            nextYHalfStep = self.calculateNextY(currentStep[0] + currentStepSize / 2, nextYHalfStep, currentStepSize / 2)
            errorEstimate = abs((nextYHalfStep - nextY) / 15)
            if (errorEstimate <= maxError) and (errorEstimate >= (maxError / 32)):
                nextStep = [nextX, nextY]
                steps.append(nextStep)
                currentStep = nextStep
                self.V2.append(nextYHalfStep)
                self.OLP.append(errorEstimate * 16)
                self.Hi.append(currentStepSize)
                self.C1.append(C1)
                self.C2.append(C2)

                if nextX <=xMax and nextX >= xMax-self.epsilonG:
                    break

            elif errorEstimate < (maxError / 32):
                C2 += 1
                nextStep = [nextX, nextY]
                steps.append(nextStep)
                currentStep = nextStep
                self.V2.append(nextYHalfStep)
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
            if len(steps) != self.maxCount:
                continue
            break

        return steps

class RungeKuttaSystem:
    def __init__(self, stepSize, initialX, initialU1, initialU2, maxCount, epsilonG, a, b):
        self.stepSize = stepSize
        self.initialX = initialX
        self.initialU1 = initialU1
        self.initialU2 = initialU2
        self.maxCount = maxCount
        self.epsilonG = epsilonG
        self.a = a
        self.b = b
        self.V2 = []
        self.OLP = []
        self.Hi = []
        self.C1 = []
        self.C2 = []

    def du1(self, x, u1, u2):
        return u2

    def du2(self, x, u1, u2):
        return -self.a * u2 - self.b * np.sin(u1)

    def calculateNextU(self, x, u1, u2, stepSize):
        k11 = self.du1(x,u1, u2)
        k12 = self.du2(x,u1, u2)
        k21 = self.du1(x + stepSize / 2, u1 + stepSize * k11 / 2, u2 + stepSize * k12 / 2)
        k22 = self.du2(x + stepSize / 2, u1 + stepSize * k11 / 2, u2 + stepSize * k12 / 2)
        k31 = self.du1(x + stepSize / 2, u1 + stepSize * k21 / 2, u2 + stepSize * k22 / 2)
        k32 = self.du2(x + stepSize / 2, u1 + stepSize * k21 / 2, u2 + stepSize * k22 / 2)
        k41 = self.du1(x + stepSize, u1 + stepSize * k31, u2 + stepSize * k32)
        k42 = self.du2(x + stepSize, u1 + stepSize * k31, u2 + stepSize * k32)
        nextU1 = u1 + stepSize * (k11 + 2 * k21 + 2 * k31 + k41)
        nextU2 = u2 + stepSize * (k12 + 2 * k22 + 2 * k32 + k42)
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
            errorEstimate = max(errorEstimate1,errorEstimate2) / 15
            if (errorEstimate <= maxError) and (errorEstimate >= (maxError / 32)):
                nextStep = [nextX, nextU1, nextU2]
                steps.append(nextStep)
                currentStep = nextStep
                self.V2.append([nextU1HalfStep,nextU2HalfStep])
                self.OLP.append([errorEstimate1 * 16, errorEstimate2 * 16])
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
                self.OLP.append([errorEstimate1 * 16, errorEstimate2 * 16])
                self.Hi.append(currentStepSize)
                self.C1.append(C1)
                self.C2.append(C2)
                currentStepSize *= 2

                if nextX <=xMax and nextX >= xMax-self.epsilonG:
                    break

            else:
                currentStepSize /= 2
                C1 += 1

            if len(steps) != self.maxCount:
                continue
            break

        return steps




def fTest(x,y):
    return y

def f1(x, y):
    if abs(y) > 10 ** 6 or abs(x) > 10 ** 6:
        return 0  # или другое безопасное значение, например, 0
    return (y ** 2) * x / (1 + x ** 2) + y - (y ** 3) * np.sin(10 * x)

def C(x0,y0):
    return y0 / (np.exp(x0))

def update_plot():
    epsilonG = float(epsilonG_entry.get())
    maxCount = float(maxCount_entry.get())
    maxError = float(maxError_entry.get())
    h0 = float(h0_entry.get())
    xMax = float(xMax_entry.get())
    x_0 = float(x0_entry.get())
    y_0 = float(u0_entry.get())
    U1_0 = float(u1_0_entry.get())
    U2_0 = float(u2_0_entry.get())
    task = task_var.get()
    step_type = step_type_var.get()
    a = float(a_entry.get())
    b = float(b_entry.get())
    if(task == "Основная 2"):
        func1 = RungeKuttaSystem(h0, x_0, U1_0, U2_0, maxCount, epsilonG, a, b)
        data = np.array([func1.variableSteps(xMax, maxError)]) if step_type == "Переменный" else np.array(
            [func1.fixecStep(xMax)])
        x, u1, u2 = data.T
        V2 = func1.V2
        OLP = func1.OLP
        Hi = func1.Hi
        C1 = func1.C1
        C2 = func1.C2

        tree.delete(*tree.get_children())

        Data = []
        for i in range(1, len(x)):
            Data.append((i, x[i], [u1[i], u2[i]], V2[i - 1] if i - 1 < len(V2) else "",
                         [V2[i - 1][0] - u1[i], V2[i - 1][1] - u2[i]] if i - 1 < len(V2) else "",
                         OLP[i - 1] if i - 1 < len(OLP) else "", Hi[i - 1] if i - 1 < len(Hi) else "",
                         C1[i - 1] if i - 1 < len(C1) else "",
                         C2[i - 1] if i - 1 < len(C2) else "", "", ""))

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
        tk.Label(results_window, text=difference).grid(row=1, column=1)

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
        tk.Label(results_window, text=max_h_x).grid(row=9, column=1)

        tk.Label(results_window, text="Значение x для минимального Hi:").grid(row=10, column=0)
        tk.Label(results_window, text=min_h_x).grid(row=10, column=1)

        fig, axarr = plt.subplots(3, sharex=True, figsize=(8, 10))

        axarr[0].plot(x, u1, label='U1(x)')
        axarr[0].set_ylabel('U1')
        axarr[0].legend()

        axarr[1].plot(x, u2, label='U2(x)')
        axarr[1].set_ylabel('U2')
        axarr[1].legend()

        axarr[2].plot(u1, u2, label='U2(U1)')
        axarr[2].set_xlabel('U1')
        axarr[2].set_ylabel('U2')
        axarr[2].legend()

        plt.xlabel("x")
        plt.tight_layout()
        plt.show()

    else:
        func1 = RungeKutta(h0, x_0, y_0, maxCount, epsilonG, f1 if task == "Основная 1" else fTest)
        data = np.array([func1.variableStep(xMax, maxError)]) if step_type == "Переменный" else np.array(
            [func1.fixedStep(xMax)])
        x, y = data.T
        V2 = func1.V2
        OLP = func1.OLP
        Hi = func1.Hi
        C1 = func1.C1
        C2 = func1.C2

        tree.delete(*tree.get_children())

        C_value = C(x_0, y_0)

        Data = []
        for i in range(1, len(x)):
            ui = C_value*np.exp(x[i])
            Data.append((i, x[i], y[i], V2[i-1] if i-1 < len(V2) else "", y[i] - V2[i-1] if i-1 < len(V2) else "",
                         OLP[i-1] if i-1 < len(OLP) else "", Hi[i-1] if i-1 < len(Hi) else "", C1[i-1] if i-1 < len(C1) else "",
                         C2[i-1] if i-1 < len(C2) else "", ui if task == "Тестовая" else "", y[i]-ui if task == "Тестовая" else ""))

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
        tk.Label(results_window, text=difference).grid(row=1, column=1)

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
        tk.Label(results_window, text=max_h_x).grid(row=9, column=1)

        tk.Label(results_window, text="Значение x для минимального Hi:").grid(row=10, column=0)
        tk.Label(results_window, text=min_h_x).grid(row=10, column=1)


        plt.cla()
        plt.plot(x, y, label=f'v(x)')
        plt.xlabel("x")
        plt.ylabel("u")



        if task == "Тестовая":
            x_C = np.linspace(x_0, xMax, 100)

            y_C = C_value * np.exp(x_C)

            plt.plot(x_C, y_C, label=f'u(x)', linestyle='--')

        plt.legend()

        plt.title("График v(x)")
        plt.draw()

root = tk.Tk()
root.title("Численные методы лаб. Работа")

frame = ttk.Frame(root)
frame.pack(padx=10, pady=10)

u1_0_label = ttk.Label(frame, text="U1_0:")
u1_0_label.grid(row=0, column=2)
u1_0_entry = ttk.Entry(frame)
u1_0_entry.grid(row=0, column=3)
u1_0_entry.insert(0,"1")

u2_0_label = ttk.Label(frame, text="U2_0:")
u2_0_label.grid(row=1, column=2)
u2_0_entry = ttk.Entry(frame)
u2_0_entry.grid(row=1, column=3)
u2_0_entry.insert(0,"1")


maxCount_label = ttk.Label(frame, text="Максимальное количество итераций:")
maxCount_label.grid(row=0, column=0)
maxCount_entry = ttk.Entry(frame)
maxCount_entry.grid(row=0, column=1)
maxCount_entry.insert(0, "10000")

maxError_label = ttk.Label(frame, text="Максимальная ошибка:")
maxError_label.grid(row=1, column=0)
maxError_entry = ttk.Entry(frame)
maxError_entry.grid(row=1, column=1)
maxError_entry.insert(0, "0.0001")

h0_label = ttk.Label(frame, text="Начальный шаг:")
h0_label.grid(row=2, column=0)
h0_entry = ttk.Entry(frame)
h0_entry.grid(row=2, column=1)
h0_entry.insert(0, "0.01")

xMax_label = ttk.Label(frame, text="Правая граница:")
xMax_label.grid(row=3, column=0)
xMax_entry = ttk.Entry(frame)
xMax_entry.grid(row=3, column=1)
xMax_entry.insert(0, "1.7")

x0_label = ttk.Label(frame, text="x0:")
x0_label.grid(row=4, column=0)
x0_entry = ttk.Entry(frame)
x0_entry.grid(row=4, column=1)
x0_entry.insert(0, "1")

u0_label = ttk.Label(frame, text="u0:")
u0_label.grid(row=5, column=0)
u0_entry = ttk.Entry(frame)
u0_entry.grid(row=5, column=1)
u0_entry.insert(0, "1")

epsilonG_label = ttk.Label(frame, text="Епселон граничный:")
epsilonG_label.grid(row=6, column=0)
epsilonG_entry = ttk.Entry(frame)
epsilonG_entry.grid(row=6, column=1)
epsilonG_entry.insert(0, "0.001")

task_var = tk.StringVar()

task_label = ttk.Label(frame, text="Выберите задачу:")
task_label.grid(row=9, column=0)
task_option = ttk.OptionMenu(frame, task_var, "Тестовая", "Тестовая", "Основная 1","Основная 2")
task_option.grid(row=9, column=1)

step_type_label = ttk.Label(frame, text="Выберите шаг:")
step_type_label.grid(row=10, column=0)
step_type_var = tk.StringVar()
step_type_var.set("Фиксированный")
step_type_option = ttk.OptionMenu(frame, step_type_var, "Фиксированный","Фиксированный", "Переменный")
step_type_option.grid(row=10, column=1)

a_label = ttk.Label(frame, text="a:")
a_label.grid(row=7, column=0)
a_entry = ttk.Entry(frame)
a_entry.grid(row=7, column=1)
a_entry.insert(0, "1")

b_label = ttk.Label(frame, text="b:")
b_label.grid(row=8, column=0)
b_entry = ttk.Entry(frame)
b_entry.grid(row=8, column=1)
b_entry.insert(0, "1")

update_button = ttk.Button(frame, text="Обновить", command=update_plot)
update_button.grid(row=11, columnspan=2)


columns = ("i", "xi", "vi", "v2i", "vi-v2i", "OLP", "hi", "C1", "C2", "ui", "ui-vi")
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