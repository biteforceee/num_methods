import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import tkinter as tk
from tkinter import ttk
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure

import warnings

##warnings.filterwarnings("ignore")
class Phi:
    def func(self, x):
        if (-1 <= x < 0):
            return x ** 3 + 3 * x ** 2
        elif (0 <= x <= 1):
            return -x ** 3 + 3 * x ** 2

    def derivative(self, x):
        if (-1 <= x < 0):
            return 3 * x ** 2 + 6 * x
        elif (0 <= x <= 1):
            return (-3) * x ** 2 + 6 * x

    def derivative_2(self, x):
        if (-1 <= x < 0):
            return 6 * x + 6
        elif (0 <= x <= 1):
            return (-6) * x + 6


class main_func:
    def func(self, x):
        return np.log(x+1)/(x+1)

    def derivative(self, x):
        return (1-np.log(x+1))/((x+1)**2)

    def derivative_2(self, x):
        return -(3-2*np.log(x+1))/((x+1)**3)


class Main_Func:
    def func(self, x):
        return main_func().func(x) + np.cos(10 * x)

    def derivative(self, x):
        return main_func().derivative(x) - 10 * np.sin(10 * x)

    def derivative_2(self, x):
        return main_func().derivative_2(x) - 100 * np.cos(10 * x)


class Spline:
    def __init__(self, n, a, b, eps, func):
        self.n = n
        self.a = a
        self.b = b
        self.h = (b - a) / n
        self.av = np.zeros(n + 1)
        self.bv = np.zeros(n + 1)
        self.cv = np.zeros(n + 1)
        self.dv = np.zeros(n + 1)
        self.last_eps = np.inf
        self.eps = eps
        self.func = func

        self.max_error_spline = None
        self.max_error_derivative = None
        self.max_error_derivative_2 = None

        for i in range(n + 1):
            self.av[i] = func.func(self.a + self.h * i)

    def count_coeffs(self):
        alpha = np.zeros(self.n)
        beta = np.zeros(self.n)

        for i in range(1, self.n):
            alpha[i] = -self.h / (self.h * alpha[i - 1] + 4 * self.h)
            x = self.a + self.h * i
            x_prev = x - self.h
            x_next = x + self.h
            Fi = 6 * (self.func.func(x_next) - 2 * self.func.func(x) + self.func.func(x_prev)) / self.h
            beta[i] = (Fi - self.h * beta[i - 1]) / (self.h * alpha[i - 1] + 4 * self.h)

        for i in reversed(range(1, self.n)):
            self.cv[i] = alpha[i] * self.cv[i + 1] + beta[i]

        for i in range(1, self.n + 1):
            self.dv[i] = (self.cv[i] - self.cv[i - 1]) / self.h
            self.bv[i] = (self.av[i] - self.av[i - 1]) / self.h + self.cv[i] * self.h / 3 + self.cv[i - 1] * self.h / 6

    def plot_spline(self, ticks, ax=None):
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))

        for i in range(1, self.n + 1):
            x_left = self.a + self.h * (i - 1)
            x_right = x_left + self.h
            x_space = np.linspace(x_left, x_right, ticks)
            spline_i = [self.av[i] + self.bv[i] * (x - x_right) + self.cv[i] / 2 * \
                        (x - x_right) ** 2 + self.dv[i] / 6 * (x - x_right) ** 3 for x in x_space]
            ax.plot(x_space, spline_i, color="purple", linewidth=2.0, label='Spline' if i == 1 else "")
            func_i = [self.func.func(x) for x in x_space]
            ax.plot(x_space, func_i, color="yellow", label='Function' if i == 1 else "")

        xticks = [self.a + i * self.h for i in range(self.n + 1)]
        ax.set_xticks(xticks if self.n < 10 else [])
        ax.grid()
        ax.set_title("F(x) vs S(x)")
        ax.legend()

        if ax is None:
            plt.show()
            return fig
        return None

    def plot_derivative(self, ticks, ax=None):
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))

        for i in range(1, self.n + 1):
            x_left = self.a + self.h * (i - 1)
            x_right = x_left + self.h
            x_space = np.linspace(x_left, x_right, ticks)
            spline_i = [self.bv[i] + self.cv[i] * (x - x_right) + self.dv[i] / 2 * (x - x_right) ** 2 for x in x_space]
            ax.plot(x_space, spline_i, color="purple", linewidth=2.0, label="Spline'" if i == 1 else "")
            func_i = [self.func.derivative(x) for x in x_space]
            ax.plot(x_space, func_i, color="yellow", label="Function'" if i == 1 else "")

        xticks = [self.a + i * self.h for i in range(self.n + 1)]
        ax.set_xticks(xticks if self.n < 10 else [])
        ax.grid()
        ax.set_title("F'(x) vs S'(x)")
        ax.legend()

        if ax is None:
            plt.show()
            return fig
        return None

    def plot_derivative_2(self, ticks, ax=None):
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))

        for i in range(1, self.n + 1):
            x_left = self.a + self.h * (i - 1)
            x_right = x_left + self.h
            x_space = np.linspace(x_left, x_right, ticks)
            spline_i = [self.cv[i] + self.dv[i] * (x - x_right) for x in x_space]
            ax.plot(x_space, spline_i, color="purple", linewidth=2.0, label="Spline''" if i == 1 else "")
            func_i = [self.func.derivative_2(x) for x in x_space]
            ax.plot(x_space, func_i, color="yellow", label="Function''" if i == 1 else "")

        xticks = [self.a + i * self.h for i in range(self.n + 1)]
        ax.set_xticks(xticks if self.n < 10 else [])
        ax.grid()
        ax.set_title("F''(x) vs S''(x)")
        ax.legend()

        if ax is None:
            plt.show()
            return fig
        return None

    def coeffs_to_table(self):
        grid = [self.a + i * self.h for i in range(self.n + 1)]
        coeffs = {'Xi-1': grid[0:-1], 'Xi': grid[1:], 'a': self.av[1:], \
                  'b': self.bv[1:], 'c': self.cv[1:], 'd': self.dv[1:]}
        self.df_coeffs = pd.DataFrame(coeffs)

    def plot_error_spline(self, ticks, ax=None):
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))

        self.max_error_spline = []
        for i in range(1, self.n + 1):
            x_left = self.a + self.h * (i - 1)
            x_right = x_left + self.h
            x_space = np.linspace(x_left, x_right, ticks)
            spline_i = [self.av[i] + self.bv[i] * (x - x_right) + self.cv[i] / 2 * \
                        (x - x_right) ** 2 + self.dv[i] / 6 * (x - x_right) ** 3 for x in x_space]
            func_i = [self.func.func(x) for x in x_space]
            error_i = np.absolute(np.array(spline_i) - np.array(func_i))
            ax.plot(x_space, error_i, color="purple")
            self.max_error_spline.append(np.max(error_i))

        xticks = [self.a + i * self.h for i in range(self.n + 1)]
        ax.set_xticks(xticks if self.n < 10 else [])
        ax.grid()
        ax.set_title("Погрешность |F(x) - S(x)|")

        if ax is None:
            plt.show()
            return fig
        return None

    def plot_error_derivative(self, ticks, ax=None):
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))

        self.max_error_derivative = []
        for i in range(1, self.n + 1):
            x_left = self.a + self.h * (i - 1)
            x_right = x_left + self.h
            x_space = np.linspace(x_left, x_right, ticks)
            spline_i = [self.bv[i] + self.cv[i] * (x - x_right) + self.dv[i] / 2 * (x - x_right) ** 2 for x in x_space]
            func_i = [self.func.derivative(x) for x in x_space]
            error_i = np.absolute(np.array(spline_i) - np.array(func_i))
            ax.plot(x_space, error_i, color="purple")
            self.max_error_derivative.append(np.max(error_i))

        xticks = [self.a + i * self.h for i in range(self.n + 1)]
        ax.set_xticks(xticks if self.n < 10 else [])
        ax.grid()
        ax.set_title("Погрешность |F'(x) - S'(x)|")

        if ax is None:
            plt.show()
            return fig
        return None

    def plot_error_derivative_2(self, ticks, ax=None):
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))

        self.max_error_derivative_2 = []
        for i in range(1, self.n + 1):
            x_left = self.a + self.h * (i - 1)
            x_right = x_left + self.h
            x_space = np.linspace(x_left, x_right, ticks)
            spline_i = [self.cv[i] + self.dv[i] * (x - x_right) for x in x_space]
            func_i = [self.func.derivative_2(x) for x in x_space]
            error_i = np.absolute(np.array(spline_i) - np.array(func_i))
            ax.plot(x_space, error_i, color="purple")
            self.max_error_derivative_2.append(np.max(error_i))

        xticks = [self.a + i * self.h for i in range(self.n + 1)]
        ax.set_xticks(xticks if self.n < 10 else [])
        ax.grid()
        ax.set_title("Погрешность |F''(x) - S''(x)|")

        if ax is None:
            plt.show()
            return fig
        return None

    def get_results_text(self):
        self.coeffs_to_table()
        np.set_printoptions(formatter={'float_kind': '{:e}'.format})
        text = f"Ceтка сплайна: n = {self.n}\n"
        text += f"Контрольная сетка: n = {self.n * 4}\n"
        text += f"max|F(x) - S(x)| = {np.array(self.max_error_spline).max():e}\n"
        text += f"max|F'(x) - S'(x)| = {np.array(self.max_error_derivative).max():e}\n"
        text += f"max|F''(x) - S''(x)| = {np.array(self.max_error_derivative_2).max():e}\n"
        return text

    def evaluate(self, x):
        """
        Вычисляет значение сплайна в точке x
        """
        if x < self.a or x > self.b:
            raise ValueError(f"Точка x={x} выходит за границы интервала [{self.a}, {self.b}]")

        # Определяем интервал, в который попадает x
        i = min(int((x - self.a) / self.h), self.n - 1)
        x_left = self.a + i * self.h
        x_right = x_left + self.h

        # Вычисляем значение сплайна на этом интервале
        dx = x - x_right
        return (self.av[i + 1] +
                self.bv[i + 1] * dx +
                self.cv[i + 1] / 2 * dx ** 2 +
                self.dv[i + 1] / 6 * dx ** 3)

    def evaluate_derivative(self, x):
        """
        Вычисляет значение первой производной сплайна в точке x
        """
        if x < self.a or x > self.b:
            raise ValueError(f"Точка x={x} выходит за границы интервала [{self.a}, {self.b}]")

        # Определяем интервал
        i = min(int((x - self.a) / self.h), self.n - 1)
        x_left = self.a + i * self.h
        x_right = x_left + self.h

        # Вычисляем производную
        dx = x - x_right
        return (self.bv[i + 1] +
                self.cv[i + 1] * dx +
                self.dv[i + 1] / 2 * dx ** 2)

    def evaluate_derivative2(self, x):
        """
        Вычисляет значение второй производной сплайна в точке x
        """
        if x < self.a or x > self.b:
            raise ValueError(f"Точка x={x} выходит за границы интервала [{self.a}, {self.b}]")

        # Определяем интервал
        i = min(int((x - self.a) / self.h), self.n - 1)
        x_left = self.a + i * self.h
        x_right = x_left + self.h

        # Вычисляем вторую производную
        dx = x - x_right
        return self.cv[i + 1] + self.dv[i + 1] * dx

class SplineApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Spline Interpolation")

        # Parameters
        self.n = tk.IntVar(value=50)
        self.ifTest = tk.BooleanVar(value=False)
        self.func_var = tk.StringVar(value="Main_Func")
        self.a = tk.DoubleVar(value=0.2)
        self.b = tk.DoubleVar(value=2.0)
        self.ticks = tk.IntVar(value=50)
        self.spline = None
        # Create UI
        self.create_widgets()

    def create_widgets(self):
        # Control frame
        control_frame = ttk.Frame(self.root, padding="10")
        control_frame.grid(row=0, column=0, sticky="nw")

        ttk.Label(control_frame, text="n:").grid(row=0, column=0, sticky="w")
        ttk.Entry(control_frame, textvariable=self.n).grid(row=0, column=1, sticky="w")

        ttk.Checkbutton(control_frame, text="Test Function", variable=self.ifTest,
                        command=self.toggle_test_function).grid(row=1, column=0, columnspan=2, sticky="w")

        ttk.Label(control_frame, text="Function:").grid(row=2, column=0, sticky="w")
        ttk.Combobox(control_frame, textvariable=self.func_var,
                     values=["Main_Func", "main_func", "Phi"], state="readonly").grid(row=2, column=1, sticky="w")

        ttk.Label(control_frame, text="a:").grid(row=3, column=0, sticky="w")
        ttk.Entry(control_frame, textvariable=self.a).grid(row=3, column=1, sticky="w")

        ttk.Label(control_frame, text="b:").grid(row=4, column=0, sticky="w")
        ttk.Entry(control_frame, textvariable=self.b).grid(row=4, column=1, sticky="w")

        ttk.Label(control_frame, text="Ticks:").grid(row=5, column=0, sticky="w")
        ttk.Entry(control_frame, textvariable=self.ticks).grid(row=5, column=1, sticky="w")

        ttk.Button(control_frame, text="Calculate", command=self.calculate).grid(row=6, column=0, columnspan=2, pady=10)

        # Кнопки для отдельных графиков
        plot_buttons = [
            ("F(x)", 'spline'),
            ("F'(x)", 'derivative'),
            ("F''(x)", 'derivative2'),
            ("|F(x) - S(x)|", 'error_spline'),
            ("|F'(x) - S'(x)|", 'error_derivative'),
            ("|F''(x) - S''(x)|", 'error_derivative2')
        ]

        for i, (text, plot_type) in enumerate(plot_buttons):
            ttk.Button(
                control_frame,
                text=text,
                command=lambda pt=plot_type: self.show_separate_plot(pt)
            ).grid(row=7 + i // 2, column=i % 2, pady=5, sticky="ew")

        # Notebook для таблиц
        self.notebook = ttk.Notebook(self.root)
        self.notebook.grid(row=1, column=0, columnspan=2, sticky="nsew", padx=10, pady=10)

        # Фреймы для таблиц
        self.coeff_frame = ttk.Frame(self.notebook)
        self.error_frame = ttk.Frame(self.notebook)

        self.notebook.add(self.coeff_frame, text="Коэффициенты сплайна")
        self.notebook.add(self.error_frame, text="Погрешности интерполяции")

        # Results frame
        results_frame = ttk.Frame(self.root, padding="10")
        results_frame.grid(row=0, column=1, sticky="nsew")

        self.text_results = tk.Text(results_frame, height=10, width=50)
        self.text_results.grid(row=0, column=0, sticky="nsew")

        # Plots frame
        plots_frame = ttk.Frame(self.root)
        plots_frame.grid(row=1, column=0, columnspan=2, sticky="nsew")

        # Create 6 subplots (2 rows, 3 columns)
        self.fig, self.axes = plt.subplots(2, 3, figsize=(15, 10))
        self.fig.tight_layout(pad=5.0)

        self.canvas = FigureCanvasTkAgg(self.fig, master=plots_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Configure grid weights
        self.root.columnconfigure(1, weight=1)
        self.root.rowconfigure(1, weight=1)

    def toggle_test_function(self):
        if self.ifTest.get():
            self.a.set(-1.0)
            self.b.set(1.0)
            self.func_var.set("Phi")
        else:
            self.a.set(0.2)
            self.b.set(2.0)
            self.func_var.set("Main_Func")

    def show_separate_plot(self, plot_type):
        """Отображает выбранный график в отдельном окне"""
        try:
            # Проверяем, был ли рассчитан сплайн
            if not hasattr(self, 'spline') or self.spline is None:
                tk.messagebox.showwarning("Предупреждение",
                                          "Сначала рассчитайте сплайн, нажав кнопку 'Calculate'")
                return

            # Получаем параметры
            ticks = self.ticks.get()

            # Создаем новое окно
            plot_window = tk.Toplevel(self.root)
            plot_window.title(f"График: {plot_type.replace('_', ' ').title()}")
            plot_window.geometry("800x600")

            # Создаем фигуру и оси
            fig = Figure(figsize=(10, 6))
            ax = fig.add_subplot(111)

            # Строим график
            plot_methods = {
                'spline': self.spline.plot_spline,
                'derivative': self.spline.plot_derivative,
                'derivative2': self.spline.plot_derivative_2,
                'error_spline': self.spline.plot_error_spline,
                'error_derivative': self.spline.plot_error_derivative,
                'error_derivative2': self.spline.plot_error_derivative_2
            }

            plot_methods[plot_type](ticks, ax)

            # Встраиваем график в окно
            canvas = FigureCanvasTkAgg(fig, master=plot_window)
            canvas.draw()
            canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

            # Добавляем панель инструментов
            toolbar = NavigationToolbar2Tk(canvas, plot_window)
            toolbar.update()
            toolbar.pack(side=tk.BOTTOM, fill=tk.X)

        except Exception as e:
            tk.messagebox.showerror("Ошибка",
                                    f"Не удалось построить график:\n{str(e)}")

    def calculate(self):
        try:
            # Получаем параметры
            n = self.n.get()
            ifTest = self.ifTest.get()
            func_name = self.func_var.get()
            a = self.a.get()
            b = self.b.get()
            ticks = self.ticks.get()

            # Выбираем функцию
            if ifTest:
                func = Phi()
            else:
                if func_name == "Main_Func":
                    func = Main_Func()
                else:
                    func = main_func()

            # Создаем и вычисляем сплайн
            eps = 1e-8
            self.spline = Spline(n, a, b, eps, func)
            self.spline.count_coeffs()

            # Очищаем предыдущие графики
            for ax in self.axes.ravel():
                ax.clear()

            # Строим новые графики
            self.spline.plot_spline(ticks, self.axes[0, 0])
            self.spline.plot_derivative(ticks, self.axes[0, 1])
            self.spline.plot_derivative_2(ticks, self.axes[0, 2])
            self.spline.plot_error_spline(ticks, self.axes[1, 0])
            self.spline.plot_error_derivative(ticks, self.axes[1, 1])
            self.spline.plot_error_derivative_2(ticks, self.axes[1, 2])

            # Обновляем таблицы
            self.show_tables_window()

            # Показываем результаты
            self.text_results.delete(1.0, tk.END)
            self.text_results.insert(tk.END, self.spline.get_results_text())

            # Обновляем canvas
            self.canvas.draw()

        except Exception as e:
            tk.messagebox.showerror("Ошибка", f"Ошибка при расчете: {str(e)}")

    def show_tables_window(self):
        """Создает окно с результатами в стиле Toplevel с таблицами"""
        if hasattr(self, 'results_window') and self.results_window.winfo_exists():
            self.results_window.lift()
            return

        self.results_window = tk.Toplevel(self.root)
        self.results_window.title("Результаты интерполяции")
        self.results_window.geometry("900x600")

        # Создаем фреймы для разных секций
        results_frame = ttk.Frame(self.results_window)
        results_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Секция с основными результатами
        self._create_results_labels(results_frame)

        # Notebook для таблиц
        notebook = ttk.Notebook(results_frame)
        notebook.pack(fill=tk.BOTH, expand=True, pady=10)

        # Таблица коэффициентов
        coeff_frame = ttk.Frame(notebook)
        notebook.add(coeff_frame, text="Коэффициенты сплайна")
        self._create_coefficients_table(coeff_frame)

        # Таблица погрешностей
        error_frame = ttk.Frame(notebook)
        notebook.add(error_frame, text="Погрешности")
        self._create_errors_table(error_frame)

        # Кнопка закрытия
        btn_close = ttk.Button(
            self.results_window,
            text="Закрыть",
            command=self.results_window.destroy
        )
        btn_close.pack(side=tk.BOTTOM, pady=10)

    def _create_results_labels(self, parent_frame):
        """Создает метки с основными результатами"""
        if not self.spline:
            return

        # Фрейм для меток
        labels_frame = ttk.Frame(parent_frame)
        labels_frame.pack(fill=tk.X, pady=5)

        # Метки с результатами
        ttk.Label(labels_frame, text="Количество узлов:").grid(row=0, column=0, sticky="e")
        ttk.Label(labels_frame, text=str(self.spline.n)).grid(row=0, column=1, sticky="w")

        ttk.Label(labels_frame, text="Интервал:").grid(row=1, column=0, sticky="e")
        ttk.Label(labels_frame, text=f"[{self.spline.a:.2f}, {self.spline.b:.2f}]").grid(row=1, column=1, sticky="w")

        if hasattr(self.spline, 'max_error_spline') and self.spline.max_error_spline:
            ttk.Label(labels_frame, text="Макс. погрешность:").grid(row=2, column=0, sticky="e")
            ttk.Label(labels_frame, text=f"{np.max(self.spline.max_error_spline):.2e}").grid(row=2, column=1,
                                                                                             sticky="w")

        # Добавьте другие метки по аналогии с вашим примером
        # ...

    def _create_coefficients_table(self, parent_frame):
        """Создает таблицу коэффициентов"""
        if not self.spline:
            return

        # Создаем Treeview
        tree = ttk.Treeview(
            parent_frame,
            columns=("i", "xi-1", "xi", "ai", "bi", "ci", "di"),
            show="headings",
            height=10
        )

        # Настраиваем столбцы
        tree.heading("i", text="i")
        tree.heading("xi-1", text="xi-1")
        tree.heading("xi", text="xi")
        tree.heading("ai", text="ai")
        tree.heading("bi", text="bi")
        tree.heading("ci", text="ci")
        tree.heading("di", text="di")

        # Заполняем данными
        for i in range(1, self.spline.n + 1):
            tree.insert("", "end", values=(
                i,
                f"{self.spline.a + self.spline.h * (i - 1):.4f}",
                f"{self.spline.a + self.spline.h * i:.4f}",
                f"{self.spline.av[i]:.6f}",
                f"{self.spline.bv[i]:.6f}",
                f"{self.spline.cv[i]:.6f}",
                f"{self.spline.dv[i]:.6f}"
            ))

        # Добавляем прокрутку
        scrollbar = ttk.Scrollbar(parent_frame, orient="vertical", command=tree.yview)
        tree.configure(yscrollcommand=scrollbar.set)

        # Размещаем на форме
        tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

    def _create_errors_table(self, parent_frame):
        """Создает таблицу с максимальными погрешностями"""
        if not self.spline:
            return

        tree = ttk.Treeview(parent_frame,
                            columns=(
                                "j", "xj", "F(xj)", "S(xj)", "F-S", "F'(xj)", "S'(xj)", "F'-S'", "F''(xj)", "S''(xj)",
                                "F''-S''"),
                            show="headings", height=10)

        # Настраиваем столбцы
        columns = ["j", "xj", "F(xj)", "S(xj)", "F-S", "F'(xj)", "S'(xj)", "F'-S'", "F''(xj)", "S''(xj)", "F''-S''"]
        for col in columns:
            tree.heading(col, text=col)
            tree.column(col, width=80, anchor="e")

        # Используем уже рассчитанные погрешности из класса Spline
        N = min(20, self.spline.n * 4)  # Ограничиваем количество точек
        h_control = (self.spline.b - self.spline.a) / N

        for j in range(N + 1):
            xj = self.spline.a + j * h_control

            # Используем методы класса Spline для получения значений
            f_val = self.spline.func.func(xj)
            s_val = self.spline.evaluate(xj)
            error = f_val - s_val

            f_deriv = self.spline.func.derivative(xj)
            s_deriv = self.spline.evaluate_derivative(xj)
            deriv_error = f_deriv - s_deriv

            f_deriv2 = self.spline.func.derivative_2(xj)
            s_deriv2 = self.spline.evaluate_derivative2(xj)
            deriv2_error = f_deriv2 - s_deriv2

            # Добавляем строку в таблицу (каждую 2-ю точку для краткости)
            if j % 2 == 0:
                tree.insert("", "end", values=(
                    j,
                    f"{xj:.4f}",
                    f"{f_val:.6f}",
                    f"{s_val:.6f}",
                    f"{error:.2e}",
                    f"{f_deriv:.6f}",
                    f"{s_deriv:.6f}",
                    f"{deriv_error:.2e}",
                    f"{f_deriv2:.6f}",
                    f"{s_deriv2:.6f}",
                    f"{deriv2_error:.2e}"
                ))

        # Добавляем полосу прокрутки
        scrollbar = ttk.Scrollbar(self.error_frame, orient="vertical", command=tree.yview)
        tree.configure(yscrollcommand=scrollbar.set)

        # Размещаем на форме
        tree.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

# Create and run the application
if __name__ == "__main__":
    root = tk.Tk()
    app = SplineApp(root)
    root.mainloop()