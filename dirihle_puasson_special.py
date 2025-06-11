import numpy as np
import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
import matplotlib.pyplot as plt


def u_exact(x, y):
    return np.sin(np.pi * x * y)


def f_rhs(x, y):
    return -(np.pi ** 2 * (x ** 2 + y ** 2) * np.sin(np.pi * x * y))


def create_domain_mask(x, y):
    X, Y = np.meshgrid(x, y, indexing='ij')
    mask = np.zeros_like(X, dtype=bool)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            xi, yj = x[i], y[j]
            in_cross = (1.25 <= xi <= 1.75) or (2.25 <= yj <= 2.75)
            in_corner = (1.75 <= xi <= 2.0) and (2.75 <= yj <= 3.0)
            on_shape = (xi == 1.25) and (2. <= yj <= 2.25)
            on_shape2 = (xi == 1.25) and (2.75 <= yj <= 3.)
            on_shape3 = (xi == 1.75) and (2. <= yj <= 2.25)
            on_shape4 = (yj == 2.25) and (1. <= xi <= 1.25)
            on_shape5 = (yj == 2.25) and (1.75 <= xi <= 2.)
            on_shape6 = (yj == 2.75) and (1. <= xi <= 1.25)
            # in_cross = (1.25 - 1e-10 <= xi <= 1.75 + 1e-10) or (2.25 - 1e-10 <= yj <= 2.75 + 1e-10)
            # in_corner = (1.75 - 1e-10 <= xi <= 2.0 + 1e-10) and (2.75 - 1e-10 <= yj <= 3.0 + 1e-10)
            mask[
                i, j] = in_cross or in_corner or on_shape or on_shape2 or on_shape3 or on_shape4 or on_shape5 or on_shape6
    return mask


def sor_method(n, m, eps, max_iter, omegaR):
    a, b = 1, 2
    c, d = 2, 3
    hx = (b - a) / n
    hy = (d - c) / m
    x = np.linspace(a, b, n + 1)
    y = np.linspace(c, d, m + 1)
    X, Y = np.meshgrid(x, y, indexing="ij")

    h2 = 1 / hx ** 2
    k2 = 1 / hy ** 2
    a2 = -2 * (h2 + k2)

    mask = create_domain_mask(x, y)
    U = np.zeros_like(X)
    F = f_rhs(X, Y) * mask
    U[~mask] = u_exact(X[~mask], Y[~mask])
    is_boundary = np.zeros_like(mask, dtype=bool)
    # Индексы, если сетка кратна 4
    i0, i1 = 0, n
    j0, j1 = 0, m
    i14 = n // 4
    i34 = 3 * n // 4
    j14 = m // 4
    j34 = 3 * m // 4

    # Вертикальные граничные "ножки"
    for j in range(0, j14 + 1):
        if mask[i14, j]:
            U[i14, j] = u_exact(X[i14, j], Y[i14, j])
            is_boundary[i14, j] = True
        if mask[i34, j]:
            U[i34, j] = u_exact(X[i34, j], Y[i34, j])
            is_boundary[i34, j] = True

    for j in range(j34, m):
        if mask[i14, j]:
            U[i14, j] = u_exact(X[i14, j], Y[i14, j])
            is_boundary[i14, j] = True

    # Горизонтальные граничные "перемычки"
    for i in range(0, i14 + 1):
        if mask[i, j14]:
            U[i, j14] = u_exact(X[i, j14], Y[i, j14])
            is_boundary[i, j14] = True
        if mask[i, j34]:
            U[i, j34] = u_exact(X[i, j34], Y[i, j34])
            is_boundary[i, j34] = True

    for i in range(i34, n):
        if mask[i, j14]:
            U[i, j14] = u_exact(X[i, j14], Y[i, j14])
            is_boundary[i, j14] = True

    # Границы всей области
    for i in range(n + 1):
        for j in [0, m]:
            if mask[i, j]:
                U[i, j] = u_exact(X[i, j], Y[i, j])
                is_boundary[i, j] = True

    for j in range(m + 1):
        for i in [0, n]:
            if mask[i, j]:
                U[i, j] = u_exact(X[i, j], Y[i, j])
                is_boundary[i, j] = True
    max_initial_residual = 0.0

    # --- Вычисление начальной невязки (нулевое приближение) ---

    for i in range(1, n):
        for j in range(1, m):
            if mask[i, j] and not is_boundary[i, j]:
                laplace = h2 * (U[i + 1, j] + U[i - 1, j]) + \
                          k2 * (U[i, j + 1] + U[i, j - 1]) + \
                          a2 * U[i, j]
                residual = abs(laplace - F[i, j])
                if residual > max_initial_residual:
                    max_initial_residual = residual
    # --- Вычисление начальной невязки по евклидовой норме ---
    sum_sq_residuals = 0.0
    count = 0
    max_initial_residual_euql = 0.0
    for i in range(1, n):
        for j in range(1, m):
            if mask[i, j] and not is_boundary[i, j]:
                laplace = (
                        h2 * (U[i + 1, j] + U[i - 1, j]) +
                        k2 * (U[i, j + 1] + U[i, j - 1]) +
                        a2 * U[i, j]
                )
                residual = laplace - F[i, j]
                sum_sq_residuals += residual ** 2
                count += 1

    if count > 0:
        max_initial_residual_euql = (sum_sq_residuals) ** 0.5
    else:
        max_initial_residual_euql = 0.0

    omega = omegaR
    S = 0

    while True:
        eps_max = 0
        for j in range(1, m):
            for i in range(1, n):
                if mask[i, j] and not is_boundary[i, j]:
                    v_old = U[i, j]
                    v_new = (
                            omega * (
                            h2 * (U[i + 1, j] + U[i - 1, j]) +
                            k2 * (U[i, j + 1] + U[i, j - 1]) -
                            F[i, j]
                    ) / -a2 +
                            (1 - omega) * v_old
                    )
                    eps_cur = abs(v_new - v_old)
                    eps_max = max(eps_max, eps_cur)
                    U[i, j] = v_new
        S += 1
        if eps_max <= eps or S >= max_iter:
            break

    residual = 0
    max_residual = 0
    max_residual_euql = 0
    for i in range(1, n):
        for j in range(1, m):
            if mask[i, j] and not is_boundary[i, j]:
                laplace = h2 * (U[i + 1, j] + U[i - 1, j]) + \
                          k2 * (U[i, j + 1] + U[i, j - 1]) + a2 * U[i, j]
                residual = abs(laplace - f_rhs(X[i, j], Y[i, j]))
                if residual > max_residual:
                    max_residual = residual

    sum_sq_residuals = 0.0
    count = 0

    for i in range(1, n):
        for j in range(1, m):
            if mask[i, j] and not is_boundary[i, j]:
                laplace = (
                        h2 * (U[i + 1, j] + U[i - 1, j]) +
                        k2 * (U[i, j + 1] + U[i, j - 1]) +
                        a2 * U[i, j]
                )
                residual = laplace - F[i, j]
                sum_sq_residuals += residual ** 2
                count += 1

    if count > 0:
        max_residual_euql = (sum_sq_residuals) ** 0.5  # среднеквадратичное значение
    else:
        max_residual_euql = 0.0

    return X, Y, U, mask, S, eps_max, max_residual, max_residual_euql, max_initial_residual, max_initial_residual_euql


# --- Интерфейс ---
def optimal_w(n, m):
    '''hx = 1 / n
    hy = 1 / m
    denom = (1 / (hx * 2)) ** 2 + (1 / (hy * 2)) ** 2
    sin_term_x = (np.sin(np.pi / (2 * n)) / hx) ** 2
    sin_term_y = (np.sin(np.pi / (2 * m)) / hy) ** 2
    k = 1 - 0.5 * (1 / denom) * 4 * (sin_term_x + sin_term_y)
    w = 2 / (1 + np.sqrt(1 - k ** 2))'''
    ro = (np.cos(np.pi / (n + 1)) + np.cos(np.pi / (m + 1))) / 2
    w = 2 / (1 + np.sqrt(1 - ro ** 2))
    return w


class SORApp:
    def __init__(self, root):
        self.root = root
        self.root.geometry("1000x700")  # <--- Увеличенное окно
        self.root.title("Метод Верхней Релаксации")

        self.frame = ttk.Frame(root, padding="10")
        self.frame.grid(row=0, column=0)

        # Ввод
        self._create_inputs()

        # Метка результата
        self.status_label = ttk.Label(self.frame, text="Ожидание запуска...", foreground="blue")
        self.status_label.grid(column=0, row=8, columnspan=2, pady=5)

    def _create_inputs(self):
        ttk.Label(self.frame, text="n (по x):").grid(column=0, row=0, sticky="e")
        self.n_entry = ttk.Entry(self.frame, width=10)
        self.n_entry.insert(0, "20")
        self.n_entry.grid(column=1, row=0)

        ttk.Label(self.frame, text="m (по y):").grid(column=0, row=1, sticky="e")
        self.m_entry = ttk.Entry(self.frame, width=10)
        self.m_entry.insert(0, "20")
        self.m_entry.grid(column=1, row=1)

        ttk.Label(self.frame, text="eps:").grid(column=0, row=2, sticky="e")
        self.eps_entry = ttk.Entry(self.frame, width=10)
        self.eps_entry.insert(0, "1e-6")
        self.eps_entry.grid(column=1, row=2)

        ttk.Label(self.frame, text="Макс. итераций:").grid(column=0, row=3, sticky="e")
        self.max_iter_entry = ttk.Entry(self.frame, width=10)
        self.max_iter_entry.insert(0, "10000")
        self.max_iter_entry.grid(column=1, row=3)

        ttk.Label(self.frame, text="ω (релаксация):").grid(column=0, row=4, sticky="e")
        self.omega_entry = ttk.Entry(self.frame, width=10)
        self.omega_entry.insert(0, "1.7")
        self.omega_entry.grid(column=1, row=4)
        self.use_optimal_w = tk.BooleanVar()
        self.use_optimal_w.set(False)  # по умолчанию — пользователь вручную вводит

        ttk.Checkbutton(
            self.frame,
            text="Оптимальное ω",
            variable=self.use_optimal_w
        ).grid(column=0, row=12, columnspan=2, pady=3)

        # Кнопки
        ttk.Button(self.frame, text="Calculate", command=self.calculate).grid(column=0, row=7, columnspan=2, pady=5)
        ttk.Button(self.frame, text="Show 2D", command=self.show_2d).grid(column=0, row=5, columnspan=2)
        ttk.Button(self.frame, text="Show 3D", command=self.show_3d).grid(column=0, row=6, columnspan=2)
        '''ttk.Button(self.frame, text="Показать таблицы в новом окне", command=self.show_result_window) \
            .grid(column=0, row=10, columnspan=2, pady=5)'''
        ttk.Button(self.frame, text="Показать численное решение", command=self.show_table_numeric) \
            .grid(column=0, row=9, columnspan=2, pady=2)

        ttk.Button(self.frame, text="Показать точное решение", command=self.show_table_exact) \
            .grid(column=0, row=10, columnspan=2, pady=2)

        ttk.Button(self.frame, text="Показать погрешность", command=self.show_table_error) \
            .grid(column=0, row=11, columnspan=2, pady=2)

    def create_table_window(self, title, data_matrix):
        n, m = self.X.shape[0] - 1, self.X.shape[1] - 1
        x_values = [f"{self.X[i, 0]:.3f}" for i in range(n + 1)]
        y_values = [f"{self.Y[0, j]:.3f}" for j in range(m + 1)]

        # Колонки: x0, x1, ..., потом под ними реальные значения x
        cols = ["yj\\xi", "Y"] + [f"x{i}" for i in range(n + 1)]

        win = tk.Toplevel(self.root)
        win.title(title)
        win.geometry("1200x600")

        frame = ttk.LabelFrame(win, text=title, padding=5)
        frame.pack(fill="both", expand=True, padx=10, pady=10)

        container = ttk.Frame(frame)
        container.pack(fill="both", expand=True)

        tree = ttk.Treeview(container, show='headings')
        tree["columns"] = cols
        for col in cols:
            tree.heading(col, text=col)
            tree.column(col, width=80, anchor="center")

        # Первая строка: индексы x
        header_row = ["X", "y\\x"] + x_values
        tree.insert("", "end", values=header_row)

        # Остальные строки: строки yj, y[j], значения
        for j in range(m + 1):
            row = [f"y{j}", y_values[j]]
            for i in range(n + 1):
                row.append(data_matrix[i, j] if self.mask[i, j] else "")
            tree.insert("", "end", values=row)

        vsb = ttk.Scrollbar(container, orient="vertical", command=tree.yview)
        hsb = ttk.Scrollbar(container, orient="horizontal", command=tree.xview)
        tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)

        tree.grid(row=0, column=0, sticky="nsew")
        vsb.grid(row=0, column=1, sticky="ns")
        hsb.grid(row=1, column=0, sticky="ew")
        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)

    '''def create_table_window(self, title, data_matrix):
        n, m = self.X.shape[0] - 1, self.X.shape[1] - 1
        cols = ["yj\\xi"] + [f"x{i}" for i in range(n + 1)]
        #cols = ["y\\x"] + [f"{self.X[i].shape[0]}" for i in range(n + 1)]

        win = tk.Toplevel(self.root)
        win.title(title)
        win.geometry("1200x600")

        frame = ttk.LabelFrame(win, text=title, padding=5)
        frame.pack(fill="both", expand=True, padx=10, pady=10)

        container = ttk.Frame(frame)
        container.pack(fill="both", expand=True)

        tree = ttk.Treeview(container, show='headings')
        tree["columns"] = cols
        for col in cols:
            tree.heading(col, text=col)
            tree.column(col, width=80, anchor="center")

        for j in range(m + 1):
            row = [f"y{j}"]
            for i in range(n + 1):
                row.append(data_matrix[i, j] if self.mask[i, j] else "")
            tree.insert("", "end", values=row)

        vsb = ttk.Scrollbar(container, orient="vertical", command=tree.yview)
        hsb = ttk.Scrollbar(container, orient="horizontal", command=tree.xview)
        tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)

        tree.grid(row=0, column=0, sticky="nsew")
        vsb.grid(row=0, column=1, sticky="ns")
        hsb.grid(row=1, column=0, sticky="ew")
        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)'''

    def show_table_numeric(self):
        if not hasattr(self, "U"):
            messagebox.showwarning("Нет данных", "Сначала нажмите 'Calculate'")
            return
        n, m = self.X.shape[0] - 1, self.X.shape[1] - 1
        V_values = np.full_like(self.U, "", dtype=object)
        for i in range(n + 1):
            for j in range(m + 1):
                if self.mask[i, j]:
                    V_values[i, j] = f"{self.U[i, j]:.5f}"
                # Показываем ВСЕ узлы, в которых посчитано значение
                '''if not np.isnan(self.U[i, j]):
                    V_values[i, j] = f"{self.U[i, j]:.5f}"'''
        self.create_table_window("Численное решение v(N)(x, y)", V_values)

    def show_table_exact(self):
        if not hasattr(self, "U"):
            messagebox.showwarning("Нет данных", "Сначала нажмите 'Calculate'")
            return
        U_ref = u_exact(self.X, self.Y)
        n, m = self.X.shape[0] - 1, self.X.shape[1] - 1
        exact_values = np.full_like(self.U, "", dtype=object)
        for i in range(n + 1):
            for j in range(m + 1):
                if self.mask[i, j]:
                    exact_values[i, j] = f"{U_ref[i, j]:.5f}"
        self.create_table_window("Точное решение u*(x, y)", exact_values)

    def show_table_error(self):
        if not hasattr(self, "U"):
            messagebox.showwarning("Нет данных", "Сначала нажмите 'Calculate'")
            return
        U_ref = u_exact(self.X, self.Y)
        n, m = self.X.shape[0] - 1, self.X.shape[1] - 1
        error_values = np.full_like(self.U, "", dtype=object)
        for i in range(n + 1):
            for j in range(m + 1):
                if self.mask[i, j]:
                    error_values[i, j] = f"{U_ref[i, j] - self.U[i, j]:.2e}"
        self.create_table_window("Погрешность u*(x, y) - v(N)(x, y)", error_values)

    '''def show_result_window(self):
        if not hasattr(self, "X") or not hasattr(self, "U"):
            messagebox.showwarning("Нет данных", "Сначала нажмите 'Calculate'")
            return

        U_ref = u_exact(self.X, self.Y)
        n, m = self.X.shape[0] - 1, self.X.shape[1] - 1
        cols = ["yj\\xi"] + [f"x{i}" for i in range(n + 1)]

        # Новое окно
        win = tk.Toplevel(self.root)
        win.title("Результаты: Таблицы")
        win.geometry("1300x900")
        win.grid_columnconfigure(0, weight=1)

        def create_table(parent, title, data_matrix):
            frame = ttk.LabelFrame(parent, text=title, padding=5)
            frame.pack(fill="both", expand=True, padx=10, pady=10)

            container = ttk.Frame(frame)
            container.pack(fill="both", expand=True)

            tree = ttk.Treeview(container, show='headings')
            tree["columns"] = cols
            for col in cols:
                tree.heading(col, text=col)
                tree.column(col, width=80, anchor="center")

            for j in range(m + 1):
                row = [f"y{j}"]
                for i in range(n + 1):
                    row.append(data_matrix[i, j] if self.mask[i, j] else "")
                tree.insert("", "end", values=row)

            vsb = ttk.Scrollbar(container, orient="vertical", command=tree.yview)
            hsb = ttk.Scrollbar(container, orient="horizontal", command=tree.xview)
            tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)

            tree.grid(row=0, column=0, sticky="nsew")
            vsb.grid(row=0, column=1, sticky="ns")
            hsb.grid(row=1, column=0, sticky="ew")
            container.grid_rowconfigure(0, weight=1)
            container.grid_columnconfigure(0, weight=1)

        # --- Создание таблиц ---
        # Численное решение
        V_values = np.full_like(self.U, "", dtype=object)
        for i in range(n + 1):
            for j in range(m + 1):
                if self.mask[i, j]:
                    V_values[i, j] = f"{self.U[i, j]:.5f}"
        create_table(win, "v(N)(x, y) — численное решение", V_values)

        # Погрешность
        err_values = np.full_like(self.U, "", dtype=object)
        for i in range(n + 1):
            for j in range(m + 1):
                if self.mask[i, j]:
                    err_values[i, j] = f"{(U_ref[i, j] - self.U[i, j]):.2e}"
        create_table(win, "u*(x, y) - v(N)(x, y) — погрешность", err_values)

        # Точное решение
        exact_values = np.full_like(self.U, "", dtype=object)
        for i in range(n + 1):
            for j in range(m + 1):
                if self.mask[i, j]:
                    exact_values[i, j] = f"{U_ref[i, j]:.5f}"
        create_table(win, "u*(x, y) — точное решение", exact_values)'''

    def calculate(self):
        try:
            n = int(self.n_entry.get())
            m = int(self.m_entry.get())
            eps = float(self.eps_entry.get())
            max_iter = int(self.max_iter_entry.get())
            # omega = float(self.omega_entry.get())
            if self.use_optimal_w.get():
                omega = optimal_w(n, m)
                self.omega_entry.delete(0, tk.END)
                self.omega_entry.insert(0, f"{omega:.5f}")
            else:
                omega = float(self.omega_entry.get())
            # Численное решение
            self.X, self.Y, self.U, self.mask, S, eps_max, max_residual, max_residual_euql, max_initial_residual, max_initial_residual_euql = sor_method(
                n, m, eps, max_iter, omega)

            # Точное решение и погрешность
            U_ref = u_exact(self.X, self.Y)
            error = np.abs(self.U - U_ref)
            # max_error = np.max(error[self.mask])
            # max_error = abs(np.max(self.U[self.mask]) - np.max(U_ref[self.mask]))
            max_error = np.max(abs(self.U[self.mask] - (U_ref[self.mask])))
            max_idx = np.unravel_index(np.argmax(error * self.mask), error.shape)
            max_x = self.X[max_idx]
            max_y = self.Y[max_idx]

            print("Максимум численного решения:", np.max(self.U[self.mask]))
            print("Максимум точного решения:", np.max(U_ref[self.mask]))
            print("Максимум разности:", max_error, " в точке:", max_x, ' ', max_y)
            print("Точность:", eps_max)
            print("Невязка:", max_residual)

            self.status_label.config(
                text=f"✅ Итерации: {S} \n| Достигнутая точность ε: {eps_max:.2e} \n| Погрешность max |u* − v(N)|: {max_error:.2e} в точке (x,y): ({max_x} , {max_y}) \n| Невязка: {max_residual:.2e} \n| Невязка евклидова: {max_residual_euql:.2e} \n| Невязка на 0 приближении: {max_initial_residual:.2e}\n| Невязка евклидова на 0 приближении: {max_initial_residual_euql:.2e}",
                foreground="green"
            )

        except Exception as e:
            messagebox.showerror("Ошибка", str(e))

    def show_2d(self):
        if hasattr(self, "U"):
            plt.figure(figsize=(10, 6))
            plt.contourf(self.X, self.Y, np.where(self.mask, self.U, np.nan), levels=50, cmap='plasma')
            # plt.contourf(self.X, self.Y, self.U, levels=50, cmap='plasma')
            plt.colorbar(label="u(x, y)")
            plt.title("2D график: численное решение")
            plt.xlabel("x")
            plt.ylabel("y")
            plt.show()
        else:
            messagebox.showwarning("Нет данных", "Сначала нажмите 'Calculate'")

    def show_3d(self):
        if hasattr(self, "U"):
            fig = plt.figure(figsize=(10, 7))
            ax = fig.add_subplot(111, projection='3d')
            # U_plot = np.where(self.mask, self.U, np.nan)
            ax.plot_surface(self.X, self.Y, np.where(self.mask, self.U, np.nan), cmap='plasma', edgecolor='none')
            # ax.plot_surface(self.X, self.Y, self.U, cmap='plasma', edgecolor='none')
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_zlabel("u(x, y)")
            ax.set_title("3D график: численное решение")
            plt.show()
        else:
            messagebox.showwarning("Нет данных", "Сначала нажмите 'Calculate'")


# Запуск
if __name__ == "__main__":
    root = tk.Tk()
    app = SORApp(root)
    root.mainloop()
