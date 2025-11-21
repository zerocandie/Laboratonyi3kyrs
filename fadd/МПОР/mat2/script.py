import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import os
import time
import sys
from pathlib import Path


DISPLAY_MODE = 'interactive' 
if DISPLAY_MODE == 'save_only':
    matplotlib.use('Agg')  
else:
    try:
        matplotlib.use('TkAgg')  
    except:
        try:
            matplotlib.use('Qt5Agg')  
        except:
            matplotlib.use('Agg')  
            DISPLAY_MODE = 'save_only'
            print("⚠️  GUI недоступен, переключаюсь на режим save_only")

VARIANT = 2

X0_INIT = 1
Y0_INIT = 2

# для таблиц
EPSILONS = [1e-3, 1e-4, 1e-5, 1e-6, 1e-7]

EPSILON_FOR_PLOTS = 1e-5  # для графика

PARAMS = {
    'A1': 1, 'A2': 3,
    'a1': 2, 'a2': 1,
    'b1': 3, 'b2': 1,
    'c1': 2, 'c2': 1,
    'd1': 3, 'd2': 2
}

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

print(f"Директория результатов: {RESULTS_DIR}")
print(f"Режим отображения: {DISPLAY_MODE}")
print(f"Начальная точка: ({X0_INIT}, {Y0_INIT})")
print(f"Эпсилон для графиков: {EPSILON_FOR_PLOTS}\n")

def build_target_function(params):
    """Построение целевой функции в зависимости от типа варианта"""
    def f(x, y):
        if VARIANT % 2 == 0:  
            term1 = params['A1'] / (1 + ((x - params['a1']) / params['b1'])**2 + 
                                      ((y - params['c1']) / params['d1'])**2)
            term2 = params['A2'] / (1 + ((x - params['a2']) / params['b2'])**2 + 
                                      ((y - params['c2']) / params['d2'])**2)
            return term1 + term2
        else:  
            term1 = params['A1'] * (((x - params['a1']) / params['b1'])**2 + 
                                    ((y - params['c1']) / params['d1'])**2)
            term2 = params['A2'] * (((x - params['a2']) / params['b2'])**2 + 
                                    ((y - params['c2']) / params['d2'])**2)
            return term1 + term2
    return f

def build_objective_function(params):
    """Построение функции для минимизации (для четных вариантов ищем -f)"""
    f = build_target_function(params)
    return (lambda x, y: -f(x, y)) if VARIANT % 2 == 0 else f

def build_gradient(params):
    """Построение градиента функции"""
    if VARIANT % 2 == 0:  
        def grad(x, y):
            A1, A2 = params['A1'], params['A2']
            a1, a2 = params['a1'], params['a2']
            b1, b2 = params['b1'], params['b2']
            c1, c2 = params['c1'], params['c2']
            d1, d2 = params['d1'], params['d2']
            
            D1 = 1 + ((x - a1) / b1)**2 + ((y - c1) / d1)**2
            D2 = 1 + ((x - a2) / b2)**2 + ((y - c2) / d2)**2
            
            df_dx = A1 * (-2 * (x - a1) / (b1**2)) / (D1**2) + \
                    A2 * (-2 * (x - a2) / (b2**2)) / (D2**2)
            df_dy = A1 * (-2 * (y - c1) / (d1**2)) / (D1**2) + \
                    A2 * (-2 * (y - c2) / (d2**2)) / (D2**2)
            
            return np.array([-df_dx, -df_dy])
    else:  
        def grad(x, y):
            A1, A2 = params['A1'], params['A2']
            a1, a2 = params['a1'], params['a2']
            b1, b2 = params['b1'], params['b2']
            c1, c2 = params['c1'], params['c2']
            d1, d2 = params['d1'], params['d2']
            
            df_dx = 2 * A1 * (x - a1) / (b1**2) + 2 * A2 * (x - a2) / (b2**2)
            df_dy = 2 * A1 * (y - c1) / (d1**2) + 2 * A2 * (y - c2) / (d2**2)
            return np.array([df_dx, df_dy])
    
    return grad

def golden_section_minimize(func, a, b, tol=1e-8, max_iter=1000):
    """Метод золотого сечения для минимизации функции одной переменной"""
    phi = (1 + np.sqrt(5)) / 2
    resphi = 2 - phi
    
    x1 = a + resphi * (b - a)
    x2 = b - resphi * (b - a)
    f1 = func(x1)
    f2 = func(x2)
    
    for iteration in range(max_iter):
        if abs(b - a) < tol:
            break
        
        if f1 < f2:
            b = x2
            f2 = f1
            x2 = x1
            x1 = a + resphi * (b - a)
            f1 = func(x1)
        else:
            a = x1
            f1 = f2
            x1 = x2
            x2 = b - resphi * (b - a)
            f2 = func(x2)
    
    return (a + b) / 2

def hooke_jeeves(func, x0, y0, step_size=1.0, shrink_factor=0.5, tol=1e-6, max_iter=1000):
    """Метод Хука-Дживса"""
    trajectory = []
    x, y = float(x0), float(y0)
    current_f = func(x, y)
    trajectory.append((x, y, current_f, 0))
    
    step = step_size
    iter_count = 0
    last_improvement = float('inf')
    
    while step > tol and iter_count < max_iter:
        best_x, best_y, best_f = x, y, current_f
        improved = False
        
        for dx, dy in [(step, 0), (-step, 0), (0, step), (0, -step)]:
            nx, ny = x + dx, y + dy
            nf = func(nx, ny)
            
            if nf < best_f:
                best_f = nf
                best_x = nx
                best_y = ny
                improved = True
        
        if improved:
            x, y = best_x, best_y
            improvement = current_f - best_f
            current_f = best_f
            last_improvement = improvement
            iter_count += 1
            trajectory.append((x, y, current_f, iter_count))
        else:
            step *= shrink_factor
            if last_improvement < tol * 10:
                break
    
    return trajectory, iter_count, iter_count < max_iter

def steepest_descent(func, grad_func, x0, y0, tol=1e-6, max_iter=1000):
    """Метод наискорейшего спуска"""
    trajectory = []
    x, y = float(x0), float(y0)
    current_f = func(x, y)
    trajectory.append((x, y, current_f, 0))
    
    iter_count = 0
    last_f_improvement = float('inf')
    
    while iter_count < max_iter:
        grad = grad_func(x, y)
        grad_norm = np.linalg.norm(grad)
        
        if grad_norm < tol:
            break
        if last_f_improvement < tol * 10 and iter_count > 50:
            break
        
        direction = -grad / (grad_norm + 1e-10)
        
        def line_func(alpha):
            return func(x + alpha * direction[0], y + alpha * direction[1])
        
        a, b = 0.0, 1.0
        max_expand = 50
        expand_count = 0
        
        while expand_count < max_expand and line_func(b) < line_func(a):
            b *= 2
            expand_count += 1
        
        if b <= a:
            b = a + 1.0
        
        try:
            alpha_opt = golden_section_minimize(line_func, a, b, tol=tol)
        except:
            alpha_opt = a
        
        alpha_opt = min(alpha_opt, 10.0)
        
        x_new = x + alpha_opt * direction[0]
        y_new = y + alpha_opt * direction[1]
        f_new = func(x_new, y_new)
        
        f_improvement = current_f - f_new
        if f_improvement > 0:
            last_f_improvement = f_improvement
            x, y = x_new, y_new
            current_f = f_new
        else:
            break
        
        iter_count += 1
        trajectory.append((x, y, current_f, iter_count))
    
    return trajectory, iter_count, iter_count < max_iter

def rosenbrock_func(x, y):
    """Классическая функция Розенброка"""
    return 100 * (y - x**2)**2 + (1 - x)**2

def rosenbrock_grad(x, y):
    """Градиент функции Розенброка"""
    df_dx = -400 * x * (y - x**2) - 2 * (1 - x)
    df_dy = 200 * (y - x**2)
    return np.array([df_dx, df_dy])

def plot_trajectories_simple(func, trajectories, title="Траектории", 
                            filename="trajectory_simple.png", x_range=None, y_range=None,
                            show_plot=True):
    """
    Упрощённое построение графиков: только контуры и траектории без тепловой карты.
    """
    if x_range is None or y_range is None:
        if "Розенброка" in title or "Rosenbrock" in title:
            x_range = (-1.5, 2.0)
            y_range = (-0.5, 3.0)
        else:
            x_range = (-2, 6)
            y_range = (-2, 6)

    # Уменьшаем разрешение для ускорения
    x_vals = np.linspace(x_range[0], x_range[1], 100)
    y_vals = np.linspace(y_range[0], y_range[1], 100)
    X, Y = np.meshgrid(x_vals, y_vals)
    Z = np.array([[func(x, y) for x in x_vals] for y in y_vals])

    plt.figure(figsize=(8, 6))

    # Только линии уровня
    plt.contour(X, Y, Z, levels=15, colors='gray', linewidths=0.7, alpha=0.7)

    # Траектории
    colors = ['red', 'blue']
    labels = ['Hooke-Jeeves', 'Steepest Descent']
    for idx, traj in enumerate(trajectories):
        if not traj:
            continue
        xs = [p[0] for p in traj]
        ys = [p[1] for p in traj]
        plt.plot(xs, ys, color=colors[idx % len(colors)], 
                marker='o', markersize=3, linewidth=1.5,
                label=labels[idx] if idx < len(labels) else f'Метод {idx}')
        # Начало — зелёный круг
        plt.plot(xs[0], ys[0], 'go', markersize=8, label='Старт' if idx == 0 else "")
        # Конец — чёрный крест
        plt.plot(xs[-1], ys[-1], 'kx', markersize=10, markeredgewidth=2, label='Финиш' if idx == 0 else "")

    plt.title(title, fontsize=12, fontweight='bold')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend(fontsize=9)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()

    if DISPLAY_MODE in ['save_only', 'interactive']:
        filepath = os.path.join(RESULTS_DIR, filename)
        plt.savefig(filepath, dpi=120, bbox_inches='tight')
        print(f"  💾 Простой график сохранён: {filename}")

    if DISPLAY_MODE in ['interactive', 'show_only'] and show_plot:
        print(f"  🖥️  Отображаю простой график: {title}")
        plt.show()
    else:
        plt.close()

def save_results_to_excel(all_sheets, summary_data, filename="lab2_results.xlsx"):
    """Сохранение результатов в Excel или CSV"""
    excel_path = os.path.join(RESULTS_DIR, filename)
    
    try:
        import openpyxl
        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_excel(writer, sheet_name='Summary', index=False)
            
            for sheet_name, df in all_sheets.items():
                sheet_name_safe = sheet_name[:31]
                df.to_excel(writer, sheet_name=sheet_name_safe, index=False)
        
        print(f"✅ Результаты сохранены в Excel: {filename}")
        return excel_path
    
    except ImportError:
        try:
            with pd.ExcelWriter(excel_path, engine='xlsxwriter') as writer:
                summary_df = pd.DataFrame(summary_data)
                summary_df.to_excel(writer, sheet_name='Summary', index=False)
                
                for sheet_name, df in all_sheets.items():
                    sheet_name_safe = sheet_name[:31]
                    df.to_excel(writer, sheet_name=sheet_name_safe, index=False)
            
            print(f"✅ Результаты сохранены в Excel (xlsxwriter): {filename}")
            return excel_path
        
        except ImportError:
            print("⚠️  Excel-движки недоступны. Сохраняю в CSV...")
            
            summary_df = pd.DataFrame(summary_data)
            csv_path = os.path.join(RESULTS_DIR, filename.replace('.xlsx', '_summary.csv'))
            summary_df.to_csv(csv_path, index=False)
            print(f"✅ Сводка сохранена в CSV: {os.path.basename(csv_path)}")
            
            for i, (sheet_name, df) in enumerate(all_sheets.items()):
                csv_traj_path = os.path.join(RESULTS_DIR, f"trajectory_{i:02d}.csv")
                df.to_csv(csv_traj_path, index=False)
            
            print(f"✅ Траектории сохранены в CSV (всего {len(all_sheets)} файлов)")
            return csv_path

def main():
    print(f"\n🚀 ЛАБОРАТОРНАЯ РАБОТА №2: МЕТОДЫ СПУСКА")
    print(f"📋 Вариант: {VARIANT}\n")
    
    user_func = build_objective_function(PARAMS)
    user_grad = build_gradient(PARAMS)
    x0, y0 = X0_INIT, Y0_INIT
    
    test_functions = [
        ("Rosenbrock", rosenbrock_func, rosenbrock_grad),
        ("UserFunction", user_func, user_grad)
    ]
    
    all_sheets = {}
    summary_data = []
    
    print("🔄 Выполняю тестирование методов...\n")
    
    for func_name, func, grad in test_functions:
        print(f"📌 Функция: {func_name}")
        
        for eps in EPSILONS:
            print(f"  🔍 Точность ε = {eps:.0e}")
            
            start_time = time.time()
            traj_hj, iters_hj, ok_hj = hooke_jeeves(func, x0, y0, tol=eps)
            time_hj = time.time() - start_time
            
            start_time = time.time()
            traj_sd, iters_sd, ok_sd = steepest_descent(func, grad, x0, y0, tol=eps)
            time_sd = time.time() - start_time
            
            sheet_hj = f"{func_name}_HJ_{eps:.0e}"
            sheet_sd = f"{func_name}_SD_{eps:.0e}"
            
            all_sheets[sheet_hj] = pd.DataFrame(traj_hj, columns=['x', 'y', 'f(x,y)', 'iteration'])
            all_sheets[sheet_sd] = pd.DataFrame(traj_sd, columns=['x', 'y', 'f(x,y)', 'iteration'])
            
            summary_data.extend([
                {
                    'Function': func_name,
                    'Algorithm': 'Hooke-Jeeves',
                    'Epsilon': eps,
                    'Iterations': iters_hj,
                    'Final_x': traj_hj[-1][0],
                    'Final_y': traj_hj[-1][1],
                    'Final_f': traj_hj[-1][2],
                    'Time_sec': time_hj,
                    'Success': ok_hj
                },
                {
                    'Function': func_name,
                    'Algorithm': 'Steepest Descent',
                    'Epsilon': eps,
                    'Iterations': iters_sd,
                    'Final_x': traj_sd[-1][0],
                    'Final_y': traj_sd[-1][1],
                    'Final_f': traj_sd[-1][2],
                    'Time_sec': time_sd,
                    'Success': ok_sd
                }
            ])
    
    print("\n💾 Сохраняю результаты...\n")
    save_results_to_excel(all_sheets, summary_data)
    
    print("\n📈 Создаю УПРОЩЁННЫЕ графики траекторий...\n")
    
    # 1. Пользовательская функция
    print("📌 Функция варианта 2:")
    traj_hj_user, _, _ = hooke_jeeves(user_func, x0, y0, tol=EPSILON_FOR_PLOTS)
    traj_sd_user, _, _ = steepest_descent(user_func, user_grad, x0, y0, tol=EPSILON_FOR_PLOTS)
    
    plot_trajectories_simple(
        user_func,
        [traj_hj_user, traj_sd_user],
        title=f"Вариант {VARIANT}, ε={EPSILON_FOR_PLOTS}",
        filename="user_function_simple.png"
    )
    
    # 2. Розенброка — оба метода
    print("\n📌 Функция Розенброка:")
    traj_hj_rosenbrock, _, _ = hooke_jeeves(rosenbrock_func, x0, y0, tol=EPSILON_FOR_PLOTS)
    traj_sd_rosenbrock, _, _ = steepest_descent(rosenbrock_func, rosenbrock_grad, x0, y0, tol=EPSILON_FOR_PLOTS)
    
    plot_trajectories_simple(
        rosenbrock_func,
        [traj_hj_rosenbrock, traj_sd_rosenbrock],
        title=f"Розенброка, ε={EPSILON_FOR_PLOTS}",
        filename="rosenbrock_simple.png"
    )
    
    print(f"\n📁 Результаты сохранены в: {RESULTS_DIR}/")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ ОШИБКА: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)