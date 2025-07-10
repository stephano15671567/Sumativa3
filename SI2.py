import json
import random
import math
import numpy as np
import matplotlib.pyplot as plt

# --- CONSTANTES DEL PROBLEMA ---
# Rangos para las 5 variables de valorización (Tarde, Noche, Diario, Revistas, Radio)
VALORIZATION_RANGES = [
    (65, 85),  # v1: Tarde
    (90, 95),  # v2: Noche
    (40, 60),  # v3: Diario
    (60, 80),  # v4: Revistas
    (20, 30)   # v5: Radio
]

# Mapeo de nombres a índices para fácil acceso
varnames = ['Tarde', 'Noche', 'Diario', 'Revistas', 'Radio']

# --- Cargar dominios reducidos por AC-3 para las cantidades ---
try:
    with open("ac3_domains.json", "r") as f:
        ac3_domains = json.load(f)
except FileNotFoundError:
    print("Error: El archivo 'ac3_domains.json' no se encontró.")
    print("Por favor, ejecuta primero el script de AC-3 para generarlo.")
    exit()

# --- DEFINICIÓN DEL PROBLEMA (MODIFICADO) ---
class Problem:
    """
    Define el problema de optimización de 10 dimensiones.
    Las variables 0-4 son cantidades (discretas).
    Las variables 5-9 son valorizaciones (continuas).
    """
    def __init__(self):
        self.dim = 10
        self.max_values = [15, 10, 25, 4, 30] # Límites de cantidad

    def get_costs(self, valorizaciones):
        """Calcula los costos dinámicamente a partir de las valorizaciones."""
        v1, v2, v3, v4, v5 = valorizaciones
        c1 = 2 * v1 + 30
        c2 = 10 * v2 - 600
        c3 = 2 * v3 - 40
        c4 = v4 + 40
        c5 = v5 - 1
        return [c1, c2, c3, c4, c5]

    def check(self, x):
        """Verifica si una solución de 10D es factible."""
        cantidades = x[:5]
        valorizaciones = x[5:]
        costos = self.get_costs(valorizaciones)

        # Restricciones de presupuesto (calculadas al vuelo)
        if costos[0]*cantidades[0] + costos[1]*cantidades[1] > 3800: return False
        if costos[2]*cantidades[2] + costos[3]*cantidades[3] > 2800: return False
        if costos[2]*cantidades[2] + costos[4]*cantidades[4] > 3500: return False
        
        # Restricciones de dominio para las cantidades
        for i, val in enumerate(cantidades):
            if val not in ac3_domains[varnames[i]]:
                return False
        
        # Restricciones de rango para las valorizaciones
        for i, val in enumerate(valorizaciones):
            min_v, max_v = VALORIZATION_RANGES[i]
            if not (min_v <= val <= max_v):
                return False
        return True

    def objectives(self, x):
        """Calcula los dos objetivos para una solución de 10D."""
        cantidades = x[:5]
        valorizaciones = x[5:]
        costos = self.get_costs(valorizaciones)

        valorizacion_total = sum(v * q for v, q in zip(valorizaciones, cantidades))
        costo_total = sum(c * q for c, q in zip(costos, cantidades))

        # Devuelve [-valorizacion] para convertir maximización en minimización
        return [-valorizacion_total, costo_total]

# --- ALGORITMO COA ---
class Coati:
    """Representa una solución candidata (un coatí) de 10 dimensiones."""
    def __init__(self, problem):
        self.p = problem
        self.dimension = self.p.dim
        self.x = [0.0] * self.dimension

        # Inicializar las primeras 5 variables (cantidades discretas)
        for i in range(5):
            self.x[i] = random.choice(ac3_domains[varnames[i]])

        # Inicializar las últimas 5 variables (valorizaciones continuas)
        for i in range(5):
            min_v, max_v = VALORIZATION_RANGES[i]
            self.x[i + 5] = random.uniform(min_v, max_v)

        self.obj = self.p.objectives(self.x)

    def is_feasible(self):
        return self.p.check(self.x)

    def move_phase1_climber(self, iguana_lider):
        """Movimiento para escaladores, adaptado para 10D (discreto + continuo)."""
        I = random.choice([1, 2])
        for j in range(self.dimension):
            r = random.random()
            move_vector = r * (iguana_lider.x[j] - I * self.x[j])
            new_val_float = self.x[j] + move_vector
            
            if j < 5:  # Variable de CANTIDAD (discreta)
                domain = ac3_domains[varnames[j]]
                self.x[j] = min(domain, key=lambda v: abs(v - new_val_float))
            else:  # Variable de VALORIZACIÓN (continua)
                min_v, max_v = VALORIZATION_RANGES[j - 5]
                self.x[j] = max(min_v, min(new_val_float, max_v)) # Clamping
                
        self.obj = self.p.objectives(self.x)

    def move_phase1_ground(self, iguana_g, current_obj):
        """Movimiento para coatíes de tierra, adaptado para 10D."""
        I = random.choice([1, 2])
        
        # Decide si moverse hacia o lejos de la iguana_g
        move_towards = self.dominates(iguana_g.obj, current_obj)
        
        for j in range(self.dimension):
            r = random.random()
            if move_towards:
                move_vector = r * (iguana_g.x[j] - I * self.x[j])
            else:
                move_vector = r * (self.x[j] - iguana_g.x[j])
            
            new_val_float = self.x[j] + move_vector

            if j < 5: # Variable de CANTIDAD (discreta)
                domain = ac3_domains[varnames[j]]
                self.x[j] = min(domain, key=lambda v: abs(v - new_val_float))
            else: # Variable de VALORIZACIÓN (continua)
                min_v, max_v = VALORIZATION_RANGES[j - 5]
                self.x[j] = max(min_v, min(new_val_float, max_v))

        self.obj = self.p.objectives(self.x)

    def move_phase2(self, t, max_iter):
        """Fase 2 (Explotación), adaptada para 10D."""
        for j in range(self.dimension):
            if j < 5: # Movimiento discreto basado en índice para CANTIDADES
                domain = ac3_domains[varnames[j]]
                local_range_factor = (1 - (t / max_iter))
                max_step = math.ceil(len(domain) * local_range_factor * 0.5)
                step = random.randint(-max_step, max_step) if max_step > 0 else 0
                
                current_index = domain.index(self.x[j])
                new_index = max(0, min(len(domain) - 1, current_index + step))
                self.x[j] = domain[new_index]
            else: # Movimiento continuo para VALORIZACIONES
                min_v, max_v = VALORIZATION_RANGES[j - 5]
                local_range = (max_v - min_v) * (1 - (t / max_iter)) / 2
                new_val = self.x[j] + random.uniform(-local_range, local_range)
                self.x[j] = max(min_v, min(new_val, max_v))

        self.obj = self.p.objectives(self.x)
        
    @staticmethod
    def dominates(obj_a, obj_b):
        all_le = all(a <= b for a, b in zip(obj_a, obj_b))
        any_lt = any(a < b for a, b in zip(obj_a, obj_b))
        return all_le and any_lt

    def copy(self, other):
        if isinstance(other, Coati):
            self.x = other.x[:]
            self.obj = other.obj[:]


class COA:
    """Clase principal del algoritmo. Funciona con el problema de 10D."""
    def __init__(self, problem, n_coatis=30, max_iter=100):
        self.p = problem
        self.n_coatis = n_coatis
        self.max_iter = max_iter
        self.population = []
        self.pareto_archive = []
        self.history = {'pareto_size': [], 'best_costo': [], 'best_valorizacion': []}

    def update_archive(self, new_coati):
        is_dominated_by_archive = any(Coati.dominates(sol.obj, new_coati.obj) for sol in self.pareto_archive)
        if is_dominated_by_archive:
            return
        self.pareto_archive = [sol for sol in self.pareto_archive if not Coati.dominates(new_coati.obj, sol.obj)]
        self.pareto_archive.append(new_coati)

    def initialize_population(self):
        while len(self.population) < self.n_coatis:
            ind = Coati(self.p)
            if ind.is_feasible():
                self.population.append(ind)
        for ind in self.population:
            self.update_archive(ind)

    def evolve(self):
        t = 1
        while t <= self.max_iter:
            if not self.pareto_archive:
                print("El archivo de Pareto está vacío. Deteniendo la evolución.")
                break
            
            iguana_lider = random.choice(self.pareto_archive)
            
            # Fases de movimiento
            for i in range(self.n_coatis):
                original_coati = self.population[i]
                coati_copy = Coati(self.p)
                coati_copy.copy(original_coati)
                
                # Fase 1
                if i < self.n_coatis / 2:
                    coati_copy.move_phase1_climber(iguana_lider)
                else:
                    iguana_g = Coati(self.p) # Iguana de referencia aleatoria
                    coati_copy.move_phase1_ground(iguana_g, original_coati.obj)
                
                if coati_copy.is_feasible() and Coati.dominates(coati_copy.obj, original_coati.obj):
                    self.population[i].copy(coati_copy)
                    self.update_archive(self.population[i])
                
                # Fase 2
                coati_copy.copy(self.population[i]) # Empezar desde la posición actual
                coati_copy.move_phase2(t, self.max_iter)

                if coati_copy.is_feasible() and Coati.dominates(coati_copy.obj, self.population[i].obj):
                    self.population[i].copy(coati_copy)
                    self.update_archive(self.population[i])

            self.record_history()
            if t % 50 == 0: # Imprimir progreso cada 50 iteraciones
                 print(f"Iteración {t}/{self.max_iter} - Tamaño del Frente: {len(self.pareto_archive)}")
            t += 1
            
    def record_history(self):
        if not self.pareto_archive: return
        costs = [s.obj[1] for s in self.pareto_archive]
        valorizaciones = [-s.obj[0] for s in self.pareto_archive]
        self.history['pareto_size'].append(len(self.pareto_archive))
        self.history['best_costo'].append(min(costs))
        self.history['best_valorizacion'].append(max(valorizaciones))

    def optimizer(self):
        self.initialize_population()
        print("--- Población Inicial y Frente de Pareto Inicial ---")
        print(f"Tamaño del Frente de Pareto: {len(self.pareto_archive)}")
        self.record_history()
        self.evolve()

def analyze_and_plot_results(algorithm, escenario_id):
    if not algorithm.pareto_archive:
        print("No hay resultados para analizar.")
        return

    final_solutions = sorted(algorithm.pareto_archive, key=lambda s: s.obj[1])
    costs = np.array([s.obj[1] for s in final_solutions])
    valorizaciones = np.array([-s.obj[0] for s in final_solutions])

    # --- INICIO DE LA MODIFICACIÓN ---
    # 1. Preparar los datos para la tabla
    stats_data = [
        [f"{np.min(costs):.2f}", f"{np.max(valorizaciones):.2f}"],
        [f"{np.max(costs):.2f}", f"{np.min(valorizaciones):.2f}"],
        [f"{np.mean(costs):.2f}", f"{np.mean(valorizaciones):.2f}"],
        [f"{np.median(costs):.2f}", f"{np.median(valorizaciones):.2f}"],
        [f"{np.std(costs):.2f}", f"{np.std(valorizaciones):.2f}"]
    ]
    row_labels = ['Mejor', 'Peor', 'Promedio', 'Mediana', 'Desv. Est.']
    col_labels = ['Costo', 'Valorización']

    # 2. Crear la figura y la tabla
    fig, ax = plt.subplots(figsize=(6, 2.5)) # Ajustar tamaño según sea necesario
    ax.axis('tight')
    ax.axis('off')
    
    table = ax.table(cellText=stats_data,
                     rowLabels=row_labels,
                     colLabels=col_labels,
                     loc='center',
                     cellLoc='center')
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.2) # Ajustar escala de la tabla

    # 3. Añadir título y guardar la imagen
    plt.title(f'Resumen Estadístico - Optimización Global', fontsize=12, y=1.05)
    fig.tight_layout()
    stats_filename = f'stats_summary_{escenario_id}.png'
    plt.savefig(stats_filename, bbox_inches='tight', dpi=150)
    plt.close()
    
    # 4. Informar al usuario
    print(f"\n-> Resumen estadístico guardado como: {stats_filename}")
    # --- FIN DE LA MODIFICACIÓN ---

    # El resto de la función para generar los otros gráficos permanece igual
    plt.figure(figsize=(10, 7))
    plt.scatter(costs, valorizaciones, c=valorizaciones, cmap='viridis', alpha=0.7)
    plt.title(f'Frente de Pareto - Optimización Global', fontsize=16)
    plt.xlabel('Costo Total', fontsize=12)
    plt.ylabel('Valorización Total', fontsize=12)
    plt.colorbar(label='Nivel de Valorización')
    plt.grid(True)
    plt.savefig(f'pareto_front_{escenario_id}.png')
    plt.close()
    print(f"-> Gráfico del Frente de Pareto guardado como: pareto_front_{escenario_id}.png")

    iterations = range(1, len(algorithm.history['pareto_size']) + 1)
    if not iterations: return

    # Gráfico de convergencia del tamaño del frente
    plt.figure(figsize=(10, 7))
    plt.plot(iterations, algorithm.history['pareto_size'], marker='.', linestyle='-', color='green')
    plt.title(f'Convergencia del Tamaño del Frente de Pareto')
    plt.xlabel('Iteración')
    plt.ylabel('N° Soluciones no Dominadas')
    plt.grid(True)
    plt.savefig(f'pareto_size_convergence_{escenario_id}.png')
    plt.close()
    print(f"-> Gráfico de convergencia del frente guardado como: pareto_size_convergence_{escenario_id}.png")



# --- EJECUCIÓN (MODIFICADA) ---
if __name__ == "__main__":
    print("\n" + "="*60)
    print("INICIANDO OPTIMIZACIÓN INTEGRAL (10 VARIABLES)")
    print("="*60)

    problem_instance = Problem()
    
    # Aumentamos los parámetros para un espacio de búsqueda más grande y complejo
    algorithm = COA(problem=problem_instance, n_coatis=100, max_iter=1500)
    algorithm.optimizer()

    if not algorithm.pareto_archive:
        print("No se encontraron soluciones en el frente de Pareto.")
    else:
        print("\n" + "="*60)
        print("--- MEJORES SOLUCIONES GLOBALES ENCONTRADAS (FRENTE DE PARETO) ---")
        print("="*60)
        final_solutions_raw = sorted(algorithm.pareto_archive, key=lambda s: s.obj[1])
        unique_solutions = []
        seen_objectives = set()

        for sol in final_solutions_raw:
            # Creamos una tupla de los objetivos para poder añadirla a un 'set'
            objectives_tuple = tuple(sol.obj)
            if objectives_tuple not in seen_objectives:
                unique_solutions.append(sol)
                seen_objectives.add(objectives_tuple)

        print(f"\nSe encontraron {len(unique_solutions)} soluciones únicas en el Frente de Pareto.")

        for i, sol in enumerate(unique_solutions):
            cantidades = [int(v) for v in sol.x[:5]]
            valorizaciones = [round(v, 2) for v in sol.x[5:]]
            print(f"\n  Solución {i+1}:")
            print(f"    - Cantidades     (x1-x5): {cantidades}")
            print(f"    - Valorizaciones (v1-v5): {valorizaciones}")
            print(f"    - ==> Valorización Total: {int(-sol.obj[0])}, Costo Total: {int(sol.obj[1])}")

    analyze_and_plot_results(algorithm, escenario_id="global")

