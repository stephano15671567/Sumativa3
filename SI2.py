import json
import random
import math

# --- Algoritmo COA (Coatí Optimization Algorithm) ---
# adaptada para un problema multiobjetivo y discreto.

# Cargar dominios reducidos por AC-3
try:
    with open("ac3_domains.json", "r") as f:
        ac3_domains = json.load(f)
except FileNotFoundError:
    print("Error: El archivo 'ac3_domains.json' no se encontró.")
    print("Por favor, ejecuta primero 'Ejemplo-AC3v2.py' para generarlo.")
    exit()


# Mapeo de nombres a índices para fácil acceso
varnames = ['Tarde', 'Noche', 'Diario', 'Revistas', 'Radio']

class Problem:
    """
    Define el problema de optimización, sus restricciones y objetivos.
    """
    def __init__(self):
        self.dim = 5
        self.costos = [160, 300, 40, 100, 10]
        self.valorizaciones = [65, 90, 40, 60, 20]
        self.max_values = [15, 10, 25, 4, 30]

    def check(self, x):
        """
        Verifica si una solución 'x' cumple con todas las restricciones.
        """
        # Restricciones de presupuesto
        if 160*x[0] + 300*x[1] > 3800: return False
        if 40*x[2] + 100*x[3] > 2800: return False
        if 40*x[2] + 10*x[4] > 3500: return False
        
        # Restricciones de dominio (verificando que los valores están en los dominios de AC-3)
        for i, val in enumerate(x):
            if val not in ac3_domains[varnames[i]]:
                return False
        return True

    def objectives(self, x):
        """
        Calcula los dos objetivos del problema.
        - Se retorna la negación de la valorización para que ambos objetivos sean de minimización.
        """
        valorizacion = sum(v * x[i] for i, v in enumerate(self.valorizaciones))
        costo = sum(c * x[i] for i, c in enumerate(self.costos))
        return [-valorizacion, costo]

class Coati:
    """
    Representa una solución candidata (un coatí).
    """
    def __init__(self, problem):
        self.p = problem
        self.dimension = self.p.dim
        # Inicialización desde los dominios reducidos por AC-3
        self.x = [random.choice(ac3_domains[varnames[i]]) for i in range(self.dimension)]
        self.obj = self.p.objectives(self.x)

    def is_feasible(self):
        return self.p.check(self.x)


    # Se divide la Fase 1 en dos métodos, uno para cada grupo de coatíes
    # según lo describe el paper original.

    def move_phase1_climber(self, iguana_lider):
        """
        Movimiento para la primera mitad de la población (escaladores).
        Se mueven hacia el líder del frente de Pareto. Ecuación (4) del paper. [cite: 216]
        """
        I = random.choice([1, 2])
        for j in range(self.dimension):
            domain = ac3_domains[varnames[j]]
            r = random.random()
            
            move_vector = r * (iguana_lider.x[j] - I * self.x[j])
            new_val_float = self.x[j] + move_vector
            
            # Adaptación a espacio discreto: buscar el valor más cercano en el dominio
            self.x[j] = min(domain, key=lambda v: abs(v - new_val_float))
            
        self.obj = self.p.objectives(self.x)

    def move_phase1_ground(self, iguana_g, current_obj):
        """
        Movimiento para la segunda mitad de la población (de tierra).
        Su movimiento depende de si 'iguana_g' es mejor o peor. Ecuación (6) del paper. [cite: 229]
        """
        I = random.choice([1, 2])
        
        # En MOO, "mejor" significa que domina. Comparamos iguana_g con la solución original.
        if self.dominates(iguana_g.obj, current_obj):
            # Moverse hacia la iguana_g (es una buena posición)
            for j in range(self.dimension):
                domain = ac3_domains[varnames[j]]
                r = random.random()
                move_vector = r * (iguana_g.x[j] - I * self.x[j])
                new_val_float = self.x[j] + move_vector
                self.x[j] = min(domain, key=lambda v: abs(v - new_val_float))
        else:
            # Moverse alejándose de la iguana_g (es una mala posición)
            for j in range(self.dimension):
                domain = ac3_domains[varnames[j]]
                r = random.random()
                # La Ecuación (6) cambia la fórmula en este caso
                move_vector = r * (self.x[j] - iguana_g.x[j])
                new_val_float = self.x[j] + move_vector
                self.x[j] = min(domain, key=lambda v: abs(v - new_val_float))
                
        self.obj = self.p.objectives(self.x)

    def move_phase2(self, t, max_iter):
        """
        Fase 2 de COA: Escape de depredadores (Explotación).
        Adaptación de Ecuaciones (8) y (9) a un espacio discreto. [cite: 263, 265]
        """
        for j in range(self.dimension):
            domain = ac3_domains[varnames[j]]
            
            # Factor de rango local que disminuye con las iteraciones
            local_range_factor = (1 - (t / max_iter))
            # El paso máximo se reduce a medida que avanza el algoritmo
            max_step = math.ceil(len(domain) * local_range_factor * 0.5)
            
            if max_step == 0:
                step = 0
            else:
                step = random.randint(-max_step, max_step)
            
            # Moverse en el índice del dominio, no en el valor
            current_index_in_domain = domain.index(self.x[j])
            new_index = current_index_in_domain + step
            
            # Asegurar que el nuevo índice esté dentro de los límites del dominio
            new_index = max(0, min(len(domain) - 1, new_index))
            self.x[j] = domain[new_index]

        self.obj = self.p.objectives(self.x)
        
    @staticmethod
    def dominates(obj_a, obj_b):
        """
        Verifica si la solución A domina a la solución B.
        """
        all_le = all(a <= b for a, b in zip(obj_a, obj_b))
        any_lt = any(a < b for a, b in zip(obj_a, obj_b))
        return all_le and any_lt

    def copy(self, other):
        """
        Copia los atributos de otro individuo.
        """
        if isinstance(other, Coati):
            self.x = other.x[:]
            self.obj = other.obj[:]

    def __str__(self):
        valorizacion = -self.obj[0]
        costo = self.obj[1]
        return f"Sol: {self.x}, Valorización: {int(valorizacion)}, Costo: {costo}"

class COA:
    def __init__(self, problem, n_individual=30, max_iter=100):
        self.p = problem
        self.n_individual = n_individual
        self.max_iter = max_iter
        self.swarm = []
        self.pareto_archive = []

    def update_archive(self, new_individual):
        """
        Actualiza el archivo de Pareto con un nuevo individuo.
        """
        is_dominated_by_archive = False
        for sol in self.pareto_archive:
            if Coati.dominates(sol.obj, new_individual.obj):
                is_dominated_by_archive = True
                break
        if is_dominated_by_archive:
            return

        # Eliminar del archivo las soluciones que son dominadas por el nuevo individuo
        self.pareto_archive = [sol for sol in self.pareto_archive if not Coati.dominates(new_individual.obj, sol.obj)]
        # Añadir el nuevo individuo al archivo
        self.pareto_archive.append(new_individual)

    def initialize_population(self):
        """
        Inicializa la población de coatíes, asegurando que sean soluciones factibles.
        """
        for _ in range(self.n_individual):
            feasible = False
            while not feasible:
                ind = Coati(self.p)
                feasible = ind.is_feasible()
            self.swarm.append(ind)
        
        # Actualizar el archivo inicial de Pareto con la población inicial
        for ind in self.swarm:
            self.update_archive(ind)

    def evolve(self):
        """
        Ejecuta el ciclo de evolución del algoritmo.
        """
        t = 1
        while t <= self.max_iter:
            if not self.pareto_archive:
                print("El archivo de Pareto está vacío. Deteniendo la evolución.")
                break
            
            # Seleccionar un líder del archivo de Pareto para guiar la búsqueda
            iguana_lider = random.choice(self.pareto_archive)
            
            # --- Fase 1: Exploración (Caza) --- [cite: 205]
            for i in range(self.n_individual):
                ind_copy = Coati(self.p)
                ind_copy.copy(self.swarm[i])
                
                # Primera mitad: escaladores
                if i < self.n_individual / 2:
                    ind_copy.move_phase1_climber(iguana_lider)
                # Segunda mitad: de tierra
                else:
                    iguana_g = Coati(self.p) # Crear una posición aleatoria
                    # Pasar el objetivo del individuo original para la comparación de dominancia
                    ind_copy.move_phase1_ground(iguana_g, self.swarm[i].obj)
                
                # Actualizar si la nueva solución domina la actual
                if ind_copy.is_feasible() and Coati.dominates(ind_copy.obj, self.swarm[i].obj):
                    self.swarm[i].copy(ind_copy)
                    self.update_archive(self.swarm[i])
            
            # --- Fase 2: Explotación (Escape) --- [cite: 256]
            for i in range(self.n_individual):
                ind_copy = Coati(self.p)
                ind_copy.copy(self.swarm[i])
                
                ind_copy.move_phase2(t, self.max_iter)

                if ind_copy.is_feasible() and Coati.dominates(ind_copy.obj, self.swarm[i].obj):
                    self.swarm[i].copy(ind_copy)
                    self.update_archive(self.swarm[i])

            self.show_results(t)
            t += 1

    def show_results(self, t):
        print(f"--- Iteración {t} ---")
        print(f"Tamaño del Frente de Pareto: {len(self.pareto_archive)}")

    def optimizer(self):
        """
        Orquesta todo el proceso de optimización.
        """
        self.initialize_population()
        print("--- Población Inicial y Frente de Pareto Inicial ---")
        print(f"Tamaño del Frente de Pareto: {len(self.pareto_archive)}")
        # Ordenar por costo para una mejor visualización
        for i, sol in enumerate(sorted(self.pareto_archive, key=lambda s: s.obj[1])):
            print(f"  Solución Inicial {i+1}: {sol}")
        
        self.evolve()

# --- Ejecución ---
if __name__ == "__main__":
    problem_instance = Problem()
    # Parámetros del enjambre: puedes ajustarlos para experimentar
    swarm_instance = COA(problem=problem_instance, n_individual=50, max_iter=150)
    swarm_instance.optimizer()

    print("\n" + "="*25)
    print("--- RESULTADO FINAL ---")
    print("="*25)
    if not swarm_instance.pareto_archive:
        print("No se encontraron soluciones en el frente de Pareto.")
    else:
        print("Las mejores soluciones no dominadas (Frente de Pareto) encontradas son:")
        # Ordenar por costo (segundo objetivo) para una presentación clara
        final_solutions = sorted(swarm_instance.pareto_archive, key=lambda s: s.obj[1])
        for i, sol in enumerate(final_solutions):
            print(f"  Solución {i+1}: {sol}")