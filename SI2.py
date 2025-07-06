import json
import random
import math

# --- Algoritmo COA (Coatí Optimization Algorithm) ---
# Basado en el artículo: "Coatí Optimization Algorithm: A new bio-inspired metaheuristic for solving optimization problems" 

# Cargar dominios reducidos por AC-3
with open("ac3_domains.json", "r") as f:
    ac3_domains = json.load(f)

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
        """
        valorizacion = sum(v * x[i] for i, v in enumerate(self.valorizaciones))
        costo = sum(c * x[i] for i, c in enumerate(self.costos))
        return [-valorizacion, costo]

class Individual:
    """
    Representa una solución candidata (un coatí).
    """
    def __init__(self, problem):
        self.p = problem
        self.dimension = self.p.dim
        self.x = [random.choice(ac3_domains[varnames[i]]) for i in range(self.dimension)]
        self.obj = self.p.objectives(self.x)

    def is_feasible(self):
        return self.p.check(self.x)

    def move_phase1(self, iguana, iguana_g, index, population_size):
        """
        Fase 1 de COA: Caza de iguanas (Exploración). [cite: 205]
        MODIFICADO: La primera mitad de la población se mueve hacia 'iguana' (líder).
        La segunda mitad se mueve hacia 'iguana_g' (posición aleatoria).
        """
        # Decidir el objetivo basado en el índice del individuo
        # La primera mitad son los "escaladores", la segunda son los "de tierra"
        if index < population_size / 2:
            target_iguana = iguana
        else:
            target_iguana = iguana_g
        
        I = random.choice([1, 2])

        for j in range(self.dimension):
            domain = ac3_domains[varnames[j]]
            r = random.random()
            
            # Adaptación de la Ecuación (4) y (6) para un problema discreto
            move_vector = r * (target_iguana.x[j] - I * self.x[j])
            new_val_float = self.x[j] + move_vector
            
            # Buscamos el valor más cercano en el dominio al resultado
            new_val = min(domain, key=lambda v: abs(v - new_val_float))
            self.x[j] = new_val
            
        self.obj = self.p.objectives(self.x)

    def move_phase2(self, t, max_iter):
        """
        Fase 2 de COA: Escape de depredadores (Explotación). [cite: 256]
        """
        for j in range(self.dimension):
            domain = ac3_domains[varnames[j]]
            
            # Adaptamos la Ecuación (8) y (9) para un espacio discreto
            local_range_factor = (1 - (t / max_iter))
            max_step = math.ceil(len(domain) * local_range_factor * 0.5)
            
            step = random.randint(-max_step, max_step)
            current_index_in_domain = domain.index(self.x[j])
            new_index = current_index_in_domain + step
            
            # Asegura que el nuevo índice esté dentro de los límites del dominio
            new_index = max(0, min(len(domain) - 1, new_index))
            self.x[j] = domain[new_index]

        self.obj = self.p.objectives(self.x)
        
    @staticmethod
    def dominates(obj_a, obj_b):
        all_le = all(a <= b for a, b in zip(obj_a, obj_b))
        any_lt = any(a < b for a, b in zip(obj_a, obj_b))
        return all_le and any_lt

    def copy(self, other):
        if isinstance(other, Individual):
            self.x = other.x.copy()
            self.obj = other.obj.copy()

    def __str__(self):
        valorizacion = -self.obj[0]
        costo = self.obj[1]
        return f"Sol: {self.x}, Valorización: {int(valorizacion)}, Costo: {costo}"

class Swarm:
    def __init__(self, problem, n_individual=30, max_iter=100):
        self.p = problem
        self.n_individual = n_individual
        self.max_iter = max_iter
        self.swarm = []
        self.pareto_archive = []

    def update_archive(self, new_individual):
        is_dominated = False
        for sol in self.pareto_archive:
            if Individual.dominates(sol.obj, new_individual.obj):
                is_dominated = True
                break
        if is_dominated:
            return

        self.pareto_archive = [sol for sol in self.pareto_archive if not Individual.dominates(new_individual.obj, sol.obj)]
        self.pareto_archive.append(new_individual)

    def initialize_population(self):
        for _ in range(self.n_individual):
            feasible = False
            while not feasible:
                ind = Individual(self.p)
                feasible = ind.is_feasible()
            self.swarm.append(ind)
        
        for ind in self.swarm:
            self.update_archive(ind)

    def evolve(self):
        t = 1
        while t <= self.max_iter:
            if not self.pareto_archive:
                print("El archivo de Pareto está vacío. Deteniendo la evolución.")
                break
            
            iguana = random.choice(self.pareto_archive)
            
            # --- Fase 1: Exploración (Caza) ---
            iguana_g = Individual(self.p)
            while not iguana_g.is_feasible():
                iguana_g = Individual(self.p)
            
            # MODIFICADO: Pasamos el índice (i) y el tamaño del enjambre
            for i in range(self.n_individual):
                ind_copy = Individual(self.p)
                ind_copy.copy(self.swarm[i])
                
                ind_copy.move_phase1(iguana, iguana_g, i, self.n_individual)
                
                if ind_copy.is_feasible() and Individual.dominates(ind_copy.obj, self.swarm[i].obj):
                    self.swarm[i].copy(ind_copy)
                    self.update_archive(self.swarm[i])
            
            # --- Fase 2: Explotación (Escape) ---
            for i in range(self.n_individual):
                ind_copy = Individual(self.p)
                ind_copy.copy(self.swarm[i])
                
                ind_copy.move_phase2(t, self.max_iter)

                if ind_copy.is_feasible() and Individual.dominates(ind_copy.obj, self.swarm[i].obj):
                    self.swarm[i].copy(ind_copy)
                    self.update_archive(self.swarm[i])

            self.show_results(t)
            t += 1

    def show_results(self, t):
        print(f"--- Iteración {t} ---")
        print(f"Tamaño del Frente de Pareto: {len(self.pareto_archive)}")
        # Opcional: imprimir solo algunas soluciones para no saturar la consola
        # for i, sol in enumerate(self.pareto_archive[:5]):
        #     print(f"  Solución {i+1}: {sol}")

    def optimizer(self):
        self.initialize_population()
        print("--- Población Inicial ---")
        print(f"Tamaño del Frente de Pareto inicial: {len(self.pareto_archive)}")
        for i, sol in enumerate(self.pareto_archive):
            print(f"  Solución {i+1}: {sol}")
        
        self.evolve()

# --- Ejecución ---
problem_instance = Problem()
swarm_instance = Swarm(problem=problem_instance, n_individual=30, max_iter=100)
swarm_instance.optimizer()

print("\n--- Resultado Final ---")
print("Las mejores soluciones (Frente de Pareto) encontradas son:")
for i, sol in enumerate(sorted(swarm_instance.pareto_archive, key=lambda s: s.obj[1])): # Ordenadas por costo
    print(f"  Solución {i+1}: {sol}")