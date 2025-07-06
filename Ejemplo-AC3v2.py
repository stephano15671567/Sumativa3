import json

# Definición de dominios (basados en la cantidad máxima de anuncios permitidos)
domains = {
    'Tarde': list(range(16)),     # x1: 0 a 15
    'Noche': list(range(11)),     # x2: 0 a 10
    'Diario': list(range(26)),    # x3: 0 a 25
    'Revistas': list(range(5)),   # x4: 0 a 4
    'Radio': list(range(31)),     # x5: 0 a 30
}

# Restricciones binarias derivadas de restricciones de costo
constraints = {
    # Televisión (Tarde y Noche): 160*x1 + 300*x2 <= 3800
    ('Tarde', 'Noche'): lambda t, n: 160*t + 300*n <= 3800,
    ('Noche', 'Tarde'): lambda n, t: 300*n + 160*t <= 3800,

    # Diario y Revistas: 40*x3 + 100*x4 <= 2800
    ('Diario', 'Revistas'): lambda d, r: 40*d + 100*r <= 2800,
    ('Revistas', 'Diario'): lambda r, d: 100*r + 40*d <= 2800,

    # Diario y Radio: 40*x3 + 10*x5 <= 3500
    ('Diario', 'Radio'): lambda d, r: 40*d + 10*r <= 3500,
    ('Radio', 'Diario'): lambda r, d: 10*r + 40*d <= 3500,
}

# Función revise del algoritmo AC-3
def revise(x, y):
    revised = False
    x_domain = domains[x][:]
    y_domain = domains[y]
    all_constraints = [
        constraint for constraint in constraints if constraint[0] == x and constraint[1] == y]
    for x_value in x_domain:
        satisfies = False
        for y_value in y_domain:
            for constraint in all_constraints:
                constraint_func = constraints[constraint]
                if constraint_func(x_value, y_value):
                    satisfies = True
                    break
            if satisfies:
                break
        if not satisfies:
            domains[x].remove(x_value)
            revised = True
    return revised

# Algoritmo AC-3
def ac3(arcs):
    queue = arcs[:]
    while queue:
        (x, y) = queue.pop(0)
        if revise(x, y):
            neighbors = [arc for arc in arcs if arc[1] == x and arc[0] != y]
            queue.extend(neighbors)

# Lista de arcos (pares de variables con restricciones entre ellas)
arcs = list(constraints.keys())

# Ejecutar AC-3
ac3(arcs)

# Mostrar dominios reducidos
print("Dominios después de aplicar AC-3:")
for var, domain in domains.items():
    print(f"{var}: {domain}")

# Guardar dominios en un archivo JSON
with open("ac3_domains.json", "w") as f:
    json.dump(domains, f)