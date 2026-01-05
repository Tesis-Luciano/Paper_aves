# -*- coding: utf-8 -*-
"""
Author: Luciano Lopez Bertazza
Institution: Instituto Balseiro
Date: 2026-01-05

Description:
------------
This program simulates the spatial distribution of birds (nests) on a 2D toroidal grid.
Bird placement is governed by a local attraction rule based on nearby occupied sites.

The simulation evolves through several "tandas" (generations), and for each configuration:
    - The spatial structure is analyzed using box-counting
    - Colonies (clusters) are identified using a flood-fill algorithm
    - Colony-size distributions are computed using logarithmic binning

Outputs:
--------
- .npy files with grid configurations
- .csv files with box-counting results
- .csv files with binned colony-size distributions
"""

import random
import numpy as np
import os
import argparse
import gc
import time

# ======================================================================================
# CLASSES
# ======================================================================================

class grilla:
    """
    Represents a 2D toroidal grid.
    Each cell can be empty (0) or occupied by a bird (1).
    """
    def __init__(self, tam_X, tam_Y):
        self.mapa = np.zeros((tam_X, tam_Y), dtype=np.uint8)
        self.tamX = tam_X
        self.tamY = tam_Y

    def colocar_ave(self, bird):
        """Places a bird on the grid at its current position."""
        self.mapa[bird.last_placex, bird.last_placey] = 1

    def limpiar_ave(self, bird):
        """Removes a bird from its current position."""
        self.mapa[bird.last_placex, bird.last_placey] = 0

    def limpiar(self):
        """Deletes grid data to free memory."""
        del self.mapa
        del self.tamX
        del self.tamY
        gc.collect()

    def Counting_Box(self, nro_div):
        """
        Box-counting method.
        Divides the grid into nro_div x nro_div boxes and
        counts how many boxes contain at least one bird.
        """
        suma = 0
        tam_celda_X = self.tamX // nro_div
        tam_celda_Y = self.tamY // nro_div

        for i in range(nro_div):
            for j in range(nro_div):
                encontrado = False
                for x in range(i * tam_celda_X, (i + 1) * tam_celda_X):
                    for y in range(j * tam_celda_Y, (j + 1) * tam_celda_Y):
                        if self.mapa[x, y] == 1:
                            suma += 1
                            encontrado = True
                            break
                    if encontrado:
                        break
        return suma

    def copia(self):
        """Returns a deep copy of the grid."""
        aux = grilla(self.tamX, self.tamY)
        aux.mapa = self.mapa.copy()
        return aux

    def encontrar_colonia(self, dist_colonia):
        """
        Finds all colonies (clusters) using a flood-fill algorithm.
        Two birds belong to the same colony if their distance is <= dist_colonia.
        """
        vec_colonias = []
        mapa_aux = self.copia()

        for i in range(self.tamX):
            for j in range(self.tamY):
                if mapa_aux.mapa[i, j] != 0:
                    new_colony = Colonia(0, [])
                    flood_distancia_n([i, j], mapa_aux, new_colony, dist_colonia)
                    vec_colonias.append(new_colony)

        # Consistency check
        if np.count_nonzero(mapa_aux.mapa) == 0:
            print("Se registraron todas las aves en colonias")
        else:
            print("NO se registraron todas las aves en colonias")

        mapa_aux.limpiar()
        return vec_colonias


class Ave:
    """
    Represents a bird (nest) with an attraction-based placement rule.
    """
    def __init__(self, edad, edad_max, mapon, primera_tanda, dist_max, P_base):
        self.edad = edad
        self.vivo = True
        self.edad_max = edad_max
        self.fitness = self.edad_max - self.edad
        self.dist_max = dist_max

        if primera_tanda:
            # Random placement without attraction rule
            self.last_placex = random.randint(0, mapon.tamX - 1)
            self.last_placey = random.randint(0, mapon.tamY - 1)
            while mapon.mapa[self.last_placex, self.last_placey] == 1:
                self.last_placex = random.randint(0, mapon.tamX - 1)
                self.last_placey = random.randint(0, mapon.tamY - 1)
        else:
            # Placement governed by attraction probability
            self.last_placex = random.randint(0, mapon.tamX - 1)
            self.last_placey = random.randint(0, mapon.tamY - 1)
            nro_random_prob = random.random()
            FA = self.fn_atraccion(dist_max, mapon)

            while (mapon.mapa[self.last_placex, self.last_placey] == 1 or
                   ((nro_random_prob > FA) and (FA > P_base)) or
                   ((nro_random_prob > P_base) and (FA < P_base))):
                self.last_placex = random.randint(0, mapon.tamX - 1)
                self.last_placey = random.randint(0, mapon.tamY - 1)
                nro_random_prob = random.random()
                FA = self.fn_atraccion(dist_max, mapon)

    def nro_vecinos(self, i, mapita):
        """
        Counts neighbors at exactly distance i (square ring).
        Periodic boundary conditions are applied.
        """
        suma = 0
        x, y = self.last_placex, self.last_placey
        for dx in range(-i, i + 1):
            for dy in range(-i, i + 1):
                if abs(dx) == i or abs(dy) == i:
                    nx = (x + dx) % mapita.tamX
                    ny = (y + dy) % mapita.tamY
                    suma += mapita.mapa[nx, ny]
        return suma

    def fn_atraccion(self, dist_max, mapita):
        """
        Attraction function based on local density.
        Contributions decay as 1/r.
        """
        C = 0
        mini_prob = 0
        for i in range(1, dist_max + 1):
            C += 1 / i
            nro_sitios = 8 * i
            mini_prob += (self.nro_vecinos(i, mapita) / nro_sitios) / i
        return C * mini_prob


class Colonia:
    """
    Represents a colony (cluster) of birds.
    """
    def __init__(self, tam, vec_pos):
        self.tam = tam
        self.vec_pos = vec_pos

    def agregar_ave(self, new_bird_pos):
        self.tam += 1
        self.vec_pos.append(new_bird_pos)

    def nro_nidos(self):
        return self.tam


# ======================================================================================
# FUNCTIONS
# ======================================================================================

def nueva_tanda(mapita, edad_max_aves, nro_aves, parvada, P_base):
    """
    Generates a new generation of birds by removing old ones
    and placing new ones according to the attraction rule.
    """
    rg_vision = parvada[0].dist_max
    for _ in range(nro_aves):
        pajaro = parvada.pop(0)
        mapita.limpiar_ave(pajaro)
        nueva_ave = Ave(0, edad_max_aves, mapita, False, rg_vision, P_base)
        parvada.append(nueva_ave)
        mapita.colocar_ave(nueva_ave)


def Full_Counting_Box_2(mapilla, pot_max_div):
    """Performs box-counting for box sizes 2^k."""
    i = 2
    max_div = 2 ** pot_max_div
    Vector = []
    while i <= max_div:
        Vector.append((mapilla.tamX / i, mapilla.Counting_Box(i)))
        i *= 2
    print("Counting Box: OK")
    return Vector


def flood_distancia_n(coord, mapita, colonia, n):
    """
    Recursive flood-fill that groups birds into a colony
    using a maximum distance n.
    """
    x, y = coord
    if mapita.mapa[x, y] == 0:
        return

    colonia.agregar_ave(coord)
    mapita.mapa[x, y] = 0

    for dx in range(-n, n + 1):
        for dy in range(-n, n + 1):
            if dx == 0 and dy == 0:
                continue
            vecino = ((x + dx) % mapita.tamX, (y + dy) % mapita.tamY)
            if mapita.mapa[vecino] == 1:
                flood_distancia_n(vecino, mapita, colonia, n)


def encontrar_bines(vec_colonias):
    """
    Computes logarithmic bins for the colony-size distribution,
    choosing the smallest base that avoids empty bins.
    """
    n = max(col.nro_nidos() for col in vec_colonias)
    print(f"La colonia de mayor tamaño tiene {n} nidos")

    base = 2
    while True:
        pot_max = int(np.ceil(np.log(n) / np.log(base)))
        vec_bines = np.zeros(pot_max, dtype=int)

        for col in vec_colonias:
            for i in range(pot_max):
                if base ** i <= col.nro_nidos() < base ** (i + 1):
                    vec_bines[i] += 1
                    break

        if np.all(vec_bines > 0):
            break
        base += 1

    vec_out = np.zeros((len(vec_bines), 2))
    for i in range(len(vec_bines)):
        centro = 10 ** ((np.log10(base ** i) + np.log10(base ** (i + 1) - 1)) / 2)
        densidad = vec_bines[i] / ((base ** (i + 1) - base ** i) * len(vec_colonias))
        vec_out[i] = [centro, densidad]

    print("f(x): OK")
    return vec_out


# ======================================================================================
# MAIN
# ======================================================================================

def main(pot_tam_grilla=10, nro_tandas=3, proporcion_aves_iniciales=0.05,
         proporcion_aves=0.01, edad_max_aves=5, rango_vision=3,
         distancia_colonia=1, Prob_base=0.0001,
         nombre_gral="Carpeta Grande", nombre_chiquito="Carpeta Chiquita"):

    inicio_carga = time.time()

    tam_grilla = 2 ** pot_tam_grilla
    nro_aves = int(proporcion_aves * tam_grilla * tam_grilla)

    mapita = grilla(tam_grilla, tam_grilla)
    parvada = []

    # Initial placement
    primera_tanda = True
    for i in range(nro_aves):
        if i > int(nro_aves * proporcion_aves_iniciales):
            primera_tanda = False
        ave = Ave(0, edad_max_aves, mapita, primera_tanda, rango_vision, Prob_base)
        parvada.append(ave)
        mapita.colocar_ave(ave)

    # Directory creation
    carpeta = os.path.join(os.getcwd(), nombre_gral, nombre_chiquito)
    os.makedirs(carpeta, exist_ok=True)

    # Save initial map
    np.save(os.path.join(carpeta, "Mapa0.npy"), mapita.mapa)

    # Evolution
    for t in range(nro_tandas):
        nueva_tanda(mapita, edad_max_aves, nro_aves, parvada, Prob_base)
        np.save(os.path.join(carpeta, f"Mapa{t+1}.npy"), mapita.mapa)

    # Analysis
    for t in range(nro_tandas + 1):
        mapa_aux = grilla(tam_grilla, tam_grilla)
        mapa_aux.mapa = np.load(os.path.join(carpeta, f"Mapa{t}.npy"))
        datos_box = Full_Counting_Box_2(mapa_aux, pot_tam_grilla)
        np.savetxt(os.path.join(carpeta, f"Datos_Box_{t}.csv"),
                   datos_box, delimiter="\t", fmt="%d")

        vec_col = mapa_aux.encontrar_colonia(distancia_colonia)
        if len(vec_col) > 1 and not all(c.nro_nidos() == 1 for c in vec_col):
            vec_fx = encontrar_bines(vec_col)
            np.savetxt(os.path.join(carpeta, f"Vec_fx_{t}.csv"),
                       vec_fx, delimiter="\t")

        mapa_aux.limpiar()

    print(f"Tiempo total: {(time.time() - inicio_carga) / 60:.2f} min")


# ======================================================================================
# ARGUMENT PARSING
# ======================================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rango_vision", type=int, default=3)
    parser.add_argument("--proporcion_aves", type=float, default=0.01)
    parser.add_argument("--pot_tam_grilla", type=int, default=10)
    parser.add_argument("--nro_tandas", type=int, default=3)
    parser.add_argument("--proporcion_aves_iniciales", type=float, default=0.05)
    parser.add_argument("--edad_max_aves", type=int, default=5)
    parser.add_argument("--distancia_colonia", type=int, default=1)
    parser.add_argument("--Prob_base", type=float, default=0.0001)
    parser.add_argument("--nombre_gral", type=str, default="Carpeta Grande")
    parser.add_argument("--nombre_chiquito", type=str, default="Carpeta Chiquita")

    args = parser.parse_args()

    main(**vars(args))
