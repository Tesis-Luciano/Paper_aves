/*******************************************************
 * Author: Luciano Lopez Bertazza
 * Institution: Instituto Balseiro
 * Date: 2026-01-05
 *
 * Description:
 * This program simulates the spatial distribution of birds
 * on a 2D periodic grid, incorporating local attraction rules.
 * It analyzes the resulting spatial patterns using:
 *  - Box-counting (fractal analysis)
 *  - Colony (cluster) size distributions
 *
 * The output includes:
 *  - Numpy arrays (.npy) with spatial configurations
 *  - CSV files with box-counting data
 *  - CSV files with binned colony-size distributions
 *******************************************************/

#include <iostream>
#include <vector>
#include <random>
#include <fstream>
#include <filesystem>
#include <cmath>
#include <algorithm>
#include <cassert>
#include <stack>
#include <chrono>
#include "cnpy.h"

using namespace std;
namespace fs = std::filesystem;

/* =====================================================
 * CLASS: Grilla
 * -----------------------------------------------------
 * Represents a 2D periodic grid where birds are placed.
 * Each cell contains:
 *   0 -> empty
 *   1 -> occupied by a bird
 * ===================================================== */
class Grilla {
public:
    int tamX, tamY;                         // Grid dimensions
    vector<vector<uint8_t>> mapa;           // Occupancy map

    // Constructor: initializes an empty grid
    Grilla(int x, int y) : tamX(x), tamY(y) {
        mapa = vector<vector<uint8_t>>(tamX, vector<uint8_t>(tamY, 0));
    }

    // Places a bird at (x, y)
    void colocar_ave(int x, int y) {
        assert(x >= 0 && x < tamX);
        assert(y >= 0 && y < tamY);
        mapa[x][y] = 1;
    }

    // Clears the entire grid
    void limpiar() {
        for (auto& fila : mapa) {
            fill(fila.begin(), fila.end(), 0);
        }
    }

    /* -------------------------------------------------
     * Box-counting method:
     * Divides the grid into div x div boxes and counts
     * how many boxes contain at least one bird.
     * ------------------------------------------------- */
    int contar_boxes(int div) {
        int boxes_con_aves = 0;
        int celX = tamX / div;
        int celY = tamY / div;

        for (int i = 0; i < div; ++i) {
            for (int j = 0; j < div; ++j) {
                bool hay_ave = false;
                for (int x = 0; x < celX && !hay_ave; ++x) {
                    for (int y = 0; y < celY && !hay_ave; ++y) {
                        if (mapa[i * celX + x][j * celY + y] == 1) {
                            hay_ave = true;
                            ++boxes_con_aves;
                        }
                    }
                }
            }
        }
        return boxes_con_aves;
    }

    // Saves the grid as a NumPy .npy file
    void guardar_npy(const string& ruta) {
        vector<uint8_t> plano(tamX * tamY);
        for (int i = 0; i < tamX; ++i) {
            for (int j = 0; j < tamY; ++j) {
                plano[i * tamY + j] = mapa[i][j];
            }
        }
        cnpy::npy_save(ruta, &plano[0],
                       {static_cast<size_t>(tamX), static_cast<size_t>(tamY)}, "w");
    }

    /* -------------------------------------------------
     * Counts colonies (clusters) using DFS.
     * Two birds belong to the same colony if their
     * distance (Chebyshev) is <= distancia.
     * Periodic boundary conditions are applied.
     * ------------------------------------------------- */
    vector<pair<int, int>> contar_colonias(int distancia) {
        vector<pair<int, int>> colonias;
        vector<vector<bool>> visitado(tamX, vector<bool>(tamY, false));

        for (int i = 0; i < tamX; ++i) {
            for (int j = 0; j < tamY; ++j) {
                if (!visitado[i][j] && mapa[i][j] == 1) {
                    int tam = 0;
                    stack<pair<int, int>> pila;
                    pila.push({i, j});
                    visitado[i][j] = true;

                    while (!pila.empty()) {
                        auto [x, y] = pila.top(); pila.pop();
                        ++tam;

                        for (int dx = -distancia; dx <= distancia; ++dx) {
                            for (int dy = -distancia; dy <= distancia; ++dy) {
                                if (dx != 0 || dy != 0) {
                                    int nx = (x + dx + tamX) % tamX;
                                    int ny = (y + dy + tamY) % tamY;
                                    if (!visitado[nx][ny] && mapa[nx][ny] == 1) {
                                        visitado[nx][ny] = true;
                                        pila.push({nx, ny});
                                    }
                                }
                            }
                        }
                    }
                    // Store (colony size, frequency=1)
                    colonias.emplace_back(tam, 1);
                }
            }
        }
        return colonias;
    }
};

/* =====================================================
 * CLASS: Ave
 * -----------------------------------------------------
 * Represents a bird with a position and vision range.
 * Placement depends on a local attraction function.
 * ===================================================== */
class Ave {
public:
    int x, y;               // Position
    int rango_vision;       // Interaction range

    // Constructor: places a bird using attraction rules
    Ave(Grilla& g, mt19937& gen, int rango, bool es_semilla, double P_Base)
        : rango_vision(rango) {

        uniform_int_distribution<> distX(0, g.tamX - 1);
        uniform_int_distribution<> distY(0, g.tamY - 1);
        uniform_real_distribution<> distR(0.0, 1.0);

        double r;
        do {
            x = distX(gen);
            y = distY(gen);
            r = distR(gen);
        } while (
            g.mapa[x][y] == 1 ||
            (!es_semilla && r > max(fn_atraccion(g), P_Base))
        );

        g.colocar_ave(x, y);
    }

    // Counts neighbors exactly at distance d (ring)
    int contar_vecinos(const Grilla& g, int d) {
        int suma = 0;
        for (int dx = -d; dx <= d; ++dx) {
            for (int dy = -d; dy <= d; ++dy) {
                if (abs(dx) == d || abs(dy) == d) {
                    int nx = (x + dx + g.tamX) % g.tamX;
                    int ny = (y + dy + g.tamY) % g.tamY;
                    suma += g.mapa[nx][ny];
                }
            }
        }
        return suma;
    }

    /* -------------------------------------------------
     * Attraction function:
     * Weighted sum of neighbor densities at increasing
     * distances, decaying as 1/r.
     * ------------------------------------------------- */
    double fn_atraccion(const Grilla& g) {
        double C = 0.0, suma = 0.0;
        for (int i = 1; i <= rango_vision; ++i) {
            int sitios = 8 * i;
            C += 1.0 / i;
            suma += (contar_vecinos(g, i) / static_cast<double>(sitios)) / i;
        }
        return C * suma;
    }
};

// Creates directory if it does not exist
void crear_carpeta(const string& path) {
    if (!fs::exists(path)) {
        fs::create_directories(path);
    }
}

/* =====================================================
 * Logarithmic binning of colony sizes.
 * Automatically adjusts bin base to avoid empty bins.
 * Returns (bin center, normalized density).
 * ===================================================== */
vector<pair<double, double>> encontrar_bines(const vector<pair<int, int>>& colonias) {
    vector<int> tamanios;
    for (const auto& [tam, freq] : colonias) {
        for (int i = 0; i < freq; ++i) {
            tamanios.push_back(tam);
        }
    }

    int max_tam = *max_element(tamanios.begin(), tamanios.end());
    int base = 2;
    vector<pair<double, double>> bines_finales;

    bool sin_bines_vacios = false;
    vector<int> vec_bines_count;

    while (!sin_bines_vacios && base <= max_tam) {
        vec_bines_count.clear();
        int pot_max = 0;
        while (pow(base, pot_max) < max_tam) {
            ++pot_max;
        }

        vec_bines_count.resize(pot_max, 0);

        for (int tam : tamanios) {
            for (int i = 0; i < vec_bines_count.size(); ++i) {
                int bin_min = pow(base, i);
                int bin_max = pow(base, i + 1);
                if (tam >= bin_min && tam < bin_max) {
                    vec_bines_count[i]++;
                    break;
                }
            }
        }

        sin_bines_vacios = all_of(vec_bines_count.begin(),
                                  vec_bines_count.end(),
                                  [](int c) { return c > 0; });

        if (!sin_bines_vacios) ++base;
    }

    for (int i = 0; i < vec_bines_count.size(); ++i) {
        if (vec_bines_count[i] > 0) {
            int bin_min = pow(base, i);
            int bin_max = pow(base, i + 1);
            double centro = pow(10.0,
                (log10(bin_min) + log10(bin_max - 1.0)) / 2.0);
            double densidad =
                static_cast<double>(vec_bines_count[i]) /
                ((bin_max - bin_min) * tamanios.size());
            bines_finales.emplace_back(centro, densidad);
        }
    }

    return bines_finales;
}

/* =====================================================
 * MAIN PROGRAM
 * ===================================================== */
int main(int argc, char* argv[]) {

    // Start timing execution
    auto start = chrono::high_resolution_clock::now();

    try {
        // Default parameters
        int pot_tam_grilla = 10;
        float proporcion_aves = 0.01;
        float aves_semillas = 0.05;
        int nro_tandas = 3;
        int rango_vision = 3;
        int distancia_colonia = 1;
        double Prob_Base = 0.0001;
        string nombre_gral = "Default_Carpeta";
        string nombre_chiquito = "Subcarpeta";

        // Command-line argument parsing
        if (argc == 10) {
            pot_tam_grilla = stoi(argv[1]);
            proporcion_aves = stof(argv[2]);
            aves_semillas = stof(argv[3]);
            nro_tandas = stoi(argv[4]);
            rango_vision = stoi(argv[5]);
            distancia_colonia = stoi(argv[6]);
            Prob_Base = stod(argv[7]);
            nombre_gral = argv[8];
            nombre_chiquito = argv[9];
        }

        int tam = 1 << pot_tam_grilla;
        int nro_aves = static_cast<int>(tam * tam * proporcion_aves);

        Grilla grilla(tam, tam);
        random_device rd;
        mt19937 gen(rd());

        // Initial bird placement
        vector<Ave> aves;
        int cant_semillas = static_cast<int>(nro_aves * aves_semillas);
        for (int i = 0; i < nro_aves; ++i) {
            aves.emplace_back(grilla, gen, rango_vision,
                              i < cant_semillas, Prob_Base);
        }

        // Output directory
        string carpeta = nombre_gral + "/" + nombre_chiquito;
        crear_carpeta(carpeta);

        // Save initial configuration
        grilla.guardar_npy(carpeta + "/Mapa0.npy");

        // Subsequent simulation rounds
        for (int t = 1; t <= nro_tandas; ++t) {
            vector<Ave> nuevas_aves;
            for (auto& a : aves) {
                grilla.mapa[a.x][a.y] = 0;
                nuevas_aves.emplace_back(grilla, gen, rango_vision, false, Prob_Base);
            }
            aves = move(nuevas_aves);
            grilla.guardar_npy(carpeta + "/Mapa" + to_string(t) + ".npy");
        }

        cout << "Simulation completed successfully.\n";
    }
    catch (const exception& e) {
        cerr << "Exception: " << e.what() << endl;
    }

    // End timing
    auto end = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::seconds>(end - start);
    cout << "Execution time: "
         << duration.count() / 60 << " min "
         << duration.count() % 60 << " s\n";

    return 0;
}
