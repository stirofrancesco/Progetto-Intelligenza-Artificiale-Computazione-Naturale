# -*- coding: utf-8 -*-
"""
Created on Sun June 09 17:34:50 2024

@author: francesco_stiro
"""

import numpy as np
import random as ran
import sys
import math
import csv
from collections import defaultdict
import time
from DancingLinks import *

DIM = 9
sub_len = int(DIM**0.5)
POPULATION_SIZE = 1
MUTATE_RATE = 0.3
RINITIALIZE_RATE = 0.05
THERSHOLD_RESTART = 30
PC = 0.1

class Individual:
    
    __fitness = sys.maxsize
    __model = []
    flag = False
    
    def __generate_individual(self):      
        for i in range(0,DIM,sub_len):
            for j in range(0,DIM,sub_len):
                possible_values = list(range(1, 10))
                sub_matrix = self.__model[i:i+sub_len,j:j+sub_len]
                indexes_not_zero = np.nonzero(sub_matrix)
                not_zero_elements = sub_matrix[indexes_not_zero]
                possible_values = [elem for elem in possible_values if elem not in not_zero_elements]
                ran.shuffle(possible_values)

                for row in range(i, i + sub_len):
                    for col in range(j, j + sub_len):
                        if(self.__solution[row][col] == 0):
                            self.__solution[row][col] = possible_values.pop()
                            
    
    def __fitness_vec(self,vector):
        return DIM-len(set(vector))

    def __calc_fitness(self):

        fitness = 0
            
        #Controllo delle righe
            
        for row in self.__solution:
            fitness+= self.__fitness_vec(row)
            
        #Controllo delle colonne
            
        for col in range(DIM):
            column = [self.__solution[row][col] for row in range(DIM)]
            fitness += self.__fitness_vec(column)
        
        self.__fitness = fitness
            
            
    def __init__(self, model, copy = []):
        

        self.__model = model.copy()
            
        if len(copy) > 0:
            self.__solution = copy.copy()
        else:
            self.__solution = model.copy()
            self.__generate_individual()
        self.__calc_fitness()
    
    def __mark_duplicates(self,vector):
        marked = [False] * len(vector)  # Inizializza un vettore di booleani con False
        
        # Dizionario per tenere traccia delle posizioni dei numeri
        positions = defaultdict(list)
        
        # Scorrimento del vettore
        for i, number in enumerate(vector):
            positions[number].append(i)
        
        # Trova numeri con più di una posizione
        duplicates = {number: pos for number, pos in positions.items() if len(pos) > 1}
        
        # Segna le posizioni con numeri duplicati
        for positions in duplicates.values():
            for position in positions:
                marked[position] = True
        
        return marked
    
    def __extract_random_number(self,valori):
        while len(valori)>0:
            elemento = ran.choice(valori)  # Sceglie casualmente un elemento dalla lista
            valori.remove(elemento)  # Rimuove l'elemento dalla lista
            yield elemento  # Ritorna l'elemento
    
    def __same_block(self, row1, col1, row2, col2):
        # Calcola l'indice del blocco lungo le righe e le colonne
        block_row1, block_col1 = row1 // 3, col1 // 3
        block_row2, block_col2 = row2 // 3, col2 // 3
        
        # Verifica se le righe e le colonne appartengono allo stesso blocco
        return block_row1 == block_row2 and block_col1 == block_col2
    
    def __local_search_col(self):
        violate = []
        for i in range(DIM):
            if(self.__fitness_vec(self.__solution[:,i])):
                violate.append(i)
        
        while(len(violate)>1):
            col_1 = violate.pop()
            individual_col_1 = self.__solution[:,col_1]
            
            tmp_violate = violate.copy()
            for elem in self.__extract_random_number(tmp_violate):
                col_2 = self.__solution[:,elem]
                duplicate_1 = self.__mark_duplicates(individual_col_1)
                duplicate_2 = self.__mark_duplicates(col_2)
                for i in range(DIM):
                    for j in range(DIM):
                        if(duplicate_1[i] == 1 and duplicate_2[j] == 1 ):
                            if self.__same_block(i, col_1, j, elem) and self.__model[i,col_1] == 0 and self.__model[j,elem] == 0:
                                if(self.__solution[i,col_1] not in col_2 and self.__solution[j,elem] not in individual_col_1):
                                    #swap
                                    tmp = individual_col_1[i]
                                    individual_col_1[i] = col_2[j]
                                    col_2[j] = tmp
                                    
                                    #Aggiornamento dei vettori
                                    duplicate_1 = self.__mark_duplicates(individual_col_1)
                                    duplicate_2 = self.__mark_duplicates(col_2)
                                    
                
                self.__solution[:,elem] = col_2
                
            self.__solution[:,col_1] = individual_col_1
            #print(self.__solution)

    
    
    def __local_search_row(self):
        violate = []
        for i in range(DIM):
            if(self.__fitness_vec(self.__solution[i])):
                violate.append(i)
        
        while(len(violate)>1):
            row_1 = violate.pop()
            individual_row_1 = self.__solution[row_1]
            
            tmp_violate = violate.copy()
            for elem in self.__extract_random_number(tmp_violate):
                row_2 = self.__solution[elem]
                duplicate_1 = self.__mark_duplicates(individual_row_1)
                duplicate_2 = self.__mark_duplicates(row_2)
                for i in range(DIM):
                    for j in range(DIM):
                        if(duplicate_1[i] == 1 and duplicate_2[j] == 1 ):
                            if self.__same_block(row_1, i, elem, j) and self.__model[row_1,i] == 0 and self.__model[elem,j] == 0:
                                if(self.__solution[row_1,i] not in row_2 and self.__solution[elem,j] not in individual_row_1):
                                    #swap
                                    tmp = individual_row_1[i]
                                    individual_row_1[i] = row_2[j]
                                    row_2[j] = tmp
                                    
                                    #Aggiornamento dei vettori
                                    
                                    duplicate_1 = self.__mark_duplicates(individual_row_1)
                                    duplicate_2 = self.__mark_duplicates(row_2)
                
                self.__solution[elem] = row_2 
                
            self.__solution[row_1] = individual_row_1
    
    def local_search(self):
        self.__local_search_row()
        self.__local_search_col()
        self.__calc_fitness()
                            

    def get_fitness(self):
        return self.__fitness
    
    def get_individual(self):
        return self.__solution
    
    def get_model(self):
        return self.__model
    
    def set_flag(self):
        self.__flag = True
    
    def get_flag(self):
        return self.__flag
    
    def set_fitness(self, penalty):
        self.__fitness += penalty

    def set_individual(self, individual):
        self.__solution = individual.copy()


def fitness_vec(vector):
    return DIM-len(set(vector))          
            
#Generazione di una popolazione inziale di numerosità POPULATION_SIZE

def initial_population(configuration):
    return [Individual(configuration) for _ in range(POPULATION_SIZE)]

#Simulazione del lancio di una moneta di probabilità p 

def coin_toss(probability):
    if(ran.random() < probability):
        return True
    else:
        return False
    
#Controllo regolarità scacchiera
    
def check_board(sudoku):
    filled = np.zeros(10,int)
    filled[0] = 9
    for i in range(DIM):
        for j in range(DIM):
            filled[sudoku[i][j]]+=1

    if(filled == 9).all():
        return True
    else:
        return False

def crossover2(parent1,parent2,model):
    sub_len = int(DIM**0.5)
    offspring1 = np.zeros((9,9),int)
    offspring2 = np.zeros((9,9),int)
    
    for i in range(0,DIM,sub_len):
        if fitness_vec(parent2[0+i]) + fitness_vec(parent2[1+i]) + fitness_vec(parent2[2+i]) < fitness_vec(parent1[0+i]) + fitness_vec(parent1[1+i]) + fitness_vec(parent1[2+i]):
            for j in range(sub_len):
                offspring1[j+i] = parent2[j+i]
        else:
            for j in range(sub_len):
                offspring1[j+i] = parent1[j+i]
                
        if fitness_vec(parent2[:,0+i]) + fitness_vec(parent2[:,1+i]) + fitness_vec(parent2[:,2+i]) < fitness_vec(parent1[:,0+i]) + fitness_vec(parent1[:,1+i]) + fitness_vec(parent1[:,2+i]):
            for j in range(sub_len):
                offspring2[:,j+i] = parent2[:,j+i]
        else:
            for j in range(sub_len):
                offspring2[:,j+i] = parent1[:,j+i]
    
    return Individual(model,offspring1),Individual(model,offspring2)

def crossover(parent1,parent2,model):
    sub_len = int(DIM**0.5)
    offspring1 = np.zeros((9,9),int)
    offspring2 = np.zeros((9,9),int)
    
    for i in range(0,DIM,sub_len):
        for j in range(0,DIM,sub_len):
            if(coin_toss(PC) == True):
                for row in range(i, i + sub_len):
                    for col in range(j, j + sub_len):
                        offspring1[row][col] = parent2[row][col]
                        offspring2[row][col] = parent1[row][col]
            else:
                for row in range(i, i + sub_len):
                    for col in range(j, j + sub_len):
                        offspring1[row][col] = parent1[row][col]
                        offspring2[row][col] = parent2[row][col]
                
    return Individual(model,offspring1),Individual(model,offspring2)                                                      

def mutation(individual,model):
    sub_len = int(DIM**0.5)
    
    mutated = individual.copy()
    for i in range(0,DIM,sub_len):
        for j in range(0,DIM,sub_len):
            if(coin_toss(MUTATE_RATE) == True):
                submatrix_indices = np.argwhere(model[i:i+sub_len,j:j+sub_len] == 0)
                submatrix_indices = submatrix_indices.tolist()
                ran.shuffle(submatrix_indices)
                index_1 = submatrix_indices.pop()
                index_2 = submatrix_indices.pop()
                tmp = mutated[index_1[0]+i,index_1[1]+j]
                mutated[index_1[0]+i,index_1[1]+j] = mutated[index_2[0]+i,index_2[1]+j]
                mutated [index_2[0]+i,index_2[1]+j] = tmp
                
            elif(coin_toss(RINITIALIZE_RATE) == True):
                new = Individual(model).get_individual()
                for k in range(i,i+sub_len):
                    for z in range(j,j+sub_len):
                        mutated[k][z] = new[k][z]
    
    return Individual(model,mutated)

def swap(population,i,j):
    tmp = population[i]
    population[i] = population[j]
    population[j] = tmp

def extract_random_number(valori):
    while len(valori)>0:
        elemento = ran.choice(valori)  # Sceglie casualmente un elemento dalla lista
        valori.remove(elemento)  # Rimuove l'elemento dalla lista
        return elemento  # Ritorna l'elemento
    
import copy
from collections import defaultdict

def remove_all_conflicting_values(grid, fixed):
    new_grid = copy.deepcopy(grid)

    # RIGHE
    for i in range(N):
        count = defaultdict(int)
        for j in range(N):
            val = new_grid[i][j]
            if val != 0:
                count[val] += 1
        for j in range(N):
            val = new_grid[i][j]
            if val != 0 and count[val] > 1 and fixed[i][j]==0:
                new_grid[i][j] = 0

    # COLONNE
    for j in range(N):
        count = defaultdict(int)
        for i in range(N):
            val = new_grid[i][j]
            if val != 0:
                count[val] += 1
        for i in range(N):
            val = new_grid[i][j]
            if val != 0 and count[val] > 1 and fixed[i][j]==0:
                new_grid[i][j] = 0

    return new_grid

def initialize_candidates_constrained(grid):
    n = len(grid)
    sub = int(n**0.5)
    grid = grid.tolist() if hasattr(grid, 'tolist') else grid
    candidates = [[[] for _ in range(n)] for _ in range(n)]

    for i in range(n):
        for j in range(n):
            if grid[i][j] != 0:
                candidates[i][j] = [grid[i][j]]
            else:
                # Prendi tutti i numeri già usati in riga, colonna, blocco
                used = set()

                used.update(int(v) for v in grid[i] if v != 0) # riga
                used.update(int(grid[r][j]) for r in range(n) if grid[r][j] != 0)  # colonna

                # blocco
                bi, bj = i // sub * sub, j // sub * sub
                for r in range(bi, bi + sub):
                    for c in range(bj, bj + sub):
                        val = grid[r][c]
                        if val != 0:
                            used.add(int(val))

                candidates[i][j] = [int(v) for v in range(1, n+1) if v not in used]

    return candidates

def initialize_candidates(grid, fixed):
    n = len(grid)
    sub = int(n**0.5)
    candidates = [[[] for _ in range(n)] for _ in range(n)]

    for i in range(n):
        for j in range(n):
            if fixed[i][j] != 0 or grid[i][j] != 0:
                candidates[i][j] = [grid[i][j]]
                continue

            used = set()
            used.update(grid[i][k] for k in range(n) if grid[i][k] != 0)
            used.update(grid[k][j] for k in range(n) if grid[k][j] != 0)
            bi, bj = i // sub * sub, j // sub * sub
            for r in range(bi, bi + sub):
                for c in range(bj, bj + sub):
                    if grid[r][c] != 0:
                        used.add(grid[r][c])

            candidates[i][j] = [v for v in range(1, n+1) if v not in used]

    return candidates


def reduce_until_domains_safe(grid, fixed, initialize_candidates_fn, max_steps=81):
    n = len(grid)
    new_grid = copy.deepcopy(grid)

    # Celle non fisse
    mutable_cells = [(i, j) for i in range(n) for j in range(n) if fixed[i][j] == 0]
    ran.shuffle(mutable_cells)

    step = 0
    while step < max_steps and mutable_cells:
        i, j = mutable_cells.pop()

        # Tenta rimozione temporanea
        old_value = new_grid[i][j]
        new_grid[i][j] = 0

        # Calcola i candidati DOPO la rimozione
        candidates = initialize_candidates_fn(new_grid, fixed)

        # Verifica che NESSUNA cella vuota abbia dominio vuoto
        domains_ok = all(
            len(candidates[r][c]) > 0
            for r in range(n) for c in range(n)
            if new_grid[r][c] == 0
        )

        if domains_ok:
            step += 1  # rimozione valida
        else:
            new_grid[i][j] = old_value  # annulla la rimozione

    # Conversione profonda np.int → int
    for i in range(n):
        for j in range(n):
            candidates[i][j] = [int(x) for x in candidates[i][j]]


    return new_grid, candidates



def brkga_algorithm(sudoku):
    
    #Timer
    start_time = time.time()
    
    #Visualizzazione del sudoku iniziale
    print("Sudoku in input: ")
    print(sudoku)
    
    #Calcolo del numero di indizi del sudoku in input
    givens = np.count_nonzero(sudoku)

    #Aggiornamento numerosità popolazione
    global POPULATION_SIZE
    POPULATION_SIZE = 3000

    print("Givens: " + str(givens))
    print("Popolazione: " + str(POPULATION_SIZE))
    
    print("--------------")
    
    num_restart = 0
    
    p_e = math.ceil(0.25*POPULATION_SIZE)
    p_e_m = math.ceil(0.10 * POPULATION_SIZE)
    
    #print(p_e)
    
    p_m = math.ceil(0.05*POPULATION_SIZE)
    
    #print(p_m)
    
    p_p = math.ceil((POPULATION_SIZE-p_e-p_m)/2)
    
    #print(p_p)
    
    while(True):
        
        #Migliore soluzione inizializzata a +inf (Realisticamente al massimo intero consentito in python)
        
        best_solution = []
        best_solution_fitness = sys.maxsize  
        
        generation = 0
        
        #Generazione della popolazione iniziale
        
        population = initial_population(sudoku)
        
        changes = 0
        found = False

        while(not found and best_solution_fitness != 0 and changes < THERSHOLD_RESTART ):
            
            #best_solution_fitness = 0

            #Ordino le soluzioni in base alla fitness (in senso non decrescente)
            
            population = sorted(population, key=lambda x: x.get_fitness())
            
            if(population[0].get_fitness() < best_solution_fitness):
               changes = 0
               best_solution = population[0].get_individual()
               best_solution_fitness = population[0].get_fitness()
               
            print("Generazione: " + str(generation+1))
            print(population[0].get_individual())
            print("Fitness: " + str(population[0].get_fitness()))
            
            #Effettuo il population interchange
            
            '''if(generation % 50 == 0):
                candidates_e = list(range(p_e))
                candidates_ne = list(range(p_e,POPULATION_SIZE))
                ran.shuffle(candidates_e)
                ran.shuffle(candidates_ne)
                for _ in range(3):
                    i = extract_random_number(candidates_e)
                    j = extract_random_number(candidates_ne)
                    swap(population,i,j)'''
            
            
            #Inizializzo la nuova generazione inserendo l'elite population
            
            new_generation = [population[i] for i in range(p_e)]
            
            #Aggiungo alla nuova generazione degli elementi mutati
            
            for _ in range(p_m):
                new_generation.append(Individual(sudoku))
            
            #Aggiungo alla nuova generazione i figli
            
            for _ in range (p_p):
                i = ran.randint(0,p_e-1)
                j = ran.randint(p_e, POPULATION_SIZE-1)
                offspring1, offspring2 = crossover(population[i].get_individual(),population[j].get_individual(),sudoku)
                new_generation.append(offspring1)
                new_generation.append(offspring2)
                
            #Aggiornamento della popolazione
            
            population = new_generation.copy()
            
            for i in range(POPULATION_SIZE):
                if(i < p_e_m):
                    mutation(population[i].get_individual(),population[i].get_model())
                if(i < p_e):
                    population[i].local_search()

            for i in range(p_e):
                if(population[i].get_fitness() < 15):
                    b, candidates = reduce_until_domains_safe(population[i].get_individual(),population[i].get_model(),initialize_candidates)
                    #print(b)
                    #print(candidates)
                    solution = solve_sudoku_with_dlx(b,candidates)
                    if solution["status"] == "solved":
                        print("SOLUZIONE RAFFINATA CON DLX")
                        print(np.array(solution["solution"]))
                        found = True
                        break
            
            ran.shuffle(population)
            generation+=1
            changes+=1
            
            if(changes >= THERSHOLD_RESTART):
                break

        if found==True:
            return "End"

        if(best_solution_fitness == 0):
            end_time = time.time()
            print("Numero riavvii: " + str(num_restart))
            print("Tempo di esecuzione: " + str(end_time - start_time))
            print("------------")
            print("Soluzione trovata:")
            return best_solution
        else:
            num_restart+=1
            print("Soluzione non trovata. Riavvio.")



    
sudoku_hard = np.array([ [0,0,0,6,0,0,0,9,7], [0,0,7,0,0,2,0,0,0], [5,0,0,9,0,0,0,0,1], [0,0,9,1,0,0,4,0,0], [0,0,0,0,6,0,7,8,0], [0,2,0,0,0,7,0,0,0], [0,0,3,0,0,0,1,4,2], [0,4,2,0,3,6,0,0,0],
           [0,7,0,0,2,0,6,0,0] ])    
    
sudoku_easy = np.array([ [8,0,0,3,0,5,0,4,2], [2,6,0,7,0,9,0,3,0], [1,0,3,4,2,6,9,0,8], [0,2,0,0,5,0,8,0,7], [0,3,0,9,7,0,0,1,0], [7,8,9,0,0,2,3,6,5], [6,0,5,0,0,7,1,0,0], [0,7,0,2,3,0,0,5,0],
           [0,1,2,0,6,0,7,8,9] ])

ai_escargot = np.array([ [1,0,0,0,0,7,0,9,0], [0,3,0,0,2,0,0,0,8], [0,0,9,6,0,0,5,0,0], [0,0,5,3,0,0,9,0,0], [0,1,0,0,8,0,0,0,2], [6,0,0,0,0,4,0,0,0], [3,0,0,0,0,0,0,1,0], [0,4,0,0,0,0,0,0,7], [0,0,7,0,0,0,3,0,0]   ])



#Lettura da File
input = []
# Apre il file CSV in modalità lettura
with open('easy1.csv', 'r', encoding='utf-16', newline='') as file:
    # Crea un oggetto lettore CSV
    csv_reader = csv.reader(file)
    
    # Legge ogni riga del file CSV
    for row in csv_reader:
        if '\0' not in row:
            input.append(row)            

# Converte la lista di liste in un array NumPy
sudoku = ai_escargot
#print(sudoku)
#candidates = initialize_candidates_constrained(sudoku)
#print(candidates)

#Algoritmo BRKGA (Passare alla funzione un array numpy)

print(brkga_algorithm(sudoku))


