import numpy as np
import csv
import os
from Sudoku_Strategies import *

N = 9
MAX_DEPTH = 512

def create_exact_cover_matrix(candidates):
    """
    Crea una matrice di incidenza per un Sudoku di dimensione N x N
    considerando i candidati possibili per ogni cella.
    
    Parametri:
    candidates -- Matrice di liste con i candidati per ogni cella.
    
    Ritorna:
    matrix -- Matrice di incidenza ridotta (numpy array)
    """

    N = len(candidates)
    k = int(N**0.5)
    num_columns = 4 * N * N

    # Inizialmente crea una lista vuota per la matrice incidente
    rows = []

    def cell_constraint(row, col):
        return row * N + col

    def row_constraint(row, num):
        return N * N + row * N + (num - 1)

    def col_constraint(col, num):
        return 2 * N * N + col * N + (num - 1)

    def block_constraint(row, col, num):
        block = (row // k) * k + (col // k)
        return 3 * N * N + block * N + (num - 1)

    # Costruisce solo righe consentite dai candidati
    for r in range(N):
        for c in range(N):
            for num in candidates[r][c]:  # usa candidati reali
                row_cover = np.zeros(num_columns, dtype=int)
                row_cover[cell_constraint(r, c)] = 1
                row_cover[row_constraint(r, num)] = 1
                row_cover[col_constraint(c, num)] = 1
                row_cover[block_constraint(r, c, num)] = 1
                rows.append(row_cover)

    # Converte la lista finale in numpy array
    matrix = np.array(rows)

    return matrix



class Node:
    def __init__(self):
        self.left = self.right = self.up = self.down = self  # Puntatori circolari
        self.column = None  # Colonna di appartenenza (colonna del vincolo)
    
    def __repr__(self):
        return f"<Node: Colonna={self.column.name}>"

class ColumnNode(Node):
    def __init__(self, name):
        super().__init__()  # Chiama il costruttore della classe Node
        self.size = 0  # Contatore del numero di nodi nella colonna
        self.name = name  # Nome della colonna per identificare il vincolo

    def __repr__(self):
        return f"<ColumnNode: {self.name}, Size={self.size}>"

class DancingLinks:

    def __init__(self, matrix):
        self.columns = []
        self.header = self.build_linked_structure(matrix)
    
    def build_linked_structure(self, matrix):

        header = ColumnNode("header")  # Nodo header che collega tutte le colonne
        last = header  # Mantieni traccia dell'ultimo nodo aggiunto (inizia dall'header)

        #print("Inizio costruzione delle colonne...")
        for col_idx in range(len(matrix[0])):
            column_node = ColumnNode(col_idx)
            self.columns.append(column_node)

            # Collega l'ultimo nodo alla nuova colonna
            last = self.link_right(last, column_node)

            '''#Stampa di debug per verificare il collegamento
            print(f"Colonna {col_idx} aggiunta. Collegata a sinistra con {last.left.name}")

        # Verifica del collegamento ciclico
        print(f"L'ultima colonna ({last.name}) è collegata a destra con {last.right.name}")
        print(f"L'header è collegato a sinistra con {header.left.name}")    '''    
        
        #Crea i nodi della matrice
        for row_idx, row in enumerate(matrix):
            prev = None #Nodo precedente per il collegamento orizzontale
            for col_idx, cell in enumerate(row):
                if cell == 1: 
                    col_node = self.columns[col_idx]
                    new_node = Node()
                    new_node.column = col_node
                    col_node.size += 1

                    #Collega verticalmente nella colonna
                    self.link_down(col_node, new_node)

                    #Collega orizzontalmente nella riga
                    if prev:
                        prev = self.link_right(prev,new_node)
                    else:
                        prev = new_node

        return header
    

    def cover_fixed_numbers(self, grid):
        """
        Copre i vincoli relativi alle celle già riempite nella griglia Sudoku.
        grid: La griglia Sudoku con i numeri fissi (valori diversi da 0).
        dlx: L'oggetto Dancing Links.
        """
        N = len(grid)  # Dimensione della griglia
        for r in range(N):
            for c in range(N):
                if grid[r][c] != 0:  # Se la cella è già riempita
                    num = grid[r][c]
                    # Calcola gli indici per i vincoli
                    cell_constraint = r * N + c
                    row_constraint = N * N + r * N + (num - 1)
                    col_constraint = 2 * N * N + c * N + (num - 1)
                    block_constraint = 3 * N * N + block_index(r, c, N) * N + (num - 1)
                
                    # Copri i vincoli relativi a questa mossa
                    self.cover(self.columns[cell_constraint])
                    self.cover(self.columns[row_constraint])
                    self.cover(self.columns[col_constraint])
                    self.cover(self.columns[block_constraint])

    

    def link_right(self, node1, node2):
        """
        Collega node2 a destra di node1 e collega node1 a sinistra di node2.
        """
        node2.left = node1  # node2 è collegato a sinistra con node1
        node2.right = node1.right  # node2.right punta al nodo che era a destra di node1
        node1.right.left = node2  # Aggiorna il nodo che era a destra di node1 per puntare a node2 come suo sinistro
        node1.right = node2  # Collega node1 a destra con node2

        return node2
    

    def link_down(self, node1, node2):
        
        # Collego il nodo2 sotto il nodo1

        node2.up = node1
        node2.down = node1.down
        node1.down.up = node2
        node1.down = node2

        return node2
    
    def cover(self, col):

        #Rimuovere la colonna 'col' dalla lista circolare delle colonne
        col.left.right = col.right
        col.right.left = col.left

        #Scorrere in verticale la colonna 'col'
        row = col.down
        while row != col:
            node = row.right
            while node != row:
                node.up.down = node.down
                node.down.up = node.up
                node = node.right #Successivo nodo nella riga
            
            row = row.down #Riga successiva nella colonna


    def uncover(self, col):
        
        #Ripristina la colonna 'col' e tutte le righe associate, riportandole in Dancing Links

        row = col.up
        while row != col:
            
            node = row.left
            while node != row:
                node.up.down = node
                node.down.up = node
                node = node.left #Nodo precedente nella riga
            
            row = row.up #Riga precedente nella colonna

        #Ripristino della colonna 'col' nella lista circolare delle colonne
        col.left.right = col
        col.right.left = col

    
    def select_column(self):

        min_col = None
        min_size = float('inf')
        col = self.header.right

        while col != self.header:
            if col.size < min_size:
                min_size = col.size
                min_col = col
            col = col.right

        return min_col

    def right_column_iterator(self):
        #Iteratore per muoversi tra i nodi di colonna nel livello superiore.
        current = self.header.right
        while current != self.header:
            yield current
            current = current.right


    def algorithm_x(self, solution=[], depth = 0):

        #Se non ci sono colonne da coprire la soluzione è stata trovata
        if self.header.right == self.header:
            return {"solution": solution, "depth": depth, "status": "solved"}
        
        if depth > MAX_DEPTH:
            return {"solution": None, "depth": depth, "status": "depth_limit"}

        col = self.select_column() #Euristica da implementare per scegliere una colonna

        if col.down == col:
            return {"solution": None, "depth": depth, "status": "dead_end"}  # Nessuna riga possibile

        #Coprire la colonna selezionata precedentemente
        self.cover(col)

        # Scorrere attraverso le righe che coprono questa colonna

        row = col.down

        best_attempt = {"solution": None, "depth": depth, "status": "dead_end"}


        while row != col:
            solution.append(row)
        
            #Coprire tutte le altre colonne che sono coperta da questa riga

            node = row.right
            while node != row:
                self.cover(node.column)
                node = node.right
        
            #Ricorsione: cerca di risolvere il problema con questa scelta

            result = self.algorithm_x(solution, depth+1)
            if result["status"] == "solved":
                return result  # soluzione completa trovata
            elif result["depth"] > best_attempt["depth"]:
                best_attempt = result  # salva il migliore tentativo
        
            #Backtracking: Se una soluzione non è stata trovata, rimuovi l'ultima riga e scopri le colonne associate
            solution.pop()
            node = row.left
            while node != row:
                self.uncover(node.column)
                node = node.left

            #Riga successiva
            row = row.down
        
        #Se nessuna soluzione è stata trovata, scopri la colonna e torna indietro
        self.uncover(col)
        return best_attempt

    
    def reconstruct_solution(self, solution, grid):
        """
        Ricostruisce la soluzione del Sudoku a partire dalle righe della soluzione di Dancing Links.
        solution: La lista delle righe che rappresentano la soluzione trovata.
        grid: La griglia Sudoku iniziale, con alcune celle già riempite.
        N: Dimensione della griglia Sudoku (es. 9 per un Sudoku 9x9).
        """
        # Crea una griglia vuota per la soluzione
        solved_grid = [row[:] for row in grid]  # Copia la griglia originale

        for row in solution:
            # Ogni riga nella soluzione corrisponde a una mossa valida, cioè
            # (r, c, num) che indica che il numero 'num' è inserito nella cella (r, c).

            r = c = num = None  # Inizializza riga, colonna e numero

            # Scorri i nodi collegati nella riga e trova le informazioni rilevanti
            node = row
            while True:  # Itera fino a completare la lista circolare
                col_name = node.column.name

                # Identifica le informazioni sulla mossa (r, c, num)
                if col_name < N * N:  # Vincolo della cella (r, c)
                    r = col_name // N
                    c = col_name % N
                elif N * N <= col_name < 2 * N * N:  # Vincolo della riga
                    num = (col_name - N * N) % N + 1  # Numero da inserire nella cella

                # Passa al nodo successivo nella riga
                node = node.right

                # Esci dal ciclo se torni al nodo di partenza (completa la lista circolare)
                if node == row:
                    break

            # Inserisci il numero nella cella corrispondente
            solved_grid[r][c] = num

        return solved_grid


def solve_sudoku_with_dlx(grid, candidates):
    """
    Risolvi il Sudoku usando Dancing Links e Exact Cover.
    grid: La griglia iniziale del Sudoku (con alcuni numeri già riempiti).
    candidates: La matrice dei candidati possibili per ogni cella.
    """

    incident_matrix = create_exact_cover_matrix(candidates)

    dlx = DancingLinks(incident_matrix)  # Costruisci la struttura Dancing Links

    # Copre i vincoli per le celle già riempite nella griglia
    dlx.cover_fixed_numbers(grid)

    # Esegue l'algoritmo DLX per cercare la soluzione
    solution = dlx.algorithm_x()
    # Ricostruisce la griglia finale del Sudoku dalla soluzione trovata
    if solution["status"] == "solved":
        return {"solution": dlx.reconstruct_solution(solution["solution"],grid), "depth": solution["depth"], "status": "solved"}
    else:
        #print(f"Nessuna soluzione trovata. Profondità raggiunta: {solution['depth']}")
        return solution


def test_dancing_links():

    matrix = create_exact_cover_matrix(N)

    # Crea la struttura Dancing Links
    dlx = DancingLinks(matrix)

    # Verifica il collegamento delle colonne (livello orizzontale)
    current_col = dlx.header.right
    print("Verifica collegamenti delle colonne:")
    while current_col != dlx.header:
        print(f"Colonna {current_col.name}: collegata a sinistra con {current_col.left.name}, destra con {current_col.right.name}")
        current_col = current_col.right
    
    #Verifica il collegamento circolare delle colonne
    last_column = current_col.left  # Ultima colonna della lista
    first_column = dlx.header.right  # Prima colonna della lista
    print(f"Ultima colonna: {last_column.name}, collegata a destra con {last_column.right.name}")
    print(f"Prima colonna: {first_column.name}, collegata a sinistra con {first_column.left.name}")

    DancingLinks.right_column_iterator = right_column_iterator
    # Verifica il collegamento delle righe (livello verticale per ciascuna colonna)
    print("\nVerifica collegamenti delle righe (verticale):")
    for col_idx, column_node in enumerate(dlx.right_column_iterator()):
        print(f"Colonna {col_idx}:")
        current_node = column_node.down
        while current_node != column_node:
            print(f"  Nodo della riga {current_node}: collegato sopra a {current_node.up}, sotto a {current_node.down}")
            current_node = current_node.down
    
'''

# Esegui il test
#test_dancing_links()

sudoku_hard = np.array([ [0,0,0,6,0,0,0,9,7], [0,0,7,0,0,2,0,0,0], [5,0,0,9,0,0,0,0,1], [0,0,9,1,0,0,4,0,0], [0,0,0,0,6,0,7,8,0], [0,2,0,0,0,7,0,0,0], [0,0,3,0,0,0,1,4,2], [0,4,2,0,3,6,0,0,0],
           [0,7,0,0,2,0,6,0,0] ])    
    
sudoku_easy = np.array([ [8,0,0,3,0,5,0,4,2], [2,6,0,7,0,9,0,3,0], [1,0,3,4,2,6,9,0,8], [0,2,0,0,5,0,8,0,7], [0,3,0,9,7,0,0,1,0], [7,8,9,0,0,2,3,6,5], [6,0,5,0,0,7,1,0,0], [0,7,0,2,3,0,0,5,0],
           [0,1,2,0,6,0,7,8,9] ])

ai_escargot = np.array([ [1,0,0,0,0,7,0,9,0], [0,3,0,0,2,0,0,0,8], [0,0,9,6,0,0,5,0,0], [0,0,5,3,0,0,9,0,0], [0,1,0,0,8,0,0,0,2], [6,0,0,0,0,4,0,0,0], [3,0,0,0,0,0,0,1,0], [0,4,0,0,0,0,0,0,7], [0,0,7,0,0,0,3,0,0]   ])

#Lettura da File
input = []
# Apre il file CSV in modalità lettura
with open('board_Hard_0.csv', 'r', newline='') as file:
    # Crea un oggetto lettore CSV
    csv_reader = csv.reader(file)
    
    # Legge ogni riga del file CSV
    for row in csv_reader:
        if '\0' not in row:
            input.append(row)            

# Converte la lista di liste in un array NumPy
sudoku = sudoku_hard

print(sudoku)

print("-------------------")

candidates = initialize_candidates(sudoku)

print(candidates)

sudoku = initial_propagation(sudoku, candidates)

print("-----------------")
print(candidates)
print("------------------")

# Applica AC-3
if ac3(candidates):
    print("Sudoku ridotto dopo AC-3:")
    print(candidates)

output = solve_sudoku_with_dlx(sudoku, candidates)

print("-------------------")

if output != None:
    print("OUT")
    matrice = np.array(output["solution"])
    print(matrice)
'''