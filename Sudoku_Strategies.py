import numpy as np
from collections import deque

def block_index(r, c, N):
    """
    Calcola l'indice del blocco kxk in cui si trova la cella (r, c) nella griglia Sudoku NxN.
    r: Indice della riga.
    c: Indice della colonna.
    N: Dimensione della griglia (es. 9 per un Sudoku 9x9, 16 per un Sudoku 16x16).
    """
    k = int(N**0.5)  # Dimensione del blocco (es. 3x3 per un Sudoku 9x9)
    return (r // k) * k + (c // k)

def initialize_candidates(grid):
    """ Restituisce una matrice di liste: 
        - Se la cella contiene 0, ha una lista con i numeri da 1 a 9.
        - Se la cella contiene un numero, ha una lista con quel numero fisso.
    """
    N = len(grid)
    candidates = [[list(range(1, N+1)) if grid[r][c] == 0 else [int(grid[r][c])] for c in range(N)] for r in range(N)]
    return candidates

def eliminate_in_row(r, num, candidates):
        #Eliminazione del valore 'num' dai candidati nella riga 'r'
        for c in range(len(candidates[r])):
            if num in candidates[r][c] and len(candidates[r][c]) > 1:
                candidates[r][c].remove(num)
    

def eliminate_in_col(c, num, candidates):
    """Elimina il numero 'num' dai candidati nella colonna 'c'."""
    for r in range(len(candidates)):
        if num in candidates[r][c] and len(candidates[r][c]) > 1:
            candidates[r][c].remove(num)

    
def eliminate_in_block(r, c, num, candidates):
    """Elimina il numero 'num' dai candidati nel blocco 3x3 contenente la cella (r, c)."""
    N = len(candidates)
    sqrt_N = int(N**0.5)
    block_row_start = (r // sqrt_N) * sqrt_N
    block_col_start = (c // sqrt_N) * sqrt_N
    for i in range(block_row_start, block_row_start + sqrt_N):
        for j in range(block_col_start, block_col_start + sqrt_N):
            if num in candidates[i][j] and len(candidates[i][j]) > 1:
                candidates[i][j].remove(num)


def propagate_for_cell(r, c, num, candidates):
        """
        Propaga i vincoli solo nelle celle della stessa riga, colonna e blocco
        della cella (r, c) dove è stato inserito 'num'.
        """
        eliminate_in_row(r, num, candidates)
        eliminate_in_col(c, num, candidates)
        eliminate_in_block(r, c, num, candidates)


def naked_single(candidates, grid):
        """
        Trova le celle che hanno un solo candidato possibile e inserisce quel numero nella griglia.
        Restituisce True se ha fatto progressi, False se non ci sono Naked Single da risolvere.
        """
        N = len(grid)
        progress = False
        for r in range(N):
            for c in range(N):
                if len(candidates[r][c]) == 1:
                    num = candidates[r][c].pop()
                    grid[r][c] = num  # Inserisci il numero nella griglia
                    propagate_for_cell(r, c, num, candidates)  # Propaga i vincoli solo sulle celle rilevanti
                    progress = True
        return progress


def initial_propagation(grid, candidates):
    """
    Propaga i vincoli eliminando candidati da ogni cella in base ai numeri già presenti
    nelle righe, colonne e blocchi.
    """
    N = len(grid)
    sqrt_N = int(N**0.5)

    for r in range(N):
        for c in range(N):
            if grid[r][c] != 0:  # Se la cella contiene già un numero
                num = grid[r][c]
                # Elimina 'num' dai candidati nella riga, colonna e blocco
                propagate_for_cell(r, c, num, candidates)
    
    return grid


#AC-3

def get_peers():
    """Restituisce un dizionario con tutte le celle collegate a ogni cella (stessa riga, colonna e regione 3x3)."""
    peers = {(r, c): set() for r in range(9) for c in range(9)}
    
    for row in range(9):
        for col in range(9):
            # Stessa riga
            peers[(row, col)].update((row, c) for c in range(9) if c != col)
            # Stessa colonna
            peers[(row, col)].update((r, col) for r in range(9) if r != row)
            # Stesso blocco 3x3
            r0, c0 = (row // 3) * 3, (col // 3) * 3
            peers[(row, col)].update((r0 + r, c0 + c) for r in range(3) for c in range(3) if (r0 + r, c0 + c) != (row, col))

    return peers

def ac3(sudoku):
    """Applica AC-3 sulla matrice del Sudoku per ridurre i domini delle celle."""
    peers = get_peers()
    queue = deque((cell, neighbor) for cell in peers for neighbor in peers[cell])
    
    while queue:
        cell, neighbor = queue.popleft()
        if revise(sudoku, cell, neighbor):
            if len(sudoku[cell[0]][cell[1]]) == 0:  # Se un dominio diventa vuoto → Sudoku impossibile
                return False
            for other in peers[cell] - {neighbor}:  # Aggiunge gli altri vincoli da controllare
                queue.append((other, cell))
    return True

def revise(sudoku, cell, neighbor):
    """Rimuove i valori inconsistenti dal dominio di una cella rispetto a un vicino."""
    row, col = cell
    neighbor_row, neighbor_col = neighbor
    removed = False

    # Se il vicino ha un solo valore, rimuovilo dalla cella corrente
    if len(sudoku[neighbor_row][neighbor_col]) == 1:
        unique_value = sudoku[neighbor_row][neighbor_col][0]
        if unique_value in sudoku[row][col]:
            sudoku[row][col].remove(unique_value)
            removed = True
    return removed

