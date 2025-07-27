/*
 * Eternity II - Solução Paralela com MPI
 * 
 * Implementação paralela do quebra-cabeça Eternity II usando MPI.
 * O algoritmo utiliza backtracking para encontrar soluções, com estratégia
 * de paralelização baseada em peças de quina diferentes.
 * 
 * Estratégia de paralelização:
 * 1. Identifica todas as peças que podem ser colocadas nas quinas
 * 2. Distribui quinas diferentes para processos MPI diferentes
 * 3. Cada processo tenta resolver o puzzle com sua quina específica
 * 4. Comunicação assíncrona para parada antecipada quando solução é encontrada

 * Uso de IA para identificar peças de quina, para implementação do MPI_Iprobe
 * e para documentação do código pelo modelo Claude 4 sonnet
 *  - Prompt para função de identificar peças de quina: Gostaria de você me ajudasse 
 *  a identificar todas as peças que poderiam ser colocadas nas quinas do tabuleiro
 *  e retornr um int 1 se a peça pode ser colocada na quina e 0 caso contrário.
 *  - Prompt para implementação do MPI_Iprobe na função play: Tenho um algoritmo 
 *  de backtracking paralelo onde cada processo MPI tenta resolver um puzzle. 
 *  Quando um processo encontra solução, ele precisa avisar os outros para pararem.
 *  Gostaria de você me ajudar a implementar a função MPI para que o processo
 *  que encontrou solução possa avisar os outros processos para pararem.
 *  - Prompt para documentação do código: Gostaria de você me ajudar a documentar
 *  o código de forma clara e objetiva, com comentários e explicações de como
 *  cada função funciona.
 
 */

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <mpi.h>
#include <string.h>
#include <time.h>
#include <sys/time.h>

/* 
 * Estrutura que representa uma peça do puzzle
 * colors[4]: cores das 4 bordas (Norte, Leste, Sul, Oeste)
 * rotation: rotação atual da peça (0-3)
 * id: identificador único da peça
 * used: flag indicando se a peça já foi colocada no tabuleiro
 */
typedef struct {
    unsigned int colors[4];
    unsigned char rotation;
    unsigned int id;
    int used;
} tile;

/* 
 * Macros para acessar cores das bordas considerando rotação
 * X_COLOR: cor genérica da borda s considerando rotação
 * N_COLOR, E_COLOR, S_COLOR, W_COLOR: cores específicas das bordas
 */
#define X_COLOR(t, s) (t->colors[(s + 4 - t->rotation) % 4])
#define N_COLOR(t) (X_COLOR(t, 0))  // Norte
#define E_COLOR(t) (X_COLOR(t, 1))  // Leste
#define S_COLOR(t) (X_COLOR(t, 2))  // Sul
#define W_COLOR(t) (X_COLOR(t, 3))  // Oeste

/*
 * Função auxiliar para medição de tempo
 * Retorna tempo atual em segundos com precisão de microssegundos
 */
double get_time() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec / 1000000.0;
}

/*
 * Estrutura principal do jogo
 * size: dimensão do tabuleiro (size x size)
 * tile_count: número total de peças (= size²)
 * board: matriz de ponteiros para peças colocadas no tabuleiro
 * tiles: array com todas as peças disponíveis
 */
typedef struct {
    unsigned int size;
    unsigned int tile_count;
    tile ***board;
    tile *tiles;
} game;

/*
 * Estrutura para informações de peças de quina
 * tile_id: ID da peça que pode ser quina
 * rotation: rotação necessária para colocar na quina
 * corner_type: tipo de quina (0=sup_esq, 1=sup_dir, 2=inf_esq, 3=inf_dir)
 */
typedef struct {
    int tile_id;           
    unsigned char rotation;
    int corner_type;       
} corner_info;

/* Variáveis globais MPI e controle de execução */
int rank, size;                    // Rank e tamanho do comunicador MPI
double time_init, time_end;        // Variáveis para medição de tempo
int global_stop = 0;               // Flag para parada global
int global_solution_found = 0;     // Flag indicando se solução foi encontrada
int solution_owner = -1;           // Rank do processo que encontrou a solução

/*
 * Inicializa o jogo a partir de entrada padrão
 * Lê dimensão do tabuleiro, número de cores e todas as peças
 * Retorna ponteiro para estrutura game alocada dinamicamente
 */
game *initialize(FILE *input) {
    unsigned int bsize;
    unsigned int ncolors;
    
    // Lê dimensões e valida entrada
    int r = fscanf(input, "%u", &bsize);
    assert(r == 1);
    r = fscanf(input, "%u", &ncolors);
    assert(r == 1);
    assert(ncolors < 256);

    // Aloca estrutura principal do jogo
    game *g = malloc(sizeof(game));
    g->size = bsize;
    g->tile_count = bsize * bsize;
    
    // Aloca tabuleiro como matriz de ponteiros
    g->board = malloc(sizeof(tile**) * bsize);
    for(int i = 0; i < bsize; i++)
        g->board[i] = calloc(bsize, sizeof(tile*));

    // Aloca e inicializa array de peças
    g->tiles = malloc(g->tile_count * sizeof(tile));
    for (unsigned int i = 0; i < g->tile_count; i++) {
        g->tiles[i].rotation = 0;
        g->tiles[i].id = i;
        g->tiles[i].used = 0;
        
        // Lê cores das 4 bordas de cada peça
        for (int c = 0; c < 4; c++) {
            r = fscanf(input, "%u", &g->tiles[i].colors[c]);
            assert(r == 1);
        }
    }
    return g;
}

/*
 * Libera toda a memória alocada para o jogo
 */
void free_resources(game *game) {
    for(int i = 0; i < game->size; i++)
        free(game->board[i]);
    free(game->board);
    free(game->tiles);
    free(game);
}

/*
 * Verifica se uma peça pode ser colocada em uma posição específica
 * Valida bordas do tabuleiro (devem ter cor 0) e compatibilidade com vizinhos
 * Retorna 1 se movimento é válido, 0 caso contrário
 */
int valid_move(game *game, unsigned int x, unsigned int y, tile *tile) {
    // Verifica bordas do tabuleiro (devem ter cor 0)
    if (x == 0 && W_COLOR(tile) != 0) return 0;                    // Borda oeste
    if (y == 0 && N_COLOR(tile) != 0) return 0;                    // Borda norte
    if (x == game->size - 1 && E_COLOR(tile) != 0) return 0;       // Borda leste
    if (y == game->size - 1 && S_COLOR(tile) != 0) return 0;       // Borda sul

    // Verifica compatibilidade com vizinhos já colocados
    if (x > 0 && game->board[x - 1][y] != NULL &&
        E_COLOR(game->board[x - 1][y]) != W_COLOR(tile))
        return 0;
    if (x < game->size - 1 && game->board[x + 1][y] != NULL &&
        W_COLOR(game->board[x + 1][y]) != E_COLOR(tile))
        return 0;
    if (y > 0 && game->board[x][y - 1] != NULL &&
        S_COLOR(game->board[x][y - 1]) != N_COLOR(tile))
        return 0;
    if (y < game->size - 1 && game->board[x][y + 1] != NULL &&
        N_COLOR(game->board[x][y + 1]) != S_COLOR(tile))
        return 0;

    return 1;
}

/*
 * Verifica se uma peça pode ser colocada em alguma quina do tabuleiro
 * Testa todas as 4 rotações para encontrar configuração válida
 * Retorna 1 se é peça de quina, 0 caso contrário
 * Parâmetros de saída: rotacao_valida e tipo_canto
 */
int eh_peca_de_quina(tile *t, unsigned char *rotacao_valida, int *tipo_canto) {
    for (int rot = 0; rot < 4; rot++) {
        t->rotation = rot;
        
        // Quina superior esquerda (Norte=0, Oeste=0)
        if (N_COLOR(t) == 0 && W_COLOR(t) == 0) {
            *rotacao_valida = rot;
            *tipo_canto = 0;
            return 1;
        }
        // Quina superior direita (Norte=0, Leste=0)
        if (N_COLOR(t) == 0 && E_COLOR(t) == 0) {
            *rotacao_valida = rot;
            *tipo_canto = 1;
            return 1;
        }
        // Quina inferior esquerda (Sul=0, Oeste=0)
        if (S_COLOR(t) == 0 && W_COLOR(t) == 0) {
            *rotacao_valida = rot;
            *tipo_canto = 2;
            return 1;
        }
        // Quina inferior direita (Sul=0, Leste=0)
        if (S_COLOR(t) == 0 && E_COLOR(t) == 0) {
            *rotacao_valida = rot;
            *tipo_canto = 3;
            return 1;
        }
    }
    return 0;
}

/*
 * Identifica todas as peças que podem ser colocadas nas quinas
 * Retorna array dinâmico com informações das peças de quina
 * Parâmetro de saída: num_corner_pieces (número de peças encontradas)
 */
corner_info* separar_pecas_de_quina(game *g, int *num_corner_pieces) {
    corner_info *corners = malloc(g->tile_count * sizeof(corner_info));
    *num_corner_pieces = 0;
    
    if (rank == 0) {
        printf("=== Identificando peças de quina ===\n");
    }
    
    // Testa cada peça para ver se pode ser quina
    for (int i = 0; i < g->tile_count; i++) {
        tile *current_tile = &g->tiles[i];
        unsigned char rotacao_original = current_tile->rotation;
        unsigned char rotacao_valida;
        int tipo_canto;
        
        if (eh_peca_de_quina(current_tile, &rotacao_valida, &tipo_canto)) {
            corners[*num_corner_pieces].tile_id = current_tile->id;
            corners[*num_corner_pieces].rotation = rotacao_valida;
            corners[*num_corner_pieces].corner_type = tipo_canto;
            
            if (rank == 0) {
                printf("Peça ID %u pode ser quina tipo %d com rotação %u\n", 
                       current_tile->id, tipo_canto, rotacao_valida);
            }
            
            (*num_corner_pieces)++;
        }
        
        // Restaura rotação original
        current_tile->rotation = rotacao_original;
    }
    
    if (rank == 0) {
        printf("Total de peças de quina: %d\n\n", *num_corner_pieces);
    }
    
    // Redimensiona array para o tamanho exato
    if (*num_corner_pieces > 0) {
        corners = realloc(corners, (*num_corner_pieces) * sizeof(corner_info));
    } else {
        free(corners);
        corners = NULL;
    }
    
    return corners;
}

/*
 * Algoritmo principal de backtracking para resolver o puzzle
 * Utiliza comunicação MPI assíncrona para parada antecipada
 * Retorna 1 se encontrou solução, 0 caso contrário
 */
int play(game *game, unsigned int x, unsigned int y) {
    // Verifica flag de parada global
    if (global_stop) return 0;
    
    // Verificação periódica de mensagens de outros processos
    static int check_counter = 0;
    check_counter++;
    
    if (check_counter % 1000 == 0) {
        int flag;
        MPI_Status status;
        MPI_Iprobe(MPI_ANY_SOURCE, 999, MPI_COMM_WORLD, &flag, &status);
        
        if (flag) {
            int stop_signal;
            MPI_Recv(&stop_signal, 1, MPI_INT, status.MPI_SOURCE, 999, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            global_solution_found = 1;
            solution_owner = stop_signal;
            return 0; // Para o backtracking imediatamente
        }
    }
    
    // Para se outro processo já encontrou solução
    if (global_solution_found) {
        return 0;
    }

    // Tenta cada peça disponível na posição atual
    for (int i = 0; i < game->tile_count; i++) {
        if (game->tiles[i].used) continue;
        
        tile *tile = &game->tiles[i];
        tile->used = 1;
        
        // Testa todas as 4 rotações da peça
        for (int rot = 0; rot < 4; rot++) {
            tile->rotation = rot;
            
            if (valid_move(game, x, y, tile)) {
                game->board[x][y] = tile;
                
                // Calcula próxima posição a preencher
                unsigned int nx, ny;
                ny = nx = game->size;
                if (x < game->size - 1) {
                    nx = x + 1;
                    ny = y;
                } else if (y < game->size - 1) {
                    nx = 0;
                    ny = y + 1;
                }
                
                // Se completou tabuleiro ou recursão encontrou solução
                if (ny == game->size || play(game, nx, ny)) {
                    global_stop = 1; // Sinaliza parada global
                    
                    // Envia sinal de parada para todos os outros processos
                    for (int p = 0; p < size; p++) {
                        if (p != rank) {
                            MPI_Send(&rank, 1, MPI_INT, p, 999, MPI_COMM_WORLD);
                        }
                    }
                    return 1;
                }
                
                // Remove peça do tabuleiro (backtrack)
                game->board[x][y] = NULL;
            }
        }
        tile->used = 0;
    }
    return 0;
}

/*
 * Imprime a solução encontrada no formato esperado
 */
void print_solution(game *game) {
    printf("\n=== SOLUÇÃO ENCONTRADA ===\n");
    for(unsigned int j = 0; j < game->size; j++) {
        for(unsigned int i = 0; i < game->size; i++) {
            tile *t = game->board[i][j];
            printf("%u %u\n", t->id, t->rotation);
        }
    }
    printf("=========================\n");
}

/*
 * Função principal - coordena a execução paralela
 * Implementa estratégia de paralelização baseada em distribuição de quinas
 */
int main(int argc, char **argv) {
    // Inicialização MPI
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    game *g = NULL;
    corner_info *corners = NULL;
    int num_corners = 0;
    int solution_found = 0;
    
    // Processo 0 inicializa o jogo e identifica quinas
    if (rank == 0) {
        g = initialize(stdin);
        printf("Tabuleiro: %ux%u, %u peças\n", g->size, g->size, g->tile_count);
        corners = separar_pecas_de_quina(g, &num_corners);
    }
    
    // Broadcast do número de quinas para todos os processos
    MPI_Bcast(&num_corners, 1, MPI_INT, 0, MPI_COMM_WORLD);
    
    // Verifica se há quinas disponíveis
    if (num_corners == 0) {
        if (rank == 0) printf("Nenhuma quina encontrada!\n");
        MPI_Finalize();
        return 1;
    }
    
    // Otimização: limita número de processos ativos ao número de quinas
    int effective_processes = (size < num_corners) ? size : num_corners;
    int active = (rank < effective_processes);
    
    if (rank == 0) {
        printf("Eternity II Paralelo - %d processos (%d ativos para %d quinas)\n", 
               size, effective_processes, num_corners);
    }
    
    // Apenas processos ativos participam da resolução
    if (active) {
        // Processos não-zero recebem dados do jogo via broadcast
        if (rank != 0) {
            unsigned int game_size, tile_count;
            MPI_Bcast(&game_size, 1, MPI_UNSIGNED, 0, MPI_COMM_WORLD);
            MPI_Bcast(&tile_count, 1, MPI_UNSIGNED, 0, MPI_COMM_WORLD);
            
            // Aloca estruturas de dados locais
            g = malloc(sizeof(game));
            g->size = game_size;
            g->tile_count = tile_count;
            g->board = malloc(sizeof(tile**) * game_size);
            for(int i = 0; i < game_size; i++)
                g->board[i] = calloc(game_size, sizeof(tile*));
            g->tiles = malloc(tile_count * sizeof(tile));
            
            corners = malloc(num_corners * sizeof(corner_info));
        } else {
            MPI_Bcast(&g->size, 1, MPI_UNSIGNED, 0, MPI_COMM_WORLD);
            MPI_Bcast(&g->tile_count, 1, MPI_UNSIGNED, 0, MPI_COMM_WORLD);
        }
        
        // Broadcast das peças e informações de quinas
        MPI_Bcast(g->tiles, g->tile_count * sizeof(tile), MPI_BYTE, 0, MPI_COMM_WORLD);
        MPI_Bcast(corners, num_corners * sizeof(corner_info), MPI_BYTE, 0, MPI_COMM_WORLD);
        
        // Cada processo ativo trabalha com uma quina diferente
        int corner_idx = rank;
        
        // Inicializa tabuleiro vazio
        for (int i = 0; i < g->size; i++) {
            for (int j = 0; j < g->size; j++) {
                g->board[i][j] = NULL;
            }
        }
        
        for (int i = 0; i < g->tile_count; i++) {
            g->tiles[i].used = 0;
        }
        
        // Determina posição da quina baseada no tipo
        unsigned int corner_x = 0, corner_y = 0;
        switch(corners[corner_idx].corner_type) {
            case 0: corner_x = 0; corner_y = 0; break;                    // Superior esquerda
            case 1: corner_x = g->size-1; corner_y = 0; break;            // Superior direita
            case 2: corner_x = 0; corner_y = g->size-1; break;            // Inferior esquerda
            case 3: corner_x = g->size-1; corner_y = g->size-1; break;    // Inferior direita
        }
        
        // Coloca a peça de quina no tabuleiro
        tile *corner_tile = &g->tiles[corners[corner_idx].tile_id];
        corner_tile->rotation = corners[corner_idx].rotation;
        corner_tile->used = 1;
        g->board[corner_x][corner_y] = corner_tile;
        
        // Inicia medição de tempo
        double start_time = MPI_Wtime();
        
        // Encontra primeira posição vazia para começar backtracking
        unsigned int start_x = g->size, start_y = g->size;
        for (unsigned int y = 0; y < g->size && start_y == g->size; y++) {
            for (unsigned int x = 0; x < g->size; x++) {
                if (g->board[x][y] == NULL) {
                    start_x = x;
                    start_y = y;
                    break;
                }
            }
        }
        
        // Executa algoritmo de backtracking
        int local_solution = 0;
        if (start_y < g->size) {
            local_solution = play(g, start_x, start_y);
        }
        
        double end_time = MPI_Wtime();
        
        // Sincronização apenas entre processos ativos
        MPI_Comm active_comm;
        MPI_Comm_split(MPI_COMM_WORLD, 1, rank, &active_comm);
        
        // Verifica se algum processo encontrou solução
        int global_result;
        MPI_Allreduce(&local_solution, &global_result, 1, MPI_INT, MPI_MAX, active_comm);
        
        if (global_result) {
            solution_found = 1;
            
            // Determina qual processo encontrou a solução
            int winner_rank = -1;
            int has_solution = local_solution ? rank : effective_processes;
            MPI_Allreduce(&has_solution, &winner_rank, 1, MPI_INT, MPI_MIN, active_comm);
            
            // Imprime resultado
            if (local_solution && rank == winner_rank) {
                printf("Processo %d encontrou solução e vai imprimi-la!\n", rank);
                printf("Tempo de execução: %.6f segundos\n", end_time - start_time);
                print_solution(g);
            } else if (rank == 0 && winner_rank != 0) {
                printf("Processo %d encontrou a solução em %.3f segundos.\n", 
                       winner_rank, end_time - start_time);
                printf("Processo %d parou busca antecipadamente.\n", rank);
            }
        }
        
        MPI_Comm_free(&active_comm);
    }
    
    // Broadcast do resultado final para todos os processos
    MPI_Bcast(&solution_found, 1, MPI_INT, 0, MPI_COMM_WORLD);
    
    if (!solution_found && rank == 0) {
        printf("\nSOLUÇÃO NÃO ENCONTRADA\n");
    }
    
    // Liberação de recursos
    if (corners) free(corners);
    if (g) free_resources(g);
    
    MPI_Finalize();
    return solution_found ? 0 : 1;
} 
