import java.util.ArrayList;

/**
 * Implementação do Algoritmo Simplex (Fase 1 e Fase 2)
 */
public class SimplexTrabalho {

    // eps pra nao dar erro de precisão de ponto flutuante do double
    static final double EPSILON = 0.0000001; 

    /**
     * Classe que representa o dicionário (ou tableau) do Simplex em cada iteração.
     */
    static class Tabela {
        ArrayList<Integer> N; // Variaveis não-básicas (índices)
        ArrayList<Integer> B; // Variaveis básicas (índices)
        double[][] A;         // Matriz de coeficientes das restrições
        double[] b;           // Vetor de termos independentes (lado direito)
        double[] c;           // Vetor de coeficientes da função objetivo
        double z;             // Valor atual da função objetivo
        int tam;              // Tamanho total (variáveis originais + folgas)

        Tabela(ArrayList<Integer> N, ArrayList<Integer> B, double[][] A, double[] b, double[] c, double z, int tam) {
            this.N = new ArrayList<>(N);
            this.B = new ArrayList<>(B);
            this.tam = tam;
            
            // Copiando as matrizes na mão pra não ter problema de referência de memória
            this.A = new double[tam][tam];
            for (int i = 0; i < tam; i++) {
                for (int j = 0; j < tam; j++) {
                    this.A[i][j] = A[i][j];
                }
            }
            this.b = b.clone();
            this.c = c.clone();
            this.z = z;
        }
    }

    /**
     * Realiza a operação de pivoteamento do Simplex.
     * Troca uma variável básica por uma não-básica e atualiza a matriz (Eliminação de Gauss-Jordan).
     * 
     * @param t     A tabela atual do Simplex
     * @param sai   Índice da variável que vai sair da base
     * @param entra Índice da variável que vai entrar na base
     * @return      Uma nova Tabela atualizada após o pivoteamento
     */
    static Tabela pivoteamento(Tabela t, int sai, int entra) {
        int sz = t.tam;
        double[][] novaA = new double[sz][sz];
        double[] novoB = new double[sz];
        double[] novoC = new double[sz];

        // --- ATUALIZAÇÃO DA LINHA DO PIVÔ ---
        // A nova linha do pivô é a linha antiga dividida pelo elemento pivô
        novoB[entra] = t.b[sai] / t.A[sai][entra];
        for (int j : t.N) {
            if (j != entra) {
                novaA[entra][j] = t.A[sai][j] / t.A[sai][entra];
            }
        }
        // O coeficiente da variável que saiu em relação a que entrou
        novaA[entra][sai] = 1.0 / t.A[sai][entra];

        // --- ATUALIZAÇÃO DAS OUTRAS LINHAS ---
        for (int i : t.B) {
            if (i != sai) {
                // Nova linha = Linha antiga - (coeficiente da coluna do pivo * nova linha do pivo)
                novoB[i] = t.b[i] - t.A[i][entra] * novoB[entra];
                for (int j : t.N) {
                    if (j != entra) {
                        novaA[i][j] = t.A[i][j] - t.A[i][entra] * novaA[entra][j];
                    }
                }
                novaA[i][sai] = -t.A[i][entra] * novaA[entra][sai];
            }
        }

        // --- ATUALIZAÇÃO DA FUNÇÃO OBJETIVO (Z) E VETOR C ---
        // Z novo = Z antigo + (coeficiente da variavel que entra * novo valor de b)
        double novoZ = t.z + t.c[entra] * novoB[entra];
        for (int j : t.N) {
            if (j != entra) {
                novoC[j] = t.c[j] - t.c[entra] * novaA[entra][j];
            }
        }
        novoC[sai] = -t.c[entra] * novaA[entra][sai];

        // --- ATUALIZA AS LISTAS DE VARIÁVEIS ---
        ArrayList<Integer> novoN = new ArrayList<>(t.N);
        // Tem q castar pra Integer senão o Java acha que é pra remover na posição (index) e não o valor
        novoN.remove((Integer) entra); 
        novoN.add(sai);

        ArrayList<Integer> novoB_list = new ArrayList<>(t.B);
        novoB_list.remove((Integer) sai);
        novoB_list.add(entra);

        return new Tabela(novoN, novoB_list, novaA, novoB, novoC, novoZ, sz);
    }

    /**
     * Executa o loop principal do Simplex (Fase 2) até encontrar a solução ótima.
     * 
     * @param t A tabela inicial já viabilizada
     * @return  A tabela com a solução ótima, ou null se for ilimitado
     */
    static Tabela rodarSimplex(Tabela t) {
        double[] delta = new double[t.tam];

        while (true) {
            int entra = -1;
            // Procura quem vai entrar na base (procuramos o primeiro c positivo porque é maximização)
            for (int j : t.N) {
                if (t.c[j] > EPSILON) {
                    entra = j;
                    break; 
                }
            }
            
            // Se nenhum c é positivo, chegamos no ótimo! Para o loop.
            if (entra == -1) return t;

            // Calcula o Teste da Razão (b / coeficiente na coluna da variável que entra)
            for (int i : t.B) {
                if (t.A[i][entra] > EPSILON) {
                    delta[i] = t.b[i] / t.A[i][entra];
                } else {
                    delta[i] = 999999999; // infinito (se for <= 0 não restringe o crescimento)
                }
            }

            int sai = -1;
            double menorDelta = 999999999;
            
            // Encontra a variável básica que limita mais a entrada (menor razão positiva)
            for (int i : t.B) {
                // O segundo critério "(sai == -1 || i < sai)" é a Regra de Bland pra evitar loop infinito em caso de empate
                if (delta[i] < menorDelta - EPSILON || (Math.abs(delta[i] - menorDelta) < EPSILON && (sai == -1 || i < sai))) {
                    menorDelta = delta[i];
                    sai = i;
                }
            }

            // Se a variável pode crescer infinitamente sem violar restrições
            if (sai == -1 || menorDelta == 999999999) {
                return null; // Problema Ilimitado
            }

            // Atualiza a tabela com a nova base
            t = pivoteamento(t, sai, entra);
        }
    }

    /**
     * Inicializa a tabela para o Simplex. Cria variáveis de folga.
     * Se a origem não for viável (b negativo), roda o Método de Duas Fases (Fase 1).
     * 
     * @param A Matriz de restrições original
     * @param b Vetor de recursos original
     * @param c Vetor de custos/lucros original
     * @return  Tabela pronta para rodar a Fase 2, ou null se o problema for inviável
     */
    static Tabela inicializa(double[][] A, double[] b, double[] c) {
        int m = b.length;
        int n = c.length;

        // Procura a restrição mais "violada" na origem (menor b)
        int minB = 0;
        for (int i = 1; i < m; i++) {
            if (b[i] < b[minB]) minB = i;
        }

        // Se vetor b for todo >= 0, a origem é viável. Pula direto pra fase 2
        if (b[minB] >= 0) {
            int sz = n + m;
            double[][] A2 = new double[sz][sz];
            double[] b2 = new double[sz];
            double[] c2 = new double[sz];
            
            ArrayList<Integer> N = new ArrayList<>();
            ArrayList<Integer> B = new ArrayList<>();
            
            for (int j = 0; j < n; j++) N.add(j);
            
            for (int i = 0; i < m; i++) {
                B.add(n + i); // n+i são as variáveis de folga (slack)
                b2[n + i] = b[i];
                for (int j = 0; j < n; j++) {
                    A2[n + i][j] = A[i][j];
                }
            }
            for (int j = 0; j < n; j++) c2[j] = c[j];
            
            return new Tabela(N, B, A2, b2, c2, 0, sz);
        }

        // ============ FASE 1 (A origem não é viável) ============
        // Adicionamos uma variável artificial x0 que subtrai de todas as restrições
        int x0 = n + m;
        int sz = n + m + 1;

        double[][] auxA = new double[sz][sz];
        double[] auxB = new double[sz];
        double[] auxC = new double[sz];

        for (int i = 0; i < m; i++) {
            auxB[n + i] = b[i];
            for (int j = 0; j < n; j++) {
                auxA[n + i][j] = A[i][j];
            }
            auxA[n + i][x0] = 1.0; // coeficiente do x0 na matriz
        }
        // A nova função objetivo é minimizar x0 (ou maximizar -x0)
        auxC[x0] = -1.0;

        ArrayList<Integer> N = new ArrayList<>();
        ArrayList<Integer> B = new ArrayList<>();
        
        for (int j = 0; j < n; j++) N.add(j);
        N.add(x0);
        for (int i = 0; i < m; i++) B.add(n + i);

        Tabela aux = new Tabela(N, B, auxA, auxB, auxC, 0, sz);

        // Força o x0 a entrar na base no lugar da restrição mais negativa
        // Isso magicamente faz todos os b's ficarem positivos
        int sai = n + minB;
        aux = pivoteamento(aux, sai, x0);
        
        // Resolve o simplex para zerar o x0
        aux = rodarSimplex(aux);
        if (aux == null) return null;

        // Se no ótimo da Fase 1 o x0 for maior que 0, é impossível satisfazer as restrições originais
        double valX0 = aux.B.contains(x0) ? aux.b[x0] : 0;
        if (Math.abs(valX0) > EPSILON) return null; // Inviável

        // Se x0 zerou mas ainda está na base (degeneração), tira ele fazendo um pivoteamento forçado
        if (aux.B.contains(x0)) {
            for (int e : aux.N) {
                if (Math.abs(aux.A[x0][e]) > EPSILON) {
                    aux = pivoteamento(aux, x0, e);
                    break;
                }
            }
        }

        // --- TRANSIÇÃO PARA A FASE 2 ---
        // Removemos o x0 para sempre
        aux.N.remove((Integer) x0);
        for (int i : aux.B) {
            aux.A[i][x0] = 0;
        }
        aux.c[x0] = 0;

        // Restaura a função objetivo original (vetor c) ajustando para a base atual
        double[] c2 = new double[sz];
        double z2 = 0;
        for (int j = 0; j < n; j++) {
            if (aux.N.contains(j)) {
                c2[j] += c[j];
            } else if (aux.B.contains(j)) {
                z2 += c[j] * aux.b[j];
                for (int jj : aux.N) {
                    c2[jj] -= c[j] * aux.A[j][jj];
                }
            }
        }

        aux.c = c2;
        aux.z = z2;

        return aux;
    }

    /**
     * Ponto de entrada do programa. Configura o problema e imprime o resultado.
     */
    public static void main(String[] args) {
        // Exemplo do livro CLRS09, pagina 864
        // Maximizar 3x1 + x2 + 2x3
        // sujeito a:
        //   x1 +  x2 + 3x3 <= 30
        //  2x1 + 2x2 + 5x3 <= 24
        //  4x1 +  x2 + 2x3 <= 36
        //  x1, x2, x3 >= 0
        
        double[][] matrizRestricoes = {
            {1, 1, 3},
            {2, 2, 5},
            {4, 1, 2}
        };
        double[] vetorB = {30, 24, 36};
        double[] funcaoObjetiva = {3, 1, 2}; // vetor C

        Tabela t = inicializa(matrizRestricoes, vetorB, funcaoObjetiva);
        
        if (t == null) {
            System.out.println("Solução Inviável :(");
            return;
        }

        t = rodarSimplex(t);
        
        if (t == null) {
            System.out.println("Solução Ilimitada!");
            return;
        }

        System.out.print("Solução: ");
        for (int i = 0; i < funcaoObjetiva.length; i++) {
            // Se a variável original está na base, pega o valor de b, senão é 0
            double xi = t.B.contains(i) ? t.b[i] : 0;
            System.out.print("x" + (i+1) + " = " + String.format("%.2f", xi) + "  ");
        }
        System.out.println("\nZ Maximo = " + String.format("%.2f", t.z));
    }
}

