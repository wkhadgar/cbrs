# Import
import numpy as np
import pandas as pd
from scipy.spatial import distance
import matplotlib.pyplot as plt
import seaborn as sn


def get_data():
    lib, test = pd.read_csv('input/database.csv'), pd.read_csv("input/test_data.csv")

    print('\n> Dataset inicial em one-hot encoding:')
    print(f'\n{lib}')
    print(f'\n{test}')

    return lib, test


# Main
def main():
    library, cases = get_data()

    # A base e os problemas não contém as soluções.
    base = library.iloc[:, range(library.shape[1] - 1)]
    problems = cases.iloc[:, range(cases.shape[1] - 1)]

    print('\n> Calculando\n')
    for i in range(problems.shape[0]):

        # Obter a matriz de covariância inversa para os casos de base
        inverse_covariance_matrix = np.linalg.pinv(base.cov())

        case_row = problems.loc[i, :]
        distances = np.zeros(base.shape[0])

        for j in range(base.shape[0]):
            base_row = base.loc[j, :]

            # Calcular a distância Mahalanobis entre a linha de casos e os casos de base.
            distances[j] = distance.mahalanobis(case_row, base_row, inverse_covariance_matrix)

        min_distance_row = np.argmin(distances)

        # Obtém a solução com base no índice da distância mínima encontrada e anexa à biblioteca principal dos casos.
        case = np.append(problems.iloc[i, :], library.iloc[min_distance_row, -1])
        print(
            f'> Para o caso {i}: '
            f'{[problems.columns[ind[0]] for ind in [(ti, tl) for ti, tl in [(k, label) for k, label in enumerate(problems.iloc[i, :].to_numpy())] if tl == 1]]},'
            f' a solução é {case[-1]}')

        # Acumula o novo conhecimento.
        case = pd.DataFrame(case, list(library.columns)).transpose()
        case.iloc[-1, :-1] = [int(i) for i in case.iloc[-1, :-1]]
        library = pd.concat([library, case], ignore_index=True)

        # Dados para validação.
        # sn.heatmap(np.cov(base, bias=True), annot=True, fmt='g')
        # plt.gcf().set_size_inches(12, 6)
        # plt.title(f'Mapa de calor da covariância #{i} \n Casos base salvos {j} - Base para solucionar o problema {i}')
        # plt.savefig(f'output/covariance_heat_map_{i}.png', bbox_inches='tight')
        # plt.close()

        base = library.iloc[:, range(library.shape[1] - 1)]  # Exclui a solução novamente.

    print('\n> Conhecimento final:')
    print(f'\n{library}')

    library.to_csv('output/library.csv', index=False)


if __name__ == '__main__':
    main()