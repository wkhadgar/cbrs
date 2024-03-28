# Import
import numpy as np
import pandas as pd
from pandas import DataFrame
from scipy.spatial import distance
import tkinter as tk
from tkinter import ttk


def get_data() -> tuple[tuple[tuple[str], tuple[str]], DataFrame]:
    db = pd.read_csv('input/database.csv')

    symptoms = tuple(db.columns[:-1])
    prognostics = tuple(db['prognóstico'].unique())

    print(f"No dataset, há {len(symptoms)} sintomas para avaliar e {len(prognostics)} prognósticos possíveis.\n")

    print("Sintomas:")
    for s in sorted(symptoms):
        print(f"\t• {s}")

    print("\nPrognósticos:")
    for p in sorted(prognostics):
        print(f"\t• {p}")

    return (symptoms, prognostics), db


# Main
def make_prognostic(library: pd.DataFrame, symptoms: list[str]) -> str:
    base = library.iloc[:, range(library.shape[1] - 1)]
    symptom_presences = {f"{s}": 0 for s in library.columns[:-1]}

    for symptom in symptoms:
        symptom_presences[symptom] = 1

    print('\n> Calculando...\n')

    # Obter a matriz de covariância inversa para os casos de base
    inverse_covariance_matrix = np.linalg.pinv(base.cov())

    distances = np.zeros(base.shape[0])
    for i in range(base.shape[0]):
        base_symptoms_case = base.loc[i, :]

        # Calcular a distância Mahalanobis entre a linha de casos e os casos de base.
        distances[i] = distance.mahalanobis(list(symptom_presences.values()), base_symptoms_case,
                                            inverse_covariance_matrix)

    min_distance_row = np.argmin(distances)

    # Obtém a solução com base no índice da distância mínima encontrada e anexa à biblioteca principal dos casos.
    estimated_prognostic = library.iloc[min_distance_row, -1]
    case = np.append(list(symptom_presences.values()), estimated_prognostic)
    print(f"> Para o caso avaliado: {' + '.join(symptoms)}, o prognóstico é {case[-1]}")

    # Acumula o novo conhecimento.
    case = pd.DataFrame(case, list(library.columns)).transpose()
    case.iloc[-1, :-1] = [int(i) for i in case.iloc[-1, :-1]]
    library = pd.concat([library, case], ignore_index=True)

    library.to_csv('output/library.csv', index=False)

    return estimated_prognostic


def add_symptom_cb():
    make_prognostic_button.config(state=tk.NORMAL)
    clean_selections_button.config(state=tk.NORMAL)
    symptom = selected_symptom.get().lower().strip()
    if (symptom not in case_symptoms) and (symptom in sym):
        case_symptoms.append(symptom.lower().strip())
        symptoms_list.config(state=tk.NORMAL)
        symptoms_list.insert(tk.END, f"• {symptom.capitalize()}\n")
        symptoms_list.config(state=tk.DISABLED)


def make_prognostic_cb():
    result_label.config(text="Calculando...")
    pr = make_prognostic(lib, case_symptoms)
    result_label.config(text=f"> Para o caso avaliado, o prognóstico é {pr}")


def clean_selection_cb():
    case_symptoms.clear()
    symptoms_list.config(state=tk.NORMAL)
    symptoms_list.delete("1.0", tk.END)
    symptoms_list.config(state=tk.DISABLED)
    make_prognostic_button.config(state=tk.DISABLED)
    clean_selections_button.config(state=tk.DISABLED)
    result_label.config(text="")


def open_menu(event):
    menu_symptoms.focus()
    menu_symptoms.event_generate('<Down>')


def close_menu(event):
    menu_symptoms.selection_clear()


if __name__ == '__main__':
    (sym, prog), lib = get_data()

    case_symptoms = []

    # Inicialização da janela
    root = tk.Tk()
    root.title('Sistema de Prognóstico de Doenças')
    style = ttk.Style()
    style.theme_use("clam")

    main_frame = ttk.Frame(root)
    selection_frame = ttk.Frame(main_frame)
    selected_symptom = tk.StringVar(selection_frame)  # Menu drop-down para selecionar sintoma
    selected_symptom.set("Selecione um Sintoma")
    selected_symptom.trace_add("write", lambda x, y, z: add_symptom_button.config(state=tk.NORMAL))
    menu_symptoms = ttk.Combobox(selection_frame, textvariable=selected_symptom, width=40,
                                 values=list(map(str.capitalize, sorted(sym))))
    menu_symptoms.bind("<Button-1>", open_menu)
    menu_symptoms.bind("<FocusOut>", close_menu)
    menu_symptoms.pack(padx=10, pady=5, side=tk.LEFT)

    add_symptom_button = ttk.Button(selection_frame, text="Adicionar", state=tk.DISABLED, command=add_symptom_cb)
    add_symptom_button.pack(pady=10, side=tk.RIGHT)
    selection_frame.pack()

    symptoms_list_label = ttk.Label(main_frame, text="Sintomas relatados:", font=("Arial", 10, "bold"))
    symptoms_list_label.pack()

    symptoms_list = tk.Text(main_frame, height=10, width=40, state=tk.DISABLED)
    symptoms_list.pack(side=tk.TOP, padx=10, pady=1)

    prognostic_frame = ttk.Frame(main_frame)
    make_prognostic_button = ttk.Button(prognostic_frame, text="Realizar Prognóstico", state=tk.DISABLED,
                                        command=make_prognostic_cb)
    make_prognostic_button.pack(pady=10, padx=30, side=tk.RIGHT)

    clean_selections_button = ttk.Button(prognostic_frame, text="Limpar Seleção", state=tk.DISABLED,
                                         command=clean_selection_cb)
    clean_selections_button.pack(pady=10, padx=30, side=tk.LEFT)
    prognostic_frame.pack()

    # Rótulo para exibir resultado da verificação
    result_label = ttk.Label(main_frame, text="")
    result_label.pack(padx=10, pady=1)
    main_frame.pack()

    # Executar aplicação
    root.mainloop()
