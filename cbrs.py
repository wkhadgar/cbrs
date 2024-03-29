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

    print("\nPrognósticos avaliáveis:")
    for p in sorted(prognostics):
        print(f"\t• {p}")

    return (symptoms, prognostics), db


def commit_knowledge(new_inference: pd.DataFrame, raw_inference: list):
    try:
        library = pd.read_csv('output/library.csv')
        library = pd.concat([library, new_inference], ignore_index=True)
    except FileNotFoundError:
        library = new_inference

    inferences_list.append((raw_inference, new_inference))
    library.to_csv('output/library.csv', index=False)


def distance_similarity(distances, target_index):
    sorted_indexes = np.argsort(distances)
    return 1 - (distances[target_index] / distances[sorted_indexes[-1]])


def make_prognostic(knowledge: pd.DataFrame, symptoms: list[str]) -> (str, str):
    base = knowledge.iloc[:, range(knowledge.shape[1] - 1)]
    symptom_presences = {f"{s}": 0 for s in knowledge.columns[:-1]}

    for symptom in symptoms:
        symptom_presences[symptom] = 1

    print('\n> Calculando...\n')

    # Obter a matriz de covariância inversa para os casos de base
    inverse_covariance_matrix = np.linalg.pinv(base.cov())

    distances = np.zeros(base.shape[0])
    distances_alt = np.zeros(base.shape[0])
    one_hot_symptoms = list(symptom_presences.values())
    for i in range(base.shape[0]):
        base_symptoms_case = base.loc[i, :]

        distances[i] = distance.correlation(one_hot_symptoms, base_symptoms_case)
        distances_alt[i] = distance.mahalanobis(one_hot_symptoms, base_symptoms_case, inverse_covariance_matrix)

    distances_index_sorted = np.argsort(distances)
    distances_index_sorted_alt = np.argsort(distances_alt)
    min_distance_row = distances_index_sorted[0]
    alt_min_distance_row = distances_index_sorted_alt[0]

    # Obtém a solução com base no índice da distância mínima encontrada e anexa à biblioteca principal dos casos.
    estimated_prognostic: str = knowledge.iloc[min_distance_row, -1]
    alt_estimated_prognostic: str = knowledge.iloc[alt_min_distance_row, -1]
    case = np.append(list(symptom_presences.values()), estimated_prognostic)

    estimated_prognostic_sureness = distance_similarity(distances, min_distance_row) * 100
    alt_estimated_prognostic_sureness = distance_similarity(distances_alt, alt_min_distance_row) * 100

    print(
        f"> Para o caso avaliado: {' + '.join(symptoms)}, o prognóstico é "
        f"{estimated_prognostic}:{estimated_prognostic_sureness:.2f}% ou "
        f"{alt_estimated_prognostic}:{alt_estimated_prognostic_sureness:.2f}%")

    # Acumula o novo conhecimento.
    case = pd.DataFrame(case, list(knowledge.columns)).transpose()
    case.iloc[-1, :-1] = [int(i) for i in case.iloc[-1, :-1]]
    commit_knowledge(case, [symptoms.copy(), estimated_prognostic])

    return (estimated_prognostic, estimated_prognostic_sureness), (
        alt_estimated_prognostic, alt_estimated_prognostic_sureness)


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
    result_label.config(text="> Analisando...")
    result_label.update()
    pr, alt_pr = make_prognostic(lib, case_symptoms)
    result_label.config(text=f"> Para o caso avaliado, o prognóstico é {pr[1]}%: {pr[0]} ou {alt_pr[1]}%: {alt_pr[0]}")
    make_prognostic_button.config(state=tk.DISABLED)


def clean_selection_cb():
    case_symptoms.clear()
    symptoms_list.config(state=tk.NORMAL)
    symptoms_list.delete("1.0", tk.END)
    symptoms_list.config(state=tk.DISABLED)
    make_prognostic_button.config(state=tk.DISABLED)
    clean_selections_button.config(state=tk.DISABLED)
    result_label.config(text="")


def open_menu_cb():
    menu_symptoms.focus()
    menu_symptoms.event_generate('<Down>')


def close_menu_cb():
    menu_symptoms.selection_clear()


def save_coherent_knowledge_cb():
    validated_knowledge = pd.DataFrame()
    database = pd.read_csv('input/database.csv')

    for i, inf in enumerate(inferences_list):
        if inf_checkboxes[i].get():
            validated_knowledge = pd.concat([validated_knowledge, inf[1]], ignore_index=True)

    if not validated_knowledge.empty:
        database = pd.concat([database, validated_knowledge], ignore_index=True)
        database.to_csv('input/database.csv', index=False)

    inferences_list.clear()
    inf_checkboxes.clear()
    update_history()


def update_history():
    for widget in history_frame.winfo_children():
        widget.destroy()

    ttk.Label(history_frame, text="Selecione as inferências coerentes.").pack(fill=tk.BOTH)
    for inf in inferences_list:
        check_var = tk.BooleanVar()
        check_frame = ttk.Frame(history_frame)
        check_button = ttk.Checkbutton(check_frame, text=f"{' + '.join(inf[0][0])} → {inf[0][1]}",
                                       variable=check_var)
        check_button.pack(side=tk.LEFT)
        check_frame.pack(fill=tk.BOTH)
        inf_checkboxes.append(check_var)


if __name__ == '__main__':
    (sym, prog), lib = get_data()

    case_symptoms = []
    inferences_list = []
    inf_checkboxes = []

    # Inicialização da janela
    root = tk.Tk()
    root.title('Sistema de Prognóstico de Doenças')
    style = ttk.Style()
    style.theme_use("clam")

    tabs = ttk.Notebook(root)
    tabs.pack(fill="both", expand=True)

    main_frame = ttk.Frame(tabs)

    selection_frame = ttk.Frame(main_frame)
    selected_symptom = tk.StringVar(selection_frame)  # Menu drop-down para selecionar sintoma
    selected_symptom.set("Selecione um Sintoma")
    selected_symptom.trace_add("write", lambda x, y, z: add_symptom_button.config(state=tk.NORMAL))
    menu_symptoms = ttk.Combobox(selection_frame, textvariable=selected_symptom, width=40,
                                 values=list(map(str.capitalize, sorted(sym))))
    menu_symptoms.bind("<Button-1>", lambda e: open_menu_cb())
    menu_symptoms.bind("<FocusOut>", lambda e: close_menu_cb())
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
    tabs.add(main_frame, text="Inferência")

    # Inicializar a lista de dados temporários
    knowledge_frame = ttk.Frame(tabs)
    history_frame = ttk.Frame(knowledge_frame)
    history_frame.pack()
    knowledge_frame.bind("<FocusIn>", lambda e: update_history())

    save_coherent_knowledge_button = ttk.Button(knowledge_frame, text="Salvar Selecionadas",
                                                command=save_coherent_knowledge_cb)
    save_coherent_knowledge_button.pack(padx=10, pady=10)

    tabs.add(knowledge_frame, text="Verificação e Conhecimento")

    # Executar aplicação
    root.mainloop()
