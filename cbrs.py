import numpy as np
import pandas as pd
from pandas import DataFrame
from scipy.spatial import distance
import tkinter as tk
from tkinter import ttk

reported_symptoms = []
inferences_list = []
inf_checkboxes = []


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


def distance_similarity(distances, target_index) -> float:
    sorted_indexes = np.argsort(distances)
    return 1 - (distances[target_index] / distances[sorted_indexes[-1]])


def make_prognostic(knowledge: pd.DataFrame, symptoms: list[str]) -> (str, str):
    base = knowledge.iloc[:, range(knowledge.shape[1] - 1)]
    symptom_presences = {f"{s}": 0 for s in knowledge.columns[:-1]}

    print(f"\n\n>Caso: {' + '.join(symptoms)}")
    for symptom in symptoms:
        symptom_presences[symptom] = 1

    metrics = ["correlation", "jaccard"]
    distances = {metric: np.zeros(base.shape[0]) for metric in metrics}

    one_hot_symptoms = list(symptom_presences.values())

    best_inf = "NULL"
    best_ratio = 0.0
    estimated_prognostics = {}
    for metric in metrics:
        distances[metric] = distance.cdist(np.array([one_hot_symptoms]), base, metric)

        min_metric_index = np.argsort(distances[metric][0])[0]
        inference = knowledge.iloc[min_metric_index, -1]
        closeness = distance_similarity(distances[metric][0], min_metric_index)
        estimated_prognostics[metric] = {"inference": inference, "closeness": closeness}

        if closeness > best_ratio:
            best_ratio = closeness
            best_inf = inference

        print(f"\tPor {metric} -> {inference} ({closeness * 100:.2f})")

    case = np.append(one_hot_symptoms, best_inf)

    # Acumula o novo conhecimento.
    case = pd.DataFrame(case, list(knowledge.columns)).transpose()
    case.iloc[-1, :-1] = [int(i) for i in case.iloc[-1, :-1]]
    commit_knowledge(case, [symptoms.copy(), best_inf])

    return estimated_prognostics


def add_symptom_cb():
    symptom = selected_symptom_var.get().lower().strip()

    if (symptom not in reported_symptoms) and (symptom in all_symptoms):
        reported_symptoms.append(symptom.lower().strip())
        symptoms_list_box.config(state=tk.NORMAL)
        symptoms_list_box.insert(tk.END, f"• {symptom.capitalize()}\n")
        symptoms_list_box.config(state=tk.DISABLED)

        make_prognostic_button.config(state=tk.NORMAL)
        clean_selections_button.config(state=tk.NORMAL)
    else:
        make_prognostic_button.config(state=tk.DISABLED)
        clean_selections_button.config(state=tk.DISABLED)


def make_prognostic_cb():
    result_label.config(text="> Analisando... ⌛")
    result_label.update()
    prg_dict = make_prognostic(knowledge, reported_symptoms)

    prog_str = ""
    for m, pr in prg_dict.items():
        prog_str += f"    Por {m.capitalize()} → {pr['inference']} ({pr['closeness'] * 100:.2f}%) ou \n"

    result_label.config(text=f"> Para o caso avaliado, o prognóstico é :\n"
                             f"{prog_str[:-4]}")
    make_prognostic_button.config(state=tk.DISABLED)


def clean_selection_cb():
    reported_symptoms.clear()
    symptoms_list_box.config(state=tk.NORMAL)
    symptoms_list_box.delete("1.0", tk.END)
    symptoms_list_box.config(state=tk.DISABLED)
    make_prognostic_button.config(state=tk.DISABLED)
    clean_selections_button.config(state=tk.DISABLED)
    result_label.config(text="")


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
    update_history_cb()


def update_history_cb():
    for widget in history_frame.winfo_children():
        widget.destroy()

    for inf in inferences_list:
        check_var = tk.BooleanVar()
        check_frame = ttk.Frame(history_frame)
        check_button = ttk.Checkbutton(check_frame, text=f"{' + '.join(inf[0][0])} → {inf[0][1]}",
                                       variable=check_var)
        check_button.pack(side=tk.LEFT)
        check_frame.pack(fill=tk.BOTH)
        inf_checkboxes.append(check_var)


if __name__ == '__main__':
    (all_symptoms, all_prognostics), knowledge = get_data()

    root = tk.Tk()
    root.title('Sistema de Prognóstico de Doenças')
    style = ttk.Style()
    style.theme_use("clam")

    tabs = ttk.Notebook(root)
    tabs.pack(fill="both", expand=True)

    main_frame = ttk.Frame(tabs)

    selection_frame = ttk.Frame(main_frame)
    selected_symptom_var = tk.StringVar(selection_frame)
    selected_symptom_var.set("Selecione um Sintoma")
    selected_symptom_var.trace_add("write", lambda x, y, z: add_symptom_button.config(state=tk.NORMAL))
    symptoms_list_combobox = ttk.Combobox(selection_frame, textvariable=selected_symptom_var, width=40,
                                          values=list(map(str.capitalize, sorted(all_symptoms))))
    symptoms_list_combobox.bind("<Button-1>", lambda e:
                                symptoms_list_combobox.focus() or symptoms_list_combobox.event_generate('<Down>'))
    symptoms_list_combobox.bind("<FocusOut>", lambda e: symptoms_list_combobox.selection_clear())
    symptoms_list_combobox.pack(padx=10, pady=5, side=tk.LEFT)

    add_symptom_button = ttk.Button(selection_frame, text="Adicionar", state=tk.DISABLED, command=add_symptom_cb)
    add_symptom_button.pack(pady=10, side=tk.RIGHT)
    selection_frame.pack()

    ttk.Label(main_frame, text="Sintomas relatados:", font=("Arial", 10, "bold")).pack()
    symptoms_list_box = tk.Text(main_frame, height=10, width=40, state=tk.DISABLED)
    symptoms_list_box.pack(side=tk.TOP, padx=10, pady=1)

    prognostic_frame = ttk.Frame(main_frame)
    make_prognostic_button = ttk.Button(prognostic_frame, text="Realizar Prognóstico", state=tk.DISABLED,
                                        command=make_prognostic_cb)
    make_prognostic_button.pack(pady=10, padx=30, side=tk.RIGHT)

    clean_selections_button = ttk.Button(prognostic_frame, text="Limpar Seleção", state=tk.DISABLED,
                                         command=clean_selection_cb)
    clean_selections_button.pack(pady=10, padx=30, side=tk.LEFT)
    prognostic_frame.pack()

    result_label = ttk.Label(main_frame, text="")
    result_label.pack(padx=10, pady=1)
    tabs.add(main_frame, text="🧩 Inferência")

    knowledge_frame = ttk.Frame(tabs)
    history_frame = ttk.Frame(knowledge_frame)
    history_frame.pack()
    ttk.Label(knowledge_frame, text="Selecione as inferências coerentes.").pack()
    knowledge_frame.bind("<FocusIn>", lambda e: update_history_cb())

    ttk.Button(knowledge_frame, text="Salvar Selecionadas", command=save_coherent_knowledge_cb).pack(padx=10, pady=10)

    tabs.add(knowledge_frame, text="🧠 Verificação e Conhecimento")

    # Executar aplicação
    root.mainloop()