import os

import numpy as np
import pandas as pd
from pandas import DataFrame
from scipy.spatial import distance
import tkinter as tk
from tkinter import ttk

reported_symptoms = []
council_history = []
last_inference = tuple()

metrics = sorted([
    "correlation",
    "jaccard",
    "cosine",
    "hamming",
    "jensenshannon",
    "canberra",
    "braycurtis",
    "matching",
    "dice",
    "rogerstanimoto",
    "russellrao",
    "sokalmichener",
    "sokalsneath",
    "sqeuclidean",
    "cityblock",
    "euclidean"
])


def get_data() -> tuple[tuple[tuple[str], tuple[str]], DataFrame]:
    db = pd.read_csv('input/database.csv')

    symptoms = tuple(db.columns[:-1])
    prognostics = tuple(db['prognóstico'].unique())

    print(
        f"No dataset, há {db.last_valid_index()} casos, {len(symptoms)} sintomas para avaliar e "
        f"{len(prognostics)} prognósticos possíveis.\n")

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

    global last_inference
    last_inference = (raw_inference, new_inference)

    try:
        os.mkdir("output")
    except FileExistsError:
        pass

    library.to_csv('output/library.csv', index=False)


def council_decision(council_votes: list[str]) -> dict[str, float]:
    vote_count = {}
    for vote in set(council_votes):
        vote_count[vote] = council_votes.count(vote)
    total = sum(vote_count.values())
    return {vote: (count / total) * 100 for vote, count in vote_count.items()}


def format_decision(vote_count: list[tuple[str, float]]) -> str:
    total = sum([i[1] for i in vote_count])
    text = ""
    for vote, count in vote_count:
        porc = (count / total) * 100
        text += f" • {vote} ({porc:.2f}%)\n"
    return text


def make_prognostic(knowledge_base: pd.DataFrame, symptoms: list[str]) -> (str, str):
    base = knowledge_base.iloc[:, range(knowledge_base.shape[1] - 1)]
    symptom_presences = {f"{s}": 0 for s in knowledge_base.columns[:-1]}

    for symptom in symptoms:
        symptom_presences[symptom] = 1

    distances = {metric: np.zeros(base.shape[0]) for metric in metrics}

    one_hot_symptoms = list(symptom_presences.values())

    estimated_prognostics = {}
    for metric in metrics:
        distances[metric] = distance.cdist(np.array([one_hot_symptoms]), base, metric)

        min_metric_index = np.argsort(distances[metric][0])[0]
        estimated_prognostics[metric] = knowledge_base.iloc[min_metric_index, -1]

    inference_voting_report = council_decision(list(estimated_prognostics.values()))
    council_history.append(inference_voting_report)

    most_voted_inference = max(inference_voting_report.items(), key=lambda x: x[1])[0]

    one_hot_symptoms = [-1 if i == 0 else 1 for i in one_hot_symptoms]
    case = np.append(one_hot_symptoms, most_voted_inference)

    # Acumula o novo conhecimento.
    case = pd.DataFrame(case, list(knowledge_base.columns)).transpose()
    case.iloc[-1, :-1] = [int(i) for i in case.iloc[-1, :-1]]
    commit_knowledge(case, [symptoms.copy(), most_voted_inference])

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
        undo_selections_button.config(state=tk.NORMAL)


def make_prognostic_cb():
    result_label.config(text="⌛ Analisando...")
    tabs.select(tab_id=1)
    result_label.update()
    prg_dict = make_prognostic(knowledge, reported_symptoms)

    most_common_inf = sorted(council_history[-1].items(), key=lambda x: x[1], reverse=True)

    prog_str = "Opinião do conselho médico:\n"
    for m, prog in prg_dict.items():
        prog_str += f" • Dr. {m.capitalize() + ':': <15} {prog}\n"
    prog_str += "\nConclusões:\n" + format_decision(most_common_inf)

    symp_str = '\n'.join([f" • {i.capitalize()}" for i in sorted(reported_symptoms)])
    print(f"\n===\nSintomas:\n{symp_str}\n\n{prog_str}\n===\n")

    make_prognostic_button.config(state=tk.DISABLED)
    save_knowledge_button.config(state=tk.NORMAL)
    result_label.config(text=prog_str)
    result_label.update()


def clean_selection_cb():
    reported_symptoms.clear()
    symptoms_list_box.config(state=tk.NORMAL)
    symptoms_list_box.delete("1.0", tk.END)
    symptoms_list_box.config(state=tk.DISABLED)
    make_prognostic_button.config(state=tk.DISABLED)
    clean_selections_button.config(state=tk.DISABLED)
    undo_selections_button.config(state=tk.DISABLED)
    result_label.config(text="")


def undo_selection_cb():
    reported_symptoms.pop()
    symptoms_list_box.config(state=tk.NORMAL)
    symptoms_list_box.delete("1.0", tk.END)
    for symptom in reported_symptoms:
        symptoms_list_box.insert(tk.END, f"• {symptom.capitalize()}\n")
    symptoms_list_box.config(state=tk.DISABLED)
    if len(reported_symptoms) == 0:
        make_prognostic_button.config(state=tk.DISABLED)
        clean_selections_button.config(state=tk.DISABLED)
        undo_selections_button.config(state=tk.DISABLED)
    else:
        make_prognostic_button.config(state=tk.NORMAL)
    result_label.config(text="")


def save_coherent_knowledge_cb():
    database = pd.read_csv('input/database.csv')

    database = pd.concat([database, last_inference[1]], ignore_index=True)
    database.to_csv('input/database.csv', index=False)

    save_knowledge_button.config(state=tk.DISABLED)


if __name__ == '__main__':
    (all_symptoms, all_prognostics), knowledge = get_data()

    root = tk.Tk()
    root.title('Sistema de Prognóstico de Doenças')
    style = ttk.Style()
    style.theme_use("clam")
    style.configure('tabs.TNotebook.Tab', font="TkFixedFont")

    tabs = ttk.Notebook(root, style="tabs.TNotebook")
    tabs.pack(fill="both", expand=True)

    main_frame = ttk.Frame(tabs)

    selection_frame = ttk.Frame(main_frame)
    selected_symptom_var = tk.StringVar(selection_frame)
    selected_symptom_var.set("Selecione um Sintoma")
    selected_symptom_var.trace_add("write", lambda x, y, z: add_symptom_button.config(state=tk.NORMAL))
    symptoms_list_combobox = ttk.Combobox(selection_frame, textvariable=selected_symptom_var, width=40,
                                          values=sorted([s.capitalize() for s in all_symptoms]))
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
    make_prognostic_button.pack(pady=10, padx=5, side=tk.RIGHT)

    clean_selections_button = ttk.Button(prognostic_frame, text="Limpar", state=tk.DISABLED,
                                         command=clean_selection_cb)
    clean_selections_button.pack(pady=10, padx=5, side=tk.LEFT)

    undo_selections_button = ttk.Button(prognostic_frame, text="Desfazer", state=tk.DISABLED,
                                        command=undo_selection_cb)
    undo_selections_button.pack(pady=10, padx=5, side=tk.LEFT)

    prognostic_frame.pack()

    tabs.add(main_frame, text="🧩 Inferência")

    knowledge_frame = ttk.Frame(tabs)
    ttk.Label(knowledge_frame, text="").pack()
    result_label = ttk.Label(knowledge_frame, text="", font="TkFixedFont")
    result_label.pack(padx=5, pady=0)

    save_knowledge_button = ttk.Button(knowledge_frame, text="Salvar Prognóstico", command=save_coherent_knowledge_cb)
    save_knowledge_button.pack(padx=10, pady=10)
    save_knowledge_button.config(state=tk.DISABLED)
    ttk.Label(knowledge_frame, text="").pack()

    tabs.add(knowledge_frame, text="🧠 Verificação e Conhecimento")

    # Executar aplicação
    root.mainloop()
