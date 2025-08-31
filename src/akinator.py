

# Base de datos de los personajes
database = [
    {"name": "Iron Man", "real": False, "gender": "male"  },
    {"name": "Benito Juarez", "real": True, "gender": "male"},
    {"name": "Cleopatra", "real": True, "gender": "female"},
    {"name": "Wonder", "real": False, "gender": "female"}
]

# Lista de preguntas
questions = [
    {"question": "Tu personaje es una persona real?", "property": "real", "expected_answer":True},
    {"question": "Tu personaje es hombre?", "property": "gender", "expected_answer":"male"},
    {"question": "Tu personaje es mujer?", "property": "gender", "expected_answer":"female"}
]

def process_affirmative_answer(property, expected_answer):
    def filter_func(item):
        return item[property] == expected_answer
    return filter_func


print("BIENVENIDO A AKINATOR")

for q in questions:
    ## Realiza una pregunta
    answer = input(q["question"] + " (yes/no): ")

    ## Toma la respuesta y filtra la base de datos
    answer = answer.lower()

    # Si la respuesta es afirmativa, filtra la base de datos
    if answer == "yes":
        lol = filter(process_affirmative_answer(q["property"], q["expected_answer"]), database)
        database = list(lol)

    # Si la respuesta es negativa, filtra la base de datos
    elif answer == "no":
        lol = filter(lambda item: item[q["property"]] != q["expected_answer"], database)
        database = list(lol)

    # Si solo queda un personaje, termina el juego
    if( len(database) == 1):
        break
    
    print("Quedan " + str(len(database)) + " personajes")

print("Tu personaje es: " + database[0]["name"])
